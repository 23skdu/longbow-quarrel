package engine

import (
	"sync"

	"github.com/23skdu/longbow-quarrel/internal/device"
)

// ActivationLog stores layer-by-layer activations for debugging
type ActivationLog struct {
	Prompt      string             `json:"prompt"`
	Tokens      []int              `json:"tokens"`
	Embedding   []float32          `json:"embedding"` // First 100 values
	Layers      []LayerLog         `json:"layers"`
	FinalLogits map[string]float32 `json:"final_logits"` // String keys for JSON
}

// LayerLog captures activations for a single transformer layer
type LayerLog struct {
	Idx        int       `json:"idx"`
	QMax       float32   `json:"q_max"`
	KMax       float32   `json:"k_max"`
	VMax       float32   `json:"v_max"`
	AttnOutMax float32   `json:"attn_out_max"`
	FFNOutMax  float32   `json:"ffn_out_max"`
	QSample    []float32 `json:"q_sample"` // First 10 values
	KSample    []float32 `json:"k_sample"`
	VSample    []float32 `json:"v_sample"`

	// NaN/Inf detection
	QNaNCount    int `json:"q_nan_count"`
	QInfCount    int `json:"q_inf_count"`
	KNaNCount    int `json:"k_nan_count"`
	KInfCount    int `json:"k_inf_count"`
	VNaNCount    int `json:"v_nan_count"`
	VInfCount    int `json:"v_inf_count"`
	AttnNaNCount int `json:"attn_nan_count"`
	AttnInfCount int `json:"attn_inf_count"`
	FFNNaNCount  int `json:"ffn_nan_count"`
	FFNInfCount  int `json:"ffn_inf_count"`
}

// ActivationLogger manages activation logging during inference
type ActivationLogger struct {
	enabled bool
	log     *ActivationLog
}

// MambaWeights holds the weights for a single Mamba/SSM layer
type MambaWeights struct {
	A            *device.Tensor
	D            *device.Tensor
	Conv1dWeight *device.Tensor
	Conv1dBias   *device.Tensor
	DTWeight     *device.Tensor
	DTBias       *device.Tensor
	NormWeight   *device.Tensor
	NormBias     *device.Tensor
	OutWeight    *device.Tensor
	InWeight     *device.Tensor
}

type MambaState struct {
	ConvState *device.Tensor
	SSMState  *device.Tensor
}

// MambaLayer executes a Mamba/SSM block
type MambaLayer struct {
	Index   int
	Weights *MambaWeights
}

func (w *MambaWeights) Free() {
	if w == nil {
		return
	}
	if w.A != nil {
		w.A.Free()
	}
	if w.D != nil {
		w.D.Free()
	}
	if w.Conv1dWeight != nil {
		w.Conv1dWeight.Free()
	}
	if w.Conv1dBias != nil {
		w.Conv1dBias.Free()
	}
	if w.DTWeight != nil {
		w.DTWeight.Free()
	}
	if w.DTBias != nil {
		w.DTBias.Free()
	}
	if w.NormWeight != nil {
		w.NormWeight.Free()
	}
	if w.NormBias != nil {
		w.NormBias.Free()
	}
	if w.OutWeight != nil {
		w.OutWeight.Free()
	}
	if w.InWeight != nil {
		w.InWeight.Free()
	}
}

func (s *MambaState) Free() {
	if s == nil {
		return
	}
	if s.ConvState != nil {
		s.ConvState.Free()
	}
	if s.SSMState != nil {
		s.SSMState.Free()
	}
}

// QualityEvaluator provides metrics for evaluating generated text quality
type QualityEvaluator struct {
	tokenizer interface {
		Encode(text string) []int
		Decode(ids []int) string
	}
}

type SequenceStatus int

const (
	SequenceStatusPending SequenceStatus = iota
	SequenceStatusRunning
	SequenceStatusComplete
	SequenceStatusCancelled
)

type Sequence struct {
	ID        uint64
	PromptLen int
	MaxTokens int
	Pos       int
	Status    SequenceStatus
	Tokens    []int
	Config         SamplerConfig
	Result         chan []int
	Err            chan error
	TokenCallback  func(int)
	LogitsCallback func([]float32)
	AdapterID      string // Active LoRA adapter for this sequence
	PrefillCompleted bool // Set when the full prompt has been ingested into KV cache
	mu             sync.RWMutex
}

type SequenceManager struct {
	sequences map[uint64]*Sequence
	mu        sync.RWMutex
	counter   uint64
}

func NewSequenceManager() *SequenceManager {
	return &SequenceManager{
		sequences: make(map[uint64]*Sequence),
	}
}

func (sm *SequenceManager) NewSequence(promptLen int) *Sequence {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	sm.counter++
	seq := &Sequence{
		ID:        sm.counter,
		PromptLen: promptLen,
		Pos:       0,
		Status:    SequenceStatusRunning,
	}
	sm.sequences[sm.counter] = seq
	return seq
}

func (sm *SequenceManager) GetSequence(id uint64) (*Sequence, bool) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	seq, ok := sm.sequences[id]
	return seq, ok
}

func (sm *SequenceManager) FreeSequence(id uint64) {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	delete(sm.sequences, id)
}

func (sm *SequenceManager) SetStatus(id uint64, status SequenceStatus) {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	if seq, ok := sm.sequences[id]; ok {
		seq.Status = status
	}
}


type LlamaWeights struct {
	TokenEmb *device.Tensor // vocab x dim

	// Layers
	AttnQ []*device.Tensor
	AttnK []*device.Tensor
	AttnV []*device.Tensor
	AttnO []*device.Tensor

	// AttnNorm   []*device.Tensor
	AttnNorm []*device.Tensor // Re-added just in case

	// Gemma4-specific Q/K normalization (applied before Q/K projections)
	AttnQNorm []*device.Tensor
	AttnKNorm []*device.Tensor

	FfnGate []*device.Tensor
	FfnDown []*device.Tensor
	FfnUp   []*device.Tensor

	FfnNorm []*device.Tensor

	// Final
	OutputNorm *device.Tensor
	Output     *device.Tensor // vocab x dim (often shared with TokenEmb?)

	// Mamba/SSM Layers (Hybrid Support)
	Mamba []*MambaWeights

	// MOE Layers (Hybrid Support)
	MOE []*MOELayerWeights
}

func (w *LlamaWeights) Free() {
	if w == nil {
		return
	}

	freeTensor := func(t *device.Tensor) {
		if t == nil {
			return
		}
		// Skip BufferID check - handle by pointer
		t.Free()
	}

	freeSlices := func(slices ...[]*device.Tensor) {
		for _, slice := range slices {
			for _, t := range slice {
				freeTensor(t)
			}
		}
	}

	freeTensor(w.TokenEmb)
	freeTensor(w.Output)
	freeTensor(w.OutputNorm)

	freeSlices(w.AttnQ, w.AttnK, w.AttnV, w.AttnO, w.AttnNorm, w.AttnQNorm, w.AttnKNorm)
	freeSlices(w.FfnGate, w.FfnDown, w.FfnUp, w.FfnNorm)

	for _, m := range w.Mamba {
		if m != nil {
			m.Free()
		}
	}
	for _, m := range w.MOE {
		if m != nil {
			m.Free()
		}
	}
}

// MOEExpertWeights holds per-expert FFN weights for a single layer
type MOEExpertWeights struct {
	// Expert-specific weights (3D tensors stored as 2D: [hidden_dim * num_experts, dim])
	FfnGateExperts *device.Tensor // Gate projection for all experts
	FfnUpExperts   *device.Tensor // Up projection for all experts
	FfnDownExperts *device.Tensor // Down projection for all experts

	// 3D tensor metadata for indexing into flattened 2D tensors
	NumExperts int // Number of experts (e.g., 128 for Nemotron)
	HiddenDim  int // Hidden dimension per expert (e.g., 1856)
	Dim        int // Input/output dimension (e.g., 2688)
}

func (w *MOEExpertWeights) Free() {
	if w == nil {
		return
	}
	if w.FfnGateExperts != nil {
		w.FfnGateExperts.Free()
	}
	if w.FfnUpExperts != nil {
		w.FfnUpExperts.Free()
	}
	if w.FfnDownExperts != nil {
		w.FfnDownExperts.Free()
	}
}

// MOESharedWeights holds shared expert weights for a single layer
type MOESharedWeights struct {
	// Shared expert weights (always active, 2D tensors)
	FfnGateShared *device.Tensor // Gate projection for shared expert
	FfnUpShared   *device.Tensor // Up projection for shared expert
	FfnDownShared *device.Tensor // Down projection for shared expert
}

func (w *MOESharedWeights) Free() {
	if w == nil {
		return
	}
	if w.FfnGateShared != nil {
		w.FfnGateShared.Free()
	}
	if w.FfnUpShared != nil {
		w.FfnUpShared.Free()
	}
	if w.FfnDownShared != nil {
		w.FfnDownShared.Free()
	}
}

// MOERouterWeights holds routing/gating weights for a single layer
type MOERouterWeights struct {
	GateInput      *device.Tensor // Router input projection [dim, num_experts]
	ExpertProbBias *device.Tensor // Expert probability bias [num_experts]
}

func (w *MOERouterWeights) Free() {
	if w == nil {
		return
	}
	if w.GateInput != nil {
		w.GateInput.Free()
	}
	if w.ExpertProbBias != nil {
		w.ExpertProbBias.Free()
	}
}

// MOELayerWeights combines all MOE components for a single layer
type MOELayerWeights struct {
	Experts *MOEExpertWeights
	Shared  *MOESharedWeights
	Router  *MOERouterWeights
}

func (w *MOELayerWeights) Free() {
	if w == nil {
		return
	}
	if w.Experts != nil {
		w.Experts.Free()
	}
	if w.Shared != nil {
		w.Shared.Free()
	}
	if w.Router != nil {
		w.Router.Free()
	}
}

