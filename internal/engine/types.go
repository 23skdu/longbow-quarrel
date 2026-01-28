//go:build darwin && metal

package engine

import (
	"encoding/json"
	"os"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

type ActivationTrace struct {
	LayerName string    `json:"layer_name"`
	LayerIdx  int       `json:"layer_idx"`
	Max       float32   `json:"max"`
	Min       float32   `json:"min"`
	Mean      float32   `json:"mean"`
	RMS       float32   `json:"rms"`
	Zeros     int       `json:"zeros"`
	NANs      int       `json:"nans"`
	Infs      int       `json:"infs"`
	Sample    []float32 `json:"sample"`
}

type ActivationTraceTracker struct {
	NumLayers int               `json:"num_layers"`
	Traces    []ActivationTrace `json:"traces"`
	enabled   bool
}

const (
	CollapseThreshold   = 0.00001
	SaturationThreshold = 10000.0
)

type LlamaWeights struct {
	TokenEmb *device.Tensor // vocab x dim

	// Layers
	AttnQ []*device.Tensor
	AttnK []*device.Tensor
	AttnV []*device.Tensor
	AttnO []*device.Tensor

	// AttnNorm   []*device.Tensor
	AttnNorm []*device.Tensor // Re-added just in case

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

	freed := make(map[uintptr]bool)
	freeTensor := func(t *device.Tensor) {
		if t == nil || t.BufferID() == 0 || freed[t.BufferID()] {
			return
		}
		freed[t.BufferID()] = true
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

	freeSlices(w.AttnQ, w.AttnK, w.AttnV, w.AttnO, w.AttnNorm)
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

type Engine struct {
	Ctx     *device.Context
	Model   *gguf.GGUFFile
	Config  config.Config
	Weights *LlamaWeights

	// Quality Evaluation
	QualityEval *QualityEvaluator

	// KV Cache
	Cache KVCache
	// KVCacheK []*device.Tensor // Deprecated: Use Cache.Get()
	// KVCacheV []*device.Tensor // Deprecated: Use Cache.Get()

	// SSM Cache (Mamba)
	SSMCache    []*MambaState
	MambaLayers []*MambaLayer

	// Cache State
	CachePos int

	// Tokenizer
	Tokenizer interface {
		Encode(text string) []int
		Decode(ids []int) string
	}

	// Activation Logger
	ActLogger *ActivationLogger

	// Enhanced ScanMax Tracking for first token
	TraceTracker *ActivationTraceTracker

	// Heuristic Global Scale (1.0 default, 100.0 if detected underscaling)
	GlobalScale float32
}

func NewActivationTraceTracker(numLayers int) *ActivationTraceTracker {
	return &ActivationTraceTracker{
		NumLayers: numLayers,
		Traces:    make([]ActivationTrace, 0),
		enabled:   true,
	}
}

func (at *ActivationTraceTracker) RecordLayer(layerName string, layerIdx int, stats device.ActivationStats) {
	if !at.enabled {
		return
	}

	trace := ActivationTrace{
		LayerName: layerName,
		LayerIdx:  layerIdx,
		Max:       stats.Max,
		Min:       stats.Min,
		Mean:      stats.Mean,
		RMS:       stats.RMS,
		Zeros:     stats.Zeros,
		NANs:      stats.NaNs,
		Infs:      stats.Infs,
		Sample:    make([]float32, len(stats.Sample)),
	}
	copy(trace.Sample, stats.Sample)

	at.Traces = append(at.Traces, trace)
}

func (at *ActivationTraceTracker) IsLayerCollapsed(layerIdx int) bool {
	for _, trace := range at.Traces {
		if trace.LayerIdx == layerIdx {
			return trace.RMS < CollapseThreshold || trace.Max < CollapseThreshold
		}
	}
	return false
}

func (at *ActivationTraceTracker) IsLayerSaturated(layerIdx int) bool {
	for _, trace := range at.Traces {
		if trace.LayerIdx == layerIdx {
			return trace.RMS > SaturationThreshold || trace.Max > SaturationThreshold || trace.Infs > 0
		}
	}
	return false
}

func (at *ActivationTraceTracker) GetCollapsedLayers() []int {
	var collapsed []int
	seen := make(map[int]bool)
	for _, trace := range at.Traces {
		if !seen[trace.LayerIdx] && (trace.RMS < CollapseThreshold || trace.Max < CollapseThreshold) {
			collapsed = append(collapsed, trace.LayerIdx)
			seen[trace.LayerIdx] = true
		}
	}
	return collapsed
}

func (at *ActivationTraceTracker) GetSaturatedLayers() []int {
	var saturated []int
	seen := make(map[int]bool)
	for _, trace := range at.Traces {
		if !seen[trace.LayerIdx] && (trace.RMS > SaturationThreshold || trace.Max > SaturationThreshold || trace.Infs > 0) {
			saturated = append(saturated, trace.LayerIdx)
			seen[trace.LayerIdx] = true
		}
	}
	return saturated
}

func (at *ActivationTraceTracker) SaveToFile(filename string) error {
	data, err := at.ExportJSON()
	if err != nil {
		return err
	}

	err = os.WriteFile(filename, data, 0644)
	if err != nil {
		return err
	}
	return nil
}

func (at *ActivationTraceTracker) ExportJSON() ([]byte, error) {
	return json.MarshalIndent(at, "", "  ")
}

func (at *ActivationTraceTracker) IsEnabled() bool {
	return at.enabled
}

func (at *ActivationTraceTracker) Enable() {
	at.enabled = true
}

func (at *ActivationTraceTracker) Disable() {
	at.enabled = false
}

func (at *ActivationTraceTracker) GetFirstTokenTraces() []ActivationTrace {
	var firstToken []ActivationTrace
	seen := make(map[int]bool)
	for _, trace := range at.Traces {
		if !seen[trace.LayerIdx] {
			firstToken = append(firstToken, trace)
			seen[trace.LayerIdx] = true
		}
	}
	return firstToken
}
