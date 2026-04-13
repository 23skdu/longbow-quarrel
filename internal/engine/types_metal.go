//go:build darwin && metal

package engine

import (
	"fmt"
	"sync"
	"encoding/json"
	"os"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

type metalEngine struct {
	ctx          *device.Context
	model        *gguf.GGUFFile
	config       config.Config
	weights      *LlamaWeights
	cache        *PagedKVCache
	BatchManager *ContinuousBatchManager
	PromptCache  *PromptCache
	DraftEngine  Engine
	SpeculativeMgr *SpeculativeManager

	// Quality Evaluation
	qualityEval *QualityEvaluator

	// SSM Cache (Mamba)
	SSMCache    []*MambaState
	MambaLayers []*MambaLayer

	// Sequence Management for concurrent requests
	SeqMgr *SequenceManager

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

	// Hot-swapping
	mu sync.RWMutex

	// Lifespan Control
	stopChan chan struct{}
	doneChan chan struct{}
}

func (e *metalEngine) Ctx() *device.Context {
	return e.ctx
}

func (e *metalEngine) Config() config.Config {
	return e.config
}

func (e *metalEngine) Weights() *LlamaWeights {
	return e.weights
}

func (e *metalEngine) Model() *gguf.GGUFFile {
	return e.model
}

func (e *metalEngine) GetSeqCachePos(seqID string) int {
	var id uint64
	_, err := fmt.Sscanf(seqID, "seq-%d", &id)
	if err != nil {
		return 0
	}
	if seq, ok := e.SeqMgr.GetSequence(id); ok {
		seq.mu.RLock()
		defer seq.mu.RUnlock()
		return seq.Pos
	}
	return 0
}

func (e *metalEngine) IncSeqCachePos(seqID uint64) {
	if seq, ok := e.SeqMgr.GetSequence(seqID); ok {
		seq.mu.Lock()
		defer seq.mu.Unlock()
		seq.Pos++
	}
}

func (e *metalEngine) SeqIDStr(seqID uint64) string {
	return fmt.Sprintf("seq-%d", seqID)
}

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
