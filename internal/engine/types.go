//go:build darwin && metal

package engine

import (
	"encoding/json"
	"os"

	"github.com/23skdu/longbow-quarrel/internal/device"

	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

type PrecisionMode int

const (
	PrecisionAuto   PrecisionMode = iota // Heuristic based on Dim
	PrecisionFP16                        // Full FP16
	PrecisionF32FFN                      // FP32 FFN for small models
	PrecisionMixed                       // Mixed precision for large models
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

type LlamaConfig struct {
	Dim           int
	HiddenDim     int
	Layers        int
	Heads         int
	KVHeads       int
	HeadDim       int // Dim / Heads usually
	VocabSize     int
	SeqLen        int
	Eps           float32
	RopeTheta     float32
	WindowSize    int // Sliding window size for attention (4096 for Mistral)
	PrecisionMode PrecisionMode

	// Debug Flags
	DebugDequant bool
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

	FfnGate []*device.Tensor
	FfnDown []*device.Tensor
	FfnUp   []*device.Tensor

	FfnNorm []*device.Tensor

	// Final
	OutputNorm *device.Tensor
	Output     *device.Tensor // vocab x dim (often shared with TokenEmb?)
}

type Engine struct {
	Ctx     *device.Context
	Model   *gguf.GGUFFile
	Config  LlamaConfig
	Weights *LlamaWeights

	// Quality Evaluation
	QualityEval *QualityEvaluator

	// KV Cache
	KVCacheK []*device.Tensor // layers x (seq_len x dim) ? No, pre-allocated buffer
	KVCacheV []*device.Tensor

	// Cache State
	CachePos int

	// Tokenizer
	Tokenizer interface{} // Will be *tokenizer.Tokenizer

	// Debug
	LastLogits []float32

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
