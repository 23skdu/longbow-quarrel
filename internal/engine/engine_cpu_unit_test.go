//go:build !metal
package engine

import (
	"testing"
	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
)

func initCPUEngineWeights(e *CPUEngine, dim, layers, heads, kvHeads, headDim, hiddenDim int) {
	w := e.weights
	make2D := func(rows, cols int) [][]float32 {
		s := make([][]float32, rows)
		for i := range s {
			s[i] = make([]float32, cols)
		}
		return s
	}
	w.AttnNorm = make2D(layers, dim)
	w.AttnQ = make2D(layers, dim*(heads*headDim))
	w.AttnK = make2D(layers, dim*(kvHeads*headDim))
	w.AttnV = make2D(layers, dim*(kvHeads*headDim))
	w.AttnO = make2D(layers, (heads*headDim)*dim)
	w.FfnNorm = make2D(layers, dim)
	w.FfnGate = make2D(layers, dim*hiddenDim)
	w.FfnUp = make2D(layers, dim*hiddenDim)
	w.FfnDown = make2D(layers, hiddenDim*dim)
}

func TestCPUEngine_ForwardBatch(t *testing.T) {
	ctx := device.NewContext()
	headDim := 32
	cfg := config.Config{
		Dim:       128,
		Layers:    1,
		VocabSize: 1000,
		Heads:     4,
		KVHeads:   4,
		HeadDim:   headDim,
		HiddenDim: 256,
	}
	
	e := &CPUEngine{
		ctx:    ctx,
		config: cfg,
		cache:  &PagedKVCache{},
		weights: &CPUWeights{
			TokenEmb: make([][]float32, 1000),
		},
		BatchManager: NewContinuousBatchManager(),
	}
	for i := range e.weights.TokenEmb {
		e.weights.TokenEmb[i] = make([]float32, cfg.Dim)
	}
	initCPUEngineWeights(e, cfg.Dim, cfg.Layers, cfg.Heads, cfg.KVHeads, headDim, cfg.HiddenDim)
	e.cache.Init(ctx, cfg)
	
	// Create a mock batch descriptor
	desc := &BatchDescriptor{
		Sequences: []*Sequence{
			{ID: 1, MaxTokens: 10},
			{ID: 2, MaxTokens: 10},
		},
		Tokens: []int{1, 2, 3, 4, 5},
		Offsets: []int{0, 3},
		TokenToSeq: []int{0, 0, 0, 1, 1},
		ContextLens: []int{0, 0},
	}
	
	// Pre-allocate in cache
	e.cache.Allocate("seq-1", 10)
	e.cache.Allocate("seq-2", 10)
	
	results, err := e.ForwardBatch(desc)
	if err != nil {
		t.Errorf("ForwardBatch failed: %v", err)
	}
	
	if len(results) != 2 {
		t.Errorf("Expected 2 results, got %d", len(results))
	}
}

func TestCPUEngine_Sampling(t *testing.T) {
	logits := []float32{1.0, 2.0, 5.0, 2.0, 1.0}
	
	// Test applyTemp
	temp := float64(0.5)
	res := applyTempCPU(logits, temp)
	if res[2] != 10.0 {
		t.Errorf("applyTemp failed, expected 10.0, got %f", res[2])
	}
	
	// Test TopK
	res = applyTopKCPU(logits, 2)
	// After TopK=2, only index 2 and (one of 1,3) should be non-zero
	count := 0
	for _, v := range res {
		if v > -1e9 {
			count++
		}
	}
	if count != 2 {
		t.Errorf("applyTopK failed, expected 2 non-zero, got %d", count)
	}
}

func TestCPUEngine_Softmax(t *testing.T) {
	logits := []float32{0, 0, 2.0} // exp(0)=1, exp(2)=7.38
	res := softmaxCPU(logits)
	
	sum := float32(0)
	for _, v := range res {
		sum += v
	}
	if sum < 0.99 || sum > 1.01 {
		t.Errorf("Softmax sum should be 1.0, got %f", sum)
	}
}

func TestCPUEngine_Lifecycle(t *testing.T) {
	ctx := device.NewContext()
	e := &CPUEngine{ctx: ctx, cache: &PagedKVCache{}}
	e.cache.Init(ctx, config.Config{Layers: 1})
	e.Close() // Should not panic
}
