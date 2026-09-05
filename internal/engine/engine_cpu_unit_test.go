//go:build !metal && !cuda && !tpu
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

func TestCPUEngine_PrefillAndKVCache(t *testing.T) {
	ctx := device.NewContext()
	defer ctx.Free()

	headDim := 16
	cfg := config.Config{
		Dim:       64,
		Layers:    2,
		VocabSize: 50,
		Heads:     4,
		KVHeads:   2,
		HeadDim:   headDim,
		HiddenDim: 128,
		RopeTheta: 10000.0,
		Eps:       1e-5,
	}

	e := &CPUEngine{
		ctx:          ctx,
		config:       cfg,
		cache:        &PagedKVCache{},
		seqKVCaches:  make(map[string]*CPUKVCache),
		weights: &CPUWeights{
			TokenEmb: make([][]float32, 50),
		},
		BatchManager: NewContinuousBatchManager(),
	}
	for i := range e.weights.TokenEmb {
		e.weights.TokenEmb[i] = make([]float32, cfg.Dim)
		for j := range e.weights.TokenEmb[i] {
			e.weights.TokenEmb[i][j] = float32(i+1) * 0.01
		}
	}
	initCPUEngineWeights(e, cfg.Dim, cfg.Layers, cfg.Heads, cfg.KVHeads, headDim, cfg.HiddenDim)
	_ = e.cache.Init(ctx, cfg)
	_ = e.cache.Allocate("seq-42", 10)

	desc := &BatchDescriptor{
		Sequences: []*Sequence{
			{ID: 42, MaxTokens: 20, Pos: 0},
		},
		Tokens:      []int{5, 10, 15},
		Offsets:     []int{0},
		TokenToSeq:  []int{0, 0, 0},
		ContextLens: []int{0},
	}

	results, err := e.ForwardBatch(desc)
	if err != nil {
		t.Fatalf("ForwardBatch failed: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	logits := results[0].ToHostF32()
	results[0].Free()
	if len(logits) != cfg.Dim {
		t.Fatalf("expected logits dim %d, got %d", cfg.Dim, len(logits))
	}

	// Verify KV cache was populated for all 3 tokens
	cachedPos := e.GetSeqCachePos("seq-42")
	if cachedPos != 3 {
		t.Errorf("expected cached position 3, got %d", cachedPos)
	}

	// Now decode 1 token at pos 3
	descDecode := &BatchDescriptor{
		Sequences: []*Sequence{
			{ID: 42, MaxTokens: 20, Pos: 3},
		},
		Tokens:      []int{20},
		Offsets:     []int{0},
		TokenToSeq:  []int{0},
		ContextLens: []int{3},
	}
	results2, err := e.ForwardBatch(descDecode)
	if err != nil {
		t.Fatalf("ForwardBatch decode failed: %v", err)
	}
	results2[0].Free()

	// Verify KV cache grew to 4 tokens
	cachedPos2 := e.GetSeqCachePos("seq-42")
	if cachedPos2 != 4 {
		t.Errorf("expected cached position 4, got %d", cachedPos2)
	}

	// Test RollbackKV to position 2
	err = e.RollbackKV("seq-42", 2)
	if err != nil {
		t.Fatalf("RollbackKV failed: %v", err)
	}
	cachedPos3 := e.GetSeqCachePos("seq-42")
	if cachedPos3 != 2 {
		t.Errorf("expected cached position 2 after rollback, got %d", cachedPos3)
	}
}

func TestCPUEngine_Forward_MultiToken(t *testing.T) {
	ctx := device.NewContext()
	defer ctx.Free()

	headDim := 16
	cfg := config.Config{
		Dim:       64,
		Layers:    1,
		VocabSize: 50,
		Heads:     2,
		KVHeads:   2,
		HeadDim:   headDim,
		HiddenDim: 64,
		RopeTheta: 10000.0,
		Eps:       1e-5,
	}

	e := &CPUEngine{
		ctx:    ctx,
		config: cfg,
		weights: &CPUWeights{
			TokenEmb: make([][]float32, 50),
		},
	}
	for i := range e.weights.TokenEmb {
		e.weights.TokenEmb[i] = make([]float32, cfg.Dim)
		for j := range e.weights.TokenEmb[i] {
			e.weights.TokenEmb[i][j] = float32(i + j)
		}
	}
	initCPUEngineWeights(e, cfg.Dim, cfg.Layers, cfg.Heads, cfg.KVHeads, headDim, cfg.HiddenDim)

	out := e.forward([]int{1, 2, 3})
	if len(out) != cfg.Dim {
		t.Fatalf("expected output length %d, got %d", cfg.Dim, len(out))
	}
	for i, v := range out {
		if v != v { // NaN check
			t.Fatalf("NaN encountered in forward output at %d", i)
		}
	}
}

