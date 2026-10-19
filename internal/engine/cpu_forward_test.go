package engine

import (
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
)

// buildCPUEngine builds a small CPU engine with one layer of real weights,
// so that the forward paths can be exercised end to end.
func buildCPUEngine(t *testing.T) *CPUEngine {
	t.Helper()

	const (
		hidden   = 4
		heads    = 2
		kvHeads  = 2
		headDim  = 2
		attnDim  = heads * headDim
		kvDim    = kvHeads * headDim
		ffnDim   = 6
		vocab    = 8
		embedDim = hidden
	)

	ones := func(n int) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = 0.1
		}
		return out
	}

	embed := make([][]float32, vocab)
	for i := range embed {
		embed[i] = ones(embedDim)
	}

	weights := &CPUWeights{
		TokenEmb:   embed,
		Output:     ones(vocab * hidden),
		OutputNorm: ones(hidden),
		AttnQ:      [][]float32{ones(hidden * attnDim)},
		AttnK:      [][]float32{ones(hidden * kvDim)},
		AttnV:      [][]float32{ones(hidden * kvDim)},
		AttnO:      [][]float32{ones(attnDim * hidden)},
		AttnNorm:   [][]float32{ones(hidden)},
		FfnGate:    [][]float32{ones(hidden * ffnDim)},
		FfnUp:      [][]float32{ones(hidden * ffnDim)},
		FfnDown:    [][]float32{ones(ffnDim * hidden)},
		FfnNorm:    [][]float32{ones(hidden)},
	}

	cfg := config.Default()
	cfg.Layers = 1
	cfg.Dim = hidden
	cfg.Heads = heads
	cfg.KVHeads = kvHeads
	cfg.HeadDim = headDim
	cfg.VocabSize = vocab
	cfg.Architecture = "llama"

	ctx := device.NewContext()
	t.Cleanup(ctx.Free)

	e := &CPUEngine{
		weights:      weights,
		config:       cfg,
		ctx:          ctx,
		PromptCache:  NewPromptCache(),
		seqKVCaches:  make(map[string]*CPUKVCache),
		BatchManager: NewContinuousBatchManager(),
		stopChan:     make(chan struct{}),
		doneChan:     make(chan struct{}),
	}
	go e.runBatchLoop()
	t.Cleanup(func() { close(e.stopChan) })

	return e
}

// TestCPUEngineForward covers the CPU engine's forward pass over a
// sequence of tokens.
func TestCPUEngineForward(t *testing.T) {
	e := buildCPUEngine(t)

	logits := e.Forward([]int{1, 2, 3})
	if len(logits) != e.config.VocabSize {
		t.Fatalf("the forward produced %d logits, want %d", len(logits), e.config.VocabSize)
	}
	for i, v := range logits {
		if v != v {
			t.Errorf("the logit %d is NaN", i)
		}
	}

	// No tokens gives an empty hidden state of the right size.
	empty := e.Forward(nil)
	if len(empty) != e.config.Dim {
		t.Errorf("an empty sequence produced %d values, want %d", len(empty), e.config.Dim)
	}
}

// TestCPUEngineForwardBatch covers the batched forward, which is the path
// the continuous batching manager uses.
func TestCPUEngineForwardBatch(t *testing.T) {
	e := buildCPUEngine(t)

	batch := &BatchDescriptor{
		Sequences: []*Sequence{
			{ID: 1, Tokens: []int{1, 2}},
			{ID: 2, Tokens: []int{3, 4}},
		},
		Tokens:      []int{1, 2, 3, 4},
		Offsets:     []int{0, 2},
		ContextLens: []int{0, 0},
	}

	results, err := e.ForwardBatch(batch)
	if err != nil {
		t.Fatalf("ForwardBatch failed: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("the forward produced %d results, want 2", len(results))
	}
	for i, r := range results {
		if r == nil {
			t.Errorf("the result %d is nil", i)
			continue
		}
		if len(r.ToHostF32()) == 0 {
			t.Errorf("the result %d is empty", i)
		}
		r.Free()
	}
}

// TestCPUEngineInfer covers the token generation loop.
func TestCPUEngineInfer(t *testing.T) {
	e := buildCPUEngine(t)

	tokens, err := e.Infer([]int{1, 2}, 4, SamplerConfig{Temperature: 0.7, TopK: 8, TopP: 0.95})
	if err != nil {
		t.Fatalf("Infer failed: %v", err)
	}
	if len(tokens) != 4 {
		t.Errorf("Infer produced %d tokens, want 4", len(tokens))
	}
	for _, tok := range tokens {
		if tok < 0 || tok >= e.config.VocabSize {
			t.Errorf("the token %d is out of range", tok)
		}
	}
}

// TestCPUEngineInferWithCallback covers the streaming inference path.
func TestCPUEngineInferWithCallback(t *testing.T) {
	e := buildCPUEngine(t)

	var streamed []int
	tokens, err := e.InferWithCallback([]int{1}, 3, SamplerConfig{Temperature: 0.7},
		func(tok int) { streamed = append(streamed, tok) })
	if err != nil {
		t.Fatalf("InferWithCallback failed: %v", err)
	}
	if len(tokens) != 3 {
		t.Errorf("InferWithCallback produced %d tokens, want 3", len(tokens))
	}
	if len(streamed) != 3 {
		t.Errorf("the callback fired %d times, want 3", len(streamed))
	}
}
