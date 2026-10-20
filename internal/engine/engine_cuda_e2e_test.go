//go:build linux && amd64 && cuda && cgo

package engine

import (
	"math"
	"os"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
)

// TestCUDAEngineEndToEnd runs the CUDA engine on a model built in the
// test.
//
// The engine had no end-to-end test, because the repository could not
// write a GGUF file. Now it can, and the whole path -- loading the
// weights, the forward pass, the sampling and the KV cache -- is
// exercised on the GPU.
func TestCUDAEngineEndToEnd(t *testing.T) {
	dir := t.TempDir()
	path := buildTestGGUF(t, dir)

	if _, err := os.Stat(path); err != nil {
		t.Fatalf("the test model was not written: %v", err)
	}

	cfg := config.Default()
	cfg.KVCacheSize = 128
	cfg.SeqLen = 128

	created, err := NewcudaEngine(path, cfg)
	if err != nil {
		t.Fatalf("failed to create the CUDA engine: %v", err)
	}
	defer created.Close()
	e := created.(*cudaEngine)

	// The forward pass produces finite logits.
	forward := forwardLastLogits(t, e, []int{1, 2, 3})
	if len(forward) == 0 {
		t.Fatal("the forward pass produced no logits")
	}
	for i, v := range forward {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("the logit %d is %v", i, v)
		}
	}

	// The generation loop produces in-range tokens.
	tokens, err := e.Infer([]int{1, 2}, 4, SamplerConfig{Temperature: 0.7, TopK: 8, TopP: 0.95})
	if err != nil {
		t.Fatalf("Infer failed: %v", err)
	}
	if len(tokens) != 4 {
		t.Errorf("Infer produced %d tokens, want 4", len(tokens))
	}
	for _, tok := range tokens {
		if tok < 0 || tok >= 32 {
			t.Errorf("the token %d is out of the vocabulary", tok)
		}
	}
}

// TestCUDAEngineIsDeterministic checks that the same prompt gives the same
// logits, which is what a broken weight layout or a stale buffer would
// break.
func TestCUDAEngineIsDeterministic(t *testing.T) {
	dir := t.TempDir()
	path := buildTestGGUF(t, dir)

	cfg := config.Default()
	cfg.KVCacheSize = 128
	cfg.SeqLen = 128

	created, err := NewcudaEngine(path, cfg)
	if err != nil {
		t.Fatalf("failed to create the CUDA engine: %v", err)
	}
	defer created.Close()
	e := created.(*cudaEngine)

	first := forwardLastLogits(t, e, []int{1, 2, 3})
	second := forwardLastLogits(t, e, []int{1, 2, 3})

	if len(first) != len(second) {
		t.Fatalf("the two forwards produced %d and %d logits", len(first), len(second))
	}
	for i := range first {
		if math.Abs(float64(first[i]-second[i])) > 1e-4 {
			t.Fatalf("the logit %d is %v the first time and %v the second", i, first[i], second[i])
		}
	}
}

// forwardLastLogits runs the batched forward over the tokens and returns
// the logits of the last one.
func forwardLastLogits(t *testing.T, e *cudaEngine, tokens []int) []float32 {
	t.Helper()

	tokenToSeq := make([]int, len(tokens))
	batch := &BatchDescriptor{
		Sequences:   []*Sequence{{ID: 1, Tokens: tokens, PromptLen: len(tokens)}},
		Tokens:      tokens,
		Offsets:     []int{0},
		ContextLens: []int{0},
		TokenToSeq:  tokenToSeq,
	}
	results, err := e.ForwardBatch(batch)
	if err != nil {
		t.Fatalf("the forward failed: %v", err)
	}
	defer results[0].Free()
	return results[0].ToHostF32()
}
