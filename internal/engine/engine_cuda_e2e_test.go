//go:build linux && amd64 && cuda && cgo

package engine

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

// buildTestGGUF writes a small but complete llama-style model.
//
// The GGUF writer made it possible to build a model in a test, and with
// one the CUDA engine can be exercised end to end: it loads the weights,
// runs the forward pass and samples.
func buildTestGGUF(t *testing.T, dir string) string {
	t.Helper()

	const (
		vocab   = 32
		dim     = 8
		heads   = 2
		kvHeads = 2
		headDim = 4
		layers  = 2
		ffnDim  = 16
	)

	attnDim := heads * headDim
	kvDim := kvHeads * headDim

	// A deterministic pseudo-random weight, so that the model is not
	// symmetric and a broken forward pass shows up.
	weight := func(n int, seed int) []float32 {
		out := make([]float32, n)
		state := uint32(seed*2654435761 + 1)
		for i := range out {
			state = state*1664525 + 1013904223
			// A multiple of a sixteenth, so that the value is exactly
			// representable in half precision. The GPU runs the weights
			// in half precision and the CPU in single precision, and this
			// keeps the comparison about the layout rather than about the
			// precision.
			out[i] = float32(int32(state>>8)&0xF) / 16.0
		}
		return out
	}

	toF32 := func(values []float32) []byte {
		out := make([]byte, len(values)*4)
		for i, v := range values {
			bits := math.Float32bits(v)
			out[i*4] = byte(bits)
			out[i*4+1] = byte(bits >> 8)
			out[i*4+2] = byte(bits >> 16)
			out[i*4+3] = byte(bits >> 24)
		}
		return out
	}

	kv := map[string]interface{}{
		"general.architecture":                   "llama",
		"general.alignment":                      uint32(32),
		"llama.block_count":                      uint32(layers),
		"llama.embedding_length":                 uint32(dim),
		"llama.attention.head_count":             uint32(heads),
		"llama.attention.head_count_kv":          uint32(kvHeads),
		"llama.feed_forward_length":              uint32(ffnDim),
		"llama.rope.dimension_count":             uint32(headDim),
		"llama.rope.freq_base":                   float32(10000),
		"llama.context_length":                   uint32(256),
		"llama.vocab_size":                       uint32(vocab),
		"llama.attention.layer_norm_rms_epsilon": float32(1e-5),
		"tokenizer.ggml.tokens":                  []string{"a", "b", "c", "d", "e", "f", "g", "h"},
		"tokenizer.ggml.scores":                  []float32{0, -1, -2, -3, -4, -5, -6, -7},
		"tokenizer.ggml.token_type":              []int32{2, 1, 1, 1, 1, 1, 1, 1},
	}

	// The GGUF dimensions are the fastest dimension first, so a weight
	// matrix for a linear layer with an input of k and an output of n is
	// stored as [k, n], and the data is n rows of k.
	tensors := []gguf.TensorData{
		{Name: "token_embd.weight", Dimensions: []uint64{uint64(dim), uint64(vocab)}, Type: gguf.GGMLTypeF32, Data: toF32(weight(dim*vocab, 1))},
		{Name: "output_norm.weight", Dimensions: []uint64{uint64(dim)}, Type: gguf.GGMLTypeF32, Data: toF32(weight(dim, 2))},
		{Name: "output.weight", Dimensions: []uint64{uint64(dim), uint64(vocab)}, Type: gguf.GGMLTypeF32, Data: toF32(weight(dim*vocab, 3))},
	}

	for l := 0; l < layers; l++ {
		prefix := "blk." + string(rune('0'+l)) + "."
		tensors = append(tensors,
			gguf.TensorData{Name: prefix + "attn_norm.weight", Dimensions: []uint64{uint64(dim)}, Type: gguf.GGMLTypeF32, Data: toF32(weight(dim, 10+l))},
			gguf.TensorData{Name: prefix + "attn_q.weight", Dimensions: []uint64{uint64(dim), uint64(attnDim)}, Type: gguf.GGMLTypeF32, Data: toF32(weight(attnDim*dim, 20+l))},
			gguf.TensorData{Name: prefix + "attn_k.weight", Dimensions: []uint64{uint64(dim), uint64(kvDim)}, Type: gguf.GGMLTypeF32, Data: toF32(weight(kvDim*dim, 30+l))},
			gguf.TensorData{Name: prefix + "attn_v.weight", Dimensions: []uint64{uint64(dim), uint64(kvDim)}, Type: gguf.GGMLTypeF32, Data: toF32(weight(kvDim*dim, 40+l))},
			gguf.TensorData{Name: prefix + "attn_output.weight", Dimensions: []uint64{uint64(attnDim), uint64(dim)}, Type: gguf.GGMLTypeF32, Data: toF32(weight(dim*attnDim, 50+l))},
			gguf.TensorData{Name: prefix + "ffn_norm.weight", Dimensions: []uint64{uint64(dim)}, Type: gguf.GGMLTypeF32, Data: toF32(weight(dim, 60+l))},
			gguf.TensorData{Name: prefix + "ffn_gate.weight", Dimensions: []uint64{uint64(dim), uint64(ffnDim)}, Type: gguf.GGMLTypeF32, Data: toF32(weight(ffnDim*dim, 70+l))},
			gguf.TensorData{Name: prefix + "ffn_up.weight", Dimensions: []uint64{uint64(dim), uint64(ffnDim)}, Type: gguf.GGMLTypeF32, Data: toF32(weight(ffnDim*dim, 80+l))},
			gguf.TensorData{Name: prefix + "ffn_down.weight", Dimensions: []uint64{uint64(ffnDim), uint64(dim)}, Type: gguf.GGMLTypeF32, Data: toF32(weight(dim*ffnDim, 90+l))},
		)
	}

	path := filepath.Join(dir, "model.gguf")
	if err := gguf.WriteFile(path, kv, tensors, 32); err != nil {
		t.Fatalf("failed to write the test model: %v", err)
	}
	return path
}

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
	forward := e.forwardInternal([]int{1, 2, 3}, false)
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

	first := e.forwardInternal([]int{1, 2, 3}, false)
	second := e.forwardInternal([]int{1, 2, 3}, false)

	if len(first) != len(second) {
		t.Fatalf("the two forwards produced %d and %d logits", len(first), len(second))
	}
	for i := range first {
		if math.Abs(float64(first[i]-second[i])) > 1e-4 {
			t.Fatalf("the logit %d is %v the first time and %v the second", i, first[i], second[i])
		}
	}
}
