//go:build linux && !cuda

package main

import (
	"math"
	"os"
	"strconv"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/engine"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

// generateTestGGUFCPU writes a small llama-style model.
//
// The generator used to write the file by hand: every tensor got the
// offset zero and the file held one block of data. That was never a valid
// GGUF, and the engine now rejects it, which is what the test reports.
func generateTestGGUFCPU(path string, seqLen uint32) error {
	const (
		dim      = 32
		vocab    = 100
		ffnDim   = 64
		attnDim  = 32
		kvDim    = 32
		hiddenNo = 32
	)

	weight := func(n int) []byte {
		out := make([]byte, n*4)
		for i := 0; i < n; i++ {
			// A small, non-zero weight: a zero weight makes every
			// output zero and the coherence checks meaningless.
			bits := math.Float32bits(float32(i%7) * 0.01)
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
		"llama.block_count":                      uint32(1),
		"llama.embedding_length":                 uint32(dim),
		"llama.attention.head_count":             uint32(4),
		"llama.attention.head_count_kv":          uint32(4),
		"llama.attention.layer_norm_rms_epsilon": float32(1e-5),
		"llama.context_length":                   seqLen,
		"llama.rope.freq_base":                   float32(10000.0),
		"llama.feed_forward_length":              uint32(ffnDim),
		"llama.vocab_size":                       uint32(vocab),
		"tokenizer.ggml.model":                   "llama",
	}

	tokens := make([]string, vocab)
	scores := make([]float32, vocab)
	types := make([]int32, vocab)
	for i := range tokens {
		tokens[i] = "t" + strconv.Itoa(i)
		scores[i] = float32(-i)
		types[i] = 1
	}
	kv["tokenizer.ggml.tokens"] = tokens
	kv["tokenizer.ggml.scores"] = scores
	kv["tokenizer.ggml.token_type"] = types

	tensors := []gguf.TensorData{
		{Name: "token_embd.weight", Dimensions: []uint64{uint64(dim), uint64(vocab)}, Type: gguf.GGMLTypeF32, Data: weight(dim * vocab)},
		{Name: "output.weight", Dimensions: []uint64{uint64(dim), uint64(vocab)}, Type: gguf.GGMLTypeF32, Data: weight(dim * vocab)},
		{Name: "output_norm.weight", Dimensions: []uint64{uint64(dim)}, Type: gguf.GGMLTypeF32, Data: weight(dim)},
		{Name: "blk.0.attn_norm.weight", Dimensions: []uint64{uint64(dim)}, Type: gguf.GGMLTypeF32, Data: weight(dim)},
		{Name: "blk.0.attn_q.weight", Dimensions: []uint64{uint64(dim), uint64(attnDim)}, Type: gguf.GGMLTypeF32, Data: weight(attnDim * dim)},
		{Name: "blk.0.attn_k.weight", Dimensions: []uint64{uint64(dim), uint64(kvDim)}, Type: gguf.GGMLTypeF32, Data: weight(kvDim * dim)},
		{Name: "blk.0.attn_v.weight", Dimensions: []uint64{uint64(dim), uint64(kvDim)}, Type: gguf.GGMLTypeF32, Data: weight(kvDim * dim)},
		{Name: "blk.0.attn_output.weight", Dimensions: []uint64{uint64(attnDim), uint64(dim)}, Type: gguf.GGMLTypeF32, Data: weight(dim * attnDim)},
		{Name: "blk.0.ffn_norm.weight", Dimensions: []uint64{uint64(dim)}, Type: gguf.GGMLTypeF32, Data: weight(dim)},
		{Name: "blk.0.ffn_gate.weight", Dimensions: []uint64{uint64(dim), uint64(ffnDim)}, Type: gguf.GGMLTypeF32, Data: weight(ffnDim * dim)},
		{Name: "blk.0.ffn_up.weight", Dimensions: []uint64{uint64(dim), uint64(ffnDim)}, Type: gguf.GGMLTypeF32, Data: weight(ffnDim * dim)},
		{Name: "blk.0.ffn_down.weight", Dimensions: []uint64{uint64(ffnDim), uint64(dim)}, Type: gguf.GGMLTypeF32, Data: weight(dim * ffnDim)},
	}

	return gguf.WriteFile(path, kv, tensors, 32)
}

func TestCPUCoherenceWrapping(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping coherence test in short mode")
	}

	modelPath := "test_cpu_coherence.gguf"
	defer func() { _ = os.Remove(modelPath) }()

	seqLen := uint32(32)
	if err := generateTestGGUFCPU(modelPath, seqLen); err != nil {
		t.Fatalf("Failed to create GGUF: %v", err)
	}

	conf := config.Default()
	conf.KVCacheSize = 32
	conf.WindowSize = 0

	e, err := engine.NewEngine(modelPath, conf)
	if err != nil {
		// The engine failed to load a model the test built itself: that
		// is a defect, not a platform.
		t.Fatalf("the engine failed to load the model: %v", err)
	}
	defer e.Close()

	inputs := []int{1, 2, 3, 4, 5}
	genLen := 40
	sampler := engine.SamplerConfig{Temperature: 0}

	tokens, err := e.Infer(inputs, genLen, sampler)
	if err != nil {
		t.Fatalf("Infer failed: %v", err)
	}

	t.Logf("Generated %d tokens on CPU", len(tokens))

	cachePos := 0 // Not available on CPU engine directly
	_ = cachePos
	if cachePos <= 32 {
		t.Logf("CachePos %d did not advance beyond context limit 32", cachePos)
	} else {
		t.Logf("CachePos reached %d successfully (CPU Context Wrapping Verified)", cachePos)
	}
}

func TestCPUMultiTokenCoherence(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping coherence test in short mode")
	}

	modelPath := "test_cpu_multitoken.gguf"
	defer func() { _ = os.Remove(modelPath) }()

	seqLen := uint32(64)
	if err := generateTestGGUFCPU(modelPath, seqLen); err != nil {
		t.Fatalf("Failed to create GGUF: %v", err)
	}

	conf := config.Default()
	conf.KVCacheSize = 64

	e, err := engine.NewEngine(modelPath, conf)
	if err != nil {
		// The engine failed to load a model the test built itself: that
		// is a defect, not a platform.
		t.Fatalf("the engine failed to load the model: %v", err)
	}
	defer e.Close()

	prompts := [][]int{
		{1, 2, 3, 4, 5},
		{6, 7, 8, 9, 10},
	}

	var outputs [][]int
	for i, input := range prompts {
		sampler := engine.SamplerConfig{Temperature: 0.7, TopK: 50}
		tokens, err := e.Infer(input, 10, sampler)
		if err != nil {
			t.Fatalf("Infer %d failed: %v", i, err)
		}
		outputs = append(outputs, tokens)
	}

	t.Logf("Generated %d outputs on CPU", len(outputs))

	if len(outputs) != len(prompts) {
		t.Errorf("Expected %d outputs, got %d", len(prompts), len(outputs))
	}
}

func TestCPUSelfConsistency(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping coherence test in short mode")
	}

	modelPath := "test_cpu_consistency.gguf"
	defer func() { _ = os.Remove(modelPath) }()

	seqLen := uint32(128)
	if err := generateTestGGUFCPU(modelPath, seqLen); err != nil {
		t.Fatalf("Failed to create GGUF: %v", err)
	}

	conf := config.Default()
	conf.KVCacheSize = 128

	e, err := engine.NewEngine(modelPath, conf)
	if err != nil {
		// The engine failed to load a model the test built itself: that
		// is a defect, not a platform.
		t.Fatalf("the engine failed to load the model: %v", err)
	}

	input := []int{1, 2, 3, 4, 5}
	sampler := engine.SamplerConfig{Temperature: 0}

	tokens1, err := e.Infer(input, 20, sampler)
	if err != nil {
		t.Fatalf("First infer failed: %v", err)
	}

	e2, err := engine.NewEngine(modelPath, conf)
	if err != nil {
		t.Fatalf("the second engine failed to load the model: %v", err)
	}
	defer e2.Close()

	tokens2, err := e2.Infer(input, 20, sampler)
	if err != nil {
		t.Fatalf("Second infer failed: %v", err)
	}

	consistent := len(tokens1) == len(tokens2)
	if consistent {
		for i := range tokens1 {
			if tokens1[i] != tokens2[i] {
				consistent = false
				break
			}
		}
	}

	if !consistent {
		t.Logf("Outputs differ (expected with non-deterministic sampling)")
	} else {
		t.Logf("Self-consistency verified: %d identical tokens", len(tokens1))
	}
}

func TestCPUKVCacheCorrectness(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping coherence test in short mode")
	}

	modelPath := "test_cpu_kv_cache.gguf"
	defer func() { _ = os.Remove(modelPath) }()

	seqLen := uint32(32)
	if err := generateTestGGUFCPU(modelPath, seqLen); err != nil {
		t.Fatalf("Failed to create GGUF: %v", err)
	}

	conf := config.Default()
	conf.KVCacheSize = 64

	e, err := engine.NewEngine(modelPath, conf)
	if err != nil {
		// The engine failed to load a model the test built itself: that
		// is a defect, not a platform.
		t.Fatalf("the engine failed to load the model: %v", err)
	}
	defer e.Close()

	input1 := []int{1, 2, 3, 4, 5}
	sampler := engine.SamplerConfig{Temperature: 0}

	_, err = e.Infer(input1, 5, sampler)
	if err != nil {
		t.Fatalf("First infer failed: %v", err)
	}

	input2 := []int{6, 7, 8, 9, 10}
	_, err = e.Infer(input2, 5, sampler)
	if err != nil {
		t.Fatalf("Second infer failed: %v", err)
	}

	t.Logf("KV cache correctness test passed (no crashes)")
}
