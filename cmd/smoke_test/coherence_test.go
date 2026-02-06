//go:build darwin && metal

package main

import (
	"encoding/binary"
	"os"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/engine"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

// Minimal GGUF generator for testing
func generateTestGGUF(path string, seqLen uint32) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()

	_ = binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMagic))
	_ = binary.Write(f, binary.LittleEndian, uint32(3))
	_ = binary.Write(f, binary.LittleEndian, uint64(12)) // Tensor count
	_ = binary.Write(f, binary.LittleEndian, uint64(6))  // KV count

	writeKV := func(key string, val interface{}) {
		_ = binary.Write(f, binary.LittleEndian, uint64(len(key)))
		_, _ = f.WriteString(key)
		switch v := val.(type) {
		case uint32:
			_ = binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeUint32))
			_ = binary.Write(f, binary.LittleEndian, v)
		case float32:
			_ = binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeFloat32))
			_ = binary.Write(f, binary.LittleEndian, v)
		}
	}

	writeKV("llama.block_count", uint32(1))
	writeKV("llama.embedding_length", uint32(32)) // Dim 32
	writeKV("llama.attention.head_count", uint32(4))
	writeKV("llama.attention.head_count_kv", uint32(4))
	writeKV("llama.context_length", seqLen)
	writeKV("llama.rope.freq_base", float32(10000.0))

	// Tensors
	writeTensor := func(name string, rows, cols int) {
		_ = binary.Write(f, binary.LittleEndian, uint64(len(name)))
		_, _ = f.WriteString(name)
		_ = binary.Write(f, binary.LittleEndian, uint32(2)) // 2 dims
		_ = binary.Write(f, binary.LittleEndian, uint64(cols))
		_ = binary.Write(f, binary.LittleEndian, uint64(rows))
		_ = binary.Write(f, binary.LittleEndian, uint32(gguf.GGMLTypeF32)) // Type F32
		offset := uint64(0)
		_ = binary.Write(f, binary.LittleEndian, offset)
	}

	tensorNames := []string{
		"token_embd.weight",
		"output.weight",
		"output_norm.weight",
		"blk.0.attn_q.weight",
		"blk.0.attn_k.weight",
		"blk.0.attn_v.weight",
		"blk.0.attn_output.weight",
		"blk.0.attn_norm.weight",
		"blk.0.ffn_gate.weight",
		"blk.0.ffn_up.weight",
		"blk.0.ffn_down.weight",
		"blk.0.ffn_norm.weight",
	}

	for _, n := range tensorNames {
		// Mock dimensions
		rows, cols := 32, 32
		if n == "token_embd.weight" || n == "output.weight" {
			rows = 100 // Vocab
			cols = 32
		}
		writeTensor(n, rows, cols)
	}

	// Fake Data Padding
	_, _ = f.Write(make([]byte, 1024))

	// Write dummy float data
	data := make([]byte, 100*32*4) // Max size needed
	_, _ = f.Write(data)

	_ = f.Close()
	return nil
}

func TestCoherenceWrapping(t *testing.T) {
	modelPath := "test_coherence.gguf"
	defer func() { _ = os.Remove(modelPath) }()

	// Generate model with small context limit
	seqLen := uint32(32)
	if err := generateTestGGUF(modelPath, seqLen); err != nil {
		t.Fatalf("Failed to create GGUF: %v", err)
	}

	// Init Engine with small cache
	conf := config.Default()
	conf.KVCacheSize = 32
	// WindowSize 0 = Full Context, should use SlidingWindowKVCache with size 32
	conf.WindowSize = 0

	e, err := engine.NewEngine(modelPath, conf)
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}
	defer e.Close()

	// Initial prompt
	inputs := []int{1, 2, 3, 4, 5}

	// Generate enough tokens to exceed context
	// filled: 5. to generate: 40. Total 45 > 32.
	genLen := 40

	sampler := engine.SamplerConfig{Temperature: 0}

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("Crashed (Panic) during coherence test: %v", r)
		}
	}()

	tokens, err := e.Infer(inputs, genLen, sampler)
	if err != nil {
		t.Fatalf("Infer failed: %v", err)
	}

	t.Logf("Generated %d tokens", len(tokens))

	if e.CachePos <= 32 {
		t.Errorf("CachePos %d did not advance beyond context limit 32", e.CachePos)
	} else {
		t.Logf("CachePos reached %d successfully (Context Wrapping Verified)", e.CachePos)
	}
}

func TestMainCoherence(t *testing.T) {
	// This test is usually run via go test, so this is just a wrapper if needed
	TestCoherenceWrapping(t)
}
