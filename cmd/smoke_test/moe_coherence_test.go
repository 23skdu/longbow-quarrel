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

// generateMoeTestGGUF creates a mock GGUF with Nemotron-like MoE metadata
func generateMoeTestGGUF(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	_ = binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMagic))
	_ = binary.Write(f, binary.LittleEndian, uint32(3))
	_ = binary.Write(f, binary.LittleEndian, uint64(16)) // Increased tensor count for MoE
	_ = binary.Write(f, binary.LittleEndian, uint64(14)) // Increased KV count for MoE

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
		case string:
			_ = binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeString))
			_ = binary.Write(f, binary.LittleEndian, uint64(len(v)))
			_, _ = f.WriteString(v)
		case bool:
			_ = binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeBool))
			_ = binary.Write(f, binary.LittleEndian, v)
		}
	}

	// Basic Model Metadata
	writeKV("general.architecture", "llama")
	writeKV("llama.block_count", uint32(1))
	writeKV("llama.embedding_length", uint32(64))
	writeKV("llama.attention.head_count", uint32(4))
	writeKV("llama.attention.head_count_kv", uint32(4))
	writeKV("llama.context_length", uint32(128))
	writeKV("llama.rope.freq_base", float32(10000.0))
	writeKV("llama.vocab_size", uint32(256))

	// MOE Metadata
	writeKV("llama.expert_count", uint32(128))
	writeKV("llama.expert_used_count", uint32(6))
	writeKV("llama.expert_shared_count", uint32(1))
	writeKV("llama.expert_feed_forward_length", uint32(128))
	writeKV("llama.expert_shared_feed_forward_length", uint32(128))
	writeKV("llama.expert_weights_norm", true)

	// Tensors
	writeTensor := func(name string, dims []uint64) {
		_ = binary.Write(f, binary.LittleEndian, uint64(len(name)))
		_, _ = f.WriteString(name)
		_ = binary.Write(f, binary.LittleEndian, uint32(len(dims)))
		for _, d := range dims {
			_ = binary.Write(f, binary.LittleEndian, uint64(d))
		}
		_ = binary.Write(f, binary.LittleEndian, uint32(gguf.GGMLTypeF32))
		_ = binary.Write(f, binary.LittleEndian, uint64(0)) // offset
	}

	// Global Tensors
	writeTensor("token_embd.weight", []uint64{64, 256})
	writeTensor("output.weight", []uint64{64, 256})
	writeTensor("output_norm.weight", []uint64{64})

	// Block Tensors (Standard)
	writeTensor("blk.0.attn_q.weight", []uint64{64, 64})
	writeTensor("blk.0.attn_k.weight", []uint64{64, 64})
	writeTensor("blk.0.attn_v.weight", []uint64{64, 64})
	writeTensor("blk.0.attn_output.weight", []uint64{64, 64})
	writeTensor("blk.0.attn_norm.weight", []uint64{64})
	writeTensor("blk.0.ffn_norm.weight", []uint64{64})

	// MOE Tensors
	writeTensor("blk.0.ffn_gate_inp.weight", []uint64{64, 128})       // Router: [dim, num_experts]
	writeTensor("blk.0.ffn_gate_exps.weight", []uint64{128, 64, 128}) // Experts: [hidden_dim, dim, num_experts]
	writeTensor("blk.0.ffn_up_exps.weight", []uint64{128, 64, 128})
	writeTensor("blk.0.ffn_down_exps.weight", []uint64{64, 128, 128}) // Down: [dim, hidden_dim, num_experts]

	// Shared Expert Tensors
	writeTensor("blk.0.ffn_gate_shexp.weight", []uint64{64, 128})
	writeTensor("blk.0.ffn_up_shexp.weight", []uint64{64, 128})
	writeTensor("blk.0.ffn_down_shexp.weight", []uint64{128, 64})

	// Data Padding and Dummy Content
	_, _ = f.Write(make([]byte, 2048))
	// Large enough dummy data
	dummyData := make([]byte, 1024*1024*10) // 10MB dummy
	_, _ = f.Write(dummyData)

	return nil
}

func TestNemotronMOECoherence(t *testing.T) {
	modelPath := "nemotron_moe_mock.gguf"
	if err := generateMoeTestGGUF(modelPath); err != nil {
		t.Fatalf("Failed to generate mock MOE GGUF: %v", err)
	}
	defer os.Remove(modelPath)

	conf := config.Default()
	e, err := engine.NewEngine(modelPath, conf)
	if err != nil {
		t.Fatalf("Failed to initialize engine: %v", err)
	}
	defer e.Close()

	if !e.Config.IsMOE {
		t.Fatal("Engine did not detect MOE architecture")
	}

	t.Logf("MOE Config: experts=%d, used=%d, shared=%d",
		e.Config.ExpertCount, e.Config.ExpertUsedCount, e.Config.ExpertSharedCount)

	// Basic inference test
	inputs := []int{1, 2, 3}
	genLen := 5
	sampler := engine.SamplerConfig{Temperature: 0}

	tokens, err := e.Infer(inputs, genLen, sampler)
	if err != nil {
		t.Fatalf("Inference failed: %v", err)
	}

	t.Logf("Generated tokens: %v", tokens)
	if len(tokens) != genLen {
		t.Errorf("Expected %d tokens, got %d", genLen, len(tokens))
	}
}
