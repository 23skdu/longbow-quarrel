//go:build (darwin && metal) || (linux && amd64 && cuda && cgo)

package main

import (
	"encoding/binary"
	"os"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/engine"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

// generateDeepSeekMLATestGGUF creates a mock GGUF for DeepSeek V2/V3 Multi-Head Latent Attention (MLA)
func generateDeepSeekMLATestGGUF(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	_ = binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMagic))
	_ = binary.Write(f, binary.LittleEndian, uint32(3))
	_ = binary.Write(f, binary.LittleEndian, uint64(19)) // 19 tensors
	_ = binary.Write(f, binary.LittleEndian, uint64(17)) // 17 KV pairs

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

	// Model Architecture & Geometry
	writeKV("general.architecture", "deepseek2")
	writeKV("deepseek2.block_count", uint32(1))
	writeKV("deepseek2.embedding_length", uint32(64))
	writeKV("deepseek2.attention.head_count", uint32(4))
	writeKV("deepseek2.attention.head_count_kv", uint32(4))
	writeKV("deepseek2.attention.head_dim", uint32(16))
	writeKV("deepseek2.context_length", uint32(128))
	writeKV("deepseek2.rope.freq_base", float32(10000.0))
	writeKV("deepseek2.vocab_size", uint32(256))

	// MLA Specific Hyperparameters
	writeKV("deepseek2.attention.kv_lora_rank", uint32(16))
	writeKV("deepseek2.attention.q_lora_rank", uint32(16))
	writeKV("deepseek2.attention.qk_nope_head_dim", uint32(8))
	writeKV("deepseek2.attention.qk_rope_head_dim", uint32(8))
	writeKV("deepseek2.attention.v_head_dim", uint32(8))

	// MoE Specific Hyperparameters
	writeKV("deepseek2.expert_count", uint32(8))
	writeKV("deepseek2.expert_used_count", uint32(2))
	writeKV("deepseek2.expert_feed_forward_length", uint32(64))

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

	// Embedding & Output
	writeTensor("token_embd.weight", []uint64{64, 256})
	writeTensor("output.weight", []uint64{64, 256})
	writeTensor("output_norm.weight", []uint64{64})

	// Block Norm
	writeTensor("blk.0.attn_norm.weight", []uint64{64})

	// MLA Query Tensors (Low-Rank + RoPE)
	writeTensor("blk.0.attn_q_a.weight", []uint64{16, 64})
	writeTensor("blk.0.attn_q_a_norm.weight", []uint64{16})
	writeTensor("blk.0.attn_q_b.weight", []uint64{64, 16})

	// MLA KV Tensors (Compressed KV + Decoupled RoPE Key)
	writeTensor("blk.0.attn_kv_a_mqa.weight", []uint64{24, 64})
	writeTensor("blk.0.attn_kv_a_norm.weight", []uint64{16})
	writeTensor("blk.0.attn_kv_b.weight", []uint64{64, 16})
	writeTensor("blk.0.attn_output.weight", []uint64{64, 32})

	// FFN & MoE Tensors
	writeTensor("blk.0.ffn_norm.weight", []uint64{64})
	writeTensor("blk.0.ffn_gate_inp.weight", []uint64{64, 8})
	writeTensor("blk.0.ffn_gate_exps.weight", []uint64{64, 64, 8})
	writeTensor("blk.0.ffn_up_exps.weight", []uint64{64, 64, 8})
	writeTensor("blk.0.ffn_down_exps.weight", []uint64{64, 64, 8})
	writeTensor("blk.0.ffn_gate_shexp.weight", []uint64{0})
	writeTensor("blk.0.ffn_up_shexp.weight", []uint64{0})
	writeTensor("blk.0.ffn_down_shexp.weight", []uint64{0})

	// Data Padding and Dummy Content
	_, _ = f.Write(make([]byte, 2048))
	dummyData := make([]byte, 1024*1024*8) // 8MB dummy
	_, _ = f.Write(dummyData)

	return nil
}

func TestDeepSeekMLACoherence(t *testing.T) {
	modelPath := "deepseek_mla_mock.gguf"
	if err := generateDeepSeekMLATestGGUF(modelPath); err != nil {
		t.Fatalf("Failed to generate mock DeepSeek MLA GGUF: %v", err)
	}
	defer os.Remove(modelPath)

	conf := config.Default()
	e, err := engine.NewEngine(modelPath, conf)
	if err != nil {
		t.Fatalf("Failed to initialize engine: %v", err)
	}
	defer e.Close()

	if !e.Config().IsMLA {
		t.Fatal("Engine did not detect MLA architecture")
	}

	t.Logf("DeepSeek MLA Config: kv_lora_rank=%d, q_lora_rank=%d, qk_nope_dim=%d, qk_rope_dim=%d, v_head_dim=%d",
		e.Config().KVLoRARank, e.Config().QLoRARank, e.Config().QKNDim, e.Config().QKRopeDim, e.Config().VHeadDim)

	if e.Config().KVLoRARank != 16 {
		t.Errorf("Expected KVLoRARank=16, got %d", e.Config().KVLoRARank)
	}
	if e.Config().QKNDim != 8 {
		t.Errorf("Expected QKNDim=8, got %d", e.Config().QKNDim)
	}
	if e.Config().QKRopeDim != 8 {
		t.Errorf("Expected QKRopeDim=8, got %d", e.Config().QKRopeDim)
	}
	if e.Config().VHeadDim != 8 {
		t.Errorf("Expected VHeadDim=8, got %d", e.Config().VHeadDim)
	}

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
