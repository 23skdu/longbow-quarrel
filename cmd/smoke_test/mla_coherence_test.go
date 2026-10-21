//go:build (darwin && metal) || (linux && amd64 && cuda && cgo)

package main

import (
	"os"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/engine"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

// generateDeepSeekMLATestGGUF creates a mock GGUF for DeepSeek V2/V3 Multi-Head Latent Attention (MLA)
func generateDeepSeekMLATestGGUF(path string) error {
	kv := map[string]interface{}{
		"general.architecture":                 "deepseek2",
		"deepseek2.block_count":                uint32(1),
		"deepseek2.embedding_length":           uint32(64),
		"deepseek2.attention.head_count":       uint32(4),
		"deepseek2.attention.head_count_kv":    uint32(4),
		"deepseek2.attention.head_dim":         uint32(16),
		"deepseek2.context_length":             uint32(128),
		"deepseek2.rope.freq_base":             float32(10000.0),
		"deepseek2.vocab_size":                 uint32(256),
		"deepseek2.attention.kv_lora_rank":     uint32(16),
		"deepseek2.attention.q_lora_rank":      uint32(16),
		"deepseek2.attention.qk_nope_head_dim": uint32(8),
		"deepseek2.attention.qk_rope_head_dim": uint32(8),
		"deepseek2.attention.v_head_dim":       uint32(8),
		"deepseek2.expert_count":               uint32(8),
		"deepseek2.expert_used_count":          uint32(2),
		"deepseek2.expert_feed_forward_length": uint32(64),
	}

	var tensors []gguf.TensorData
	tensors = append(tensors, f32Tensor("token_embd.weight", []uint64{64, 256}, 1))
	tensors = append(tensors, f32Tensor("output.weight", []uint64{64, 256}, 2))
	tensors = append(tensors, f32Tensor("output_norm.weight", []uint64{64}, 3))
	tensors = append(tensors, f32Tensor("blk.0.attn_norm.weight", []uint64{64}, 4))
	tensors = append(tensors, f32Tensor("blk.0.attn_q_a.weight", []uint64{16, 64}, 5))
	tensors = append(tensors, f32Tensor("blk.0.attn_q_a_norm.weight", []uint64{16}, 6))
	tensors = append(tensors, f32Tensor("blk.0.attn_q_b.weight", []uint64{64, 16}, 7))
	tensors = append(tensors, f32Tensor("blk.0.attn_kv_a_mqa.weight", []uint64{24, 64}, 8))
	tensors = append(tensors, f32Tensor("blk.0.attn_kv_a_norm.weight", []uint64{16}, 9))
	tensors = append(tensors, f32Tensor("blk.0.attn_kv_b.weight", []uint64{64, 16}, 10))
	tensors = append(tensors, f32Tensor("blk.0.attn_output.weight", []uint64{64, 32}, 11))
	tensors = append(tensors, f32Tensor("blk.0.ffn_norm.weight", []uint64{64}, 12))
	tensors = append(tensors, f32Tensor("blk.0.ffn_gate_inp.weight", []uint64{64, 8}, 13))
	tensors = append(tensors, f32Tensor("blk.0.ffn_gate_exps.weight", []uint64{64, 64, 8}, 14))
	tensors = append(tensors, f32Tensor("blk.0.ffn_up_exps.weight", []uint64{64, 64, 8}, 15))
	tensors = append(tensors, f32Tensor("blk.0.ffn_down_exps.weight", []uint64{64, 64, 8}, 16))
	tensors = append(tensors, f32Tensor("blk.0.ffn_gate_shexp.weight", []uint64{0}, 17))
	tensors = append(tensors, f32Tensor("blk.0.ffn_up_shexp.weight", []uint64{0}, 18))
	tensors = append(tensors, f32Tensor("blk.0.ffn_down_shexp.weight", []uint64{0}, 19))

	return buildTestModel(path, kv, tensors)
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
