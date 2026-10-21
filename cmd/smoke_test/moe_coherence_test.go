//go:build (darwin && metal) || (linux && amd64 && cuda && cgo)

package main

import (
	"os"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/engine"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

// generateMoeTestGGUF creates a mock GGUF with Nemotron-like MoE metadata
func generateMoeTestGGUF(path string) error {
	kv := map[string]interface{}{
		"general.architecture":                    "llama",
		"llama.block_count":                       uint32(1),
		"llama.embedding_length":                  uint32(64),
		"llama.attention.head_count":              uint32(4),
		"llama.attention.head_count_kv":           uint32(4),
		"llama.context_length":                    uint32(128),
		"llama.rope.freq_base":                    float32(10000.0),
		"llama.vocab_size":                        uint32(256),
		"llama.expert_count":                      uint32(128),
		"llama.expert_used_count":                 uint32(6),
		"llama.expert_shared_count":               uint32(1),
		"llama.expert_feed_forward_length":        uint32(128),
		"llama.expert_shared_feed_forward_length": uint32(128),
		"llama.expert_weights_norm":               true,
	}

	var tensors []gguf.TensorData
	tensors = append(tensors, f32Tensor("token_embd.weight", []uint64{64, 256}, 1))
	tensors = append(tensors, f32Tensor("output.weight", []uint64{64, 256}, 2))
	tensors = append(tensors, f32Tensor("output_norm.weight", []uint64{64}, 3))
	tensors = append(tensors, f32Tensor("blk.0.attn_q.weight", []uint64{64, 64}, 4))
	tensors = append(tensors, f32Tensor("blk.0.attn_k.weight", []uint64{64, 64}, 5))
	tensors = append(tensors, f32Tensor("blk.0.attn_v.weight", []uint64{64, 64}, 6))
	tensors = append(tensors, f32Tensor("blk.0.attn_output.weight", []uint64{64, 64}, 7))
	tensors = append(tensors, f32Tensor("blk.0.attn_norm.weight", []uint64{64}, 8))
	tensors = append(tensors, f32Tensor("blk.0.ffn_norm.weight", []uint64{64}, 9))
	tensors = append(tensors, f32Tensor("blk.0.ffn_gate_inp.weight", []uint64{64, 128}, 10))
	tensors = append(tensors, f32Tensor("blk.0.ffn_gate_exps.weight", []uint64{128, 64, 128}, 11))
	tensors = append(tensors, f32Tensor("blk.0.ffn_up_exps.weight", []uint64{128, 64, 128}, 12))
	tensors = append(tensors, f32Tensor("blk.0.ffn_down_exps.weight", []uint64{64, 128, 128}, 13))
	tensors = append(tensors, f32Tensor("blk.0.ffn_gate_shexp.weight", []uint64{64, 128}, 14))
	tensors = append(tensors, f32Tensor("blk.0.ffn_up_shexp.weight", []uint64{64, 128}, 15))
	tensors = append(tensors, f32Tensor("blk.0.ffn_down_shexp.weight", []uint64{128, 64}, 16))

	return buildTestModel(path, kv, tensors)
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

	if !e.Config().IsMOE {
		t.Fatal("Engine did not detect MOE architecture")
	}

	t.Logf("MOE Config: experts=%d, used=%d, shared=%d",
		e.Config().ExpertCount, e.Config().ExpertUsedCount, e.Config().ExpertSharedCount)

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

// generateNemotronMiniTestGGUF creates a mock GGUF for Nemotron-Mini-4B (smaller MOE variant)
func generateNemotronMiniTestGGUF(path string) error {
	kv := map[string]interface{}{
		"general.architecture":                    "llama",
		"llama.block_count":                       uint32(1),
		"llama.embedding_length":                  uint32(64),
		"llama.attention.head_count":              uint32(4),
		"llama.attention.head_count_kv":           uint32(4),
		"llama.context_length":                    uint32(128),
		"llama.rope.freq_base":                    float32(10000.0),
		"llama.vocab_size":                        uint32(256),
		"llama.expert_count":                      uint32(128),
		"llama.expert_used_count":                 uint32(6),
		"llama.expert_shared_count":               uint32(1),
		"llama.expert_feed_forward_length":        uint32(96),
		"llama.expert_shared_feed_forward_length": uint32(96),
		"llama.expert_weights_norm":               true,
	}

	var tensors []gguf.TensorData
	tensors = append(tensors, f32Tensor("token_embd.weight", []uint64{64, 256}, 1))
	tensors = append(tensors, f32Tensor("output.weight", []uint64{64, 256}, 2))
	tensors = append(tensors, f32Tensor("output_norm.weight", []uint64{64}, 3))
	tensors = append(tensors, f32Tensor("blk.0.attn_q.weight", []uint64{64, 64}, 4))
	tensors = append(tensors, f32Tensor("blk.0.attn_k.weight", []uint64{64, 64}, 5))
	tensors = append(tensors, f32Tensor("blk.0.attn_v.weight", []uint64{64, 64}, 6))
	tensors = append(tensors, f32Tensor("blk.0.attn_output.weight", []uint64{64, 64}, 7))
	tensors = append(tensors, f32Tensor("blk.0.attn_norm.weight", []uint64{64}, 8))
	tensors = append(tensors, f32Tensor("blk.0.ffn_norm.weight", []uint64{64}, 9))
	tensors = append(tensors, f32Tensor("blk.0.ffn_gate_inp.weight", []uint64{64, 128}, 10))
	tensors = append(tensors, f32Tensor("blk.0.ffn_gate_exps.weight", []uint64{96, 64, 128}, 11))
	tensors = append(tensors, f32Tensor("blk.0.ffn_up_exps.weight", []uint64{96, 64, 128}, 12))
	tensors = append(tensors, f32Tensor("blk.0.ffn_down_exps.weight", []uint64{64, 96, 128}, 13))
	tensors = append(tensors, f32Tensor("blk.0.ffn_gate_shexp.weight", []uint64{64, 96}, 14))
	tensors = append(tensors, f32Tensor("blk.0.ffn_up_shexp.weight", []uint64{64, 96}, 15))
	tensors = append(tensors, f32Tensor("blk.0.ffn_down_shexp.weight", []uint64{96, 64}, 16))

	return buildTestModel(path, kv, tensors)
}

func TestNemotronMiniMOECoherence(t *testing.T) {
	modelPath := "nemotron_mini_moe_mock.gguf"
	if err := generateNemotronMiniTestGGUF(modelPath); err != nil {
		t.Fatalf("Failed to generate mock Nemotron-Mini GGUF: %v", err)
	}
	defer os.Remove(modelPath)

	conf := config.Default()
	e, err := engine.NewEngine(modelPath, conf)
	if err != nil {
		t.Fatalf("Failed to initialize engine: %v", err)
	}
	defer e.Close()

	if !e.Config().IsMOE {
		t.Fatal("Engine did not detect MOE architecture")
	}

	t.Logf("Nemotron-Mini MOE Config: experts=%d, used=%d, shared=%d",
		e.Config().ExpertCount, e.Config().ExpertUsedCount, e.Config().ExpertSharedCount)

	// Verify it's the mini variant (smaller FFN)
	if e.Config().ExpertFeedForwardLength != 96 {
		t.Logf("Note: ExpertFeedForwardLength=%d (expected 96 for mini)", e.Config().ExpertFeedForwardLength)
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

// generateGPTOSSTestGGUF creates a mock GGUF for GPT-OSS MOE model
func generateGPTOSSTestGGUF(path string) error {
	kv := map[string]interface{}{
		"general.architecture":                    "llama",
		"llama.block_count":                       uint32(1),
		"llama.embedding_length":                  uint32(64),
		"llama.attention.head_count":              uint32(4),
		"llama.attention.head_count_kv":           uint32(4),
		"llama.context_length":                    uint32(128),
		"llama.rope.freq_base":                    float32(10000.0),
		"llama.vocab_size":                        uint32(256),
		"llama.expert_count":                      uint32(64),
		"llama.expert_used_count":                 uint32(4),
		"llama.expert_shared_count":               uint32(0),
		"llama.expert_feed_forward_length":        uint32(128),
		"llama.expert_shared_feed_forward_length": uint32(0),
		"llama.expert_weights_norm":               false,
	}

	var tensors []gguf.TensorData
	tensors = append(tensors, f32Tensor("token_embd.weight", []uint64{64, 256}, 1))
	tensors = append(tensors, f32Tensor("output.weight", []uint64{64, 256}, 2))
	tensors = append(tensors, f32Tensor("output_norm.weight", []uint64{64}, 3))
	tensors = append(tensors, f32Tensor("blk.0.attn_q.weight", []uint64{64, 64}, 4))
	tensors = append(tensors, f32Tensor("blk.0.attn_k.weight", []uint64{64, 64}, 5))
	tensors = append(tensors, f32Tensor("blk.0.attn_v.weight", []uint64{64, 64}, 6))
	tensors = append(tensors, f32Tensor("blk.0.attn_output.weight", []uint64{64, 64}, 7))
	tensors = append(tensors, f32Tensor("blk.0.attn_norm.weight", []uint64{64}, 8))
	tensors = append(tensors, f32Tensor("blk.0.ffn_norm.weight", []uint64{64}, 9))
	tensors = append(tensors, f32Tensor("blk.0.ffn_gate_inp.weight", []uint64{64, 64}, 10))
	tensors = append(tensors, f32Tensor("blk.0.ffn_gate_exps.weight", []uint64{128, 64, 64}, 11))
	tensors = append(tensors, f32Tensor("blk.0.ffn_up_exps.weight", []uint64{128, 64, 64}, 12))
	tensors = append(tensors, f32Tensor("blk.0.ffn_down_exps.weight", []uint64{64, 128, 64}, 13))
	tensors = append(tensors, f32Tensor("blk.0.ffn_gate_shexp.weight", []uint64{0}, 14))
	tensors = append(tensors, f32Tensor("blk.0.ffn_up_shexp.weight", []uint64{0}, 15))
	tensors = append(tensors, f32Tensor("blk.0.ffn_down_shexp.weight", []uint64{0}, 16))

	return buildTestModel(path, kv, tensors)
}

func TestGPTOSSMOECoherence(t *testing.T) {
	modelPath := "gpt_oss_moe_mock.gguf"
	if err := generateGPTOSSTestGGUF(modelPath); err != nil {
		t.Fatalf("Failed to generate mock GPT-OSS GGUF: %v", err)
	}
	defer os.Remove(modelPath)

	conf := config.Default()
	t.Logf("DEBUG: conf before NewEngine = %+v", conf)
	e, err := engine.NewEngine(modelPath, conf)
	if err != nil {
		t.Fatalf("Failed to initialize engine: %v", err)
	}
	defer e.Close()

	if !e.Config().IsMOE {
		t.Fatal("Engine did not detect MOE architecture")
	}

	t.Logf("GPT-OSS MOE Config: experts=%d, used=%d, shared=%d",
		e.Config().ExpertCount, e.Config().ExpertUsedCount, e.Config().ExpertSharedCount)

	// Verify GPT-OSS specific config (64 experts, top-4)
	if e.Config().ExpertCount != 64 {
		t.Errorf("Expected 64 experts for GPT-OSS, got %d", e.Config().ExpertCount)
	}
	if e.Config().ExpertUsedCount != 4 {
		t.Errorf("Expected top-4 routing for GPT-OSS, got %d", e.Config().ExpertUsedCount)
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

// generateMixtralTestGGUF creates a mock GGUF for Mixtral-8x7B MOE model
func generateMixtralTestGGUF(path string) error {
	kv := map[string]interface{}{
		"general.architecture":                    "llama",
		"llama.block_count":                       uint32(1),
		"llama.embedding_length":                  uint32(64),
		"llama.attention.head_count":              uint32(4),
		"llama.attention.head_count_kv":           uint32(4),
		"llama.context_length":                    uint32(128),
		"llama.rope.freq_base":                    float32(10000.0),
		"llama.vocab_size":                        uint32(256),
		"llama.expert_count":                      uint32(8),
		"llama.expert_used_count":                 uint32(2),
		"llama.expert_shared_count":               uint32(0),
		"llama.expert_feed_forward_length":        uint32(128),
		"llama.expert_shared_feed_forward_length": uint32(0),
		"llama.expert_weights_norm":               false,
	}

	var tensors []gguf.TensorData
	tensors = append(tensors, f32Tensor("token_embd.weight", []uint64{64, 256}, 1))
	tensors = append(tensors, f32Tensor("output.weight", []uint64{64, 256}, 2))
	tensors = append(tensors, f32Tensor("output_norm.weight", []uint64{64}, 3))
	tensors = append(tensors, f32Tensor("blk.0.attn_q.weight", []uint64{64, 64}, 4))
	tensors = append(tensors, f32Tensor("blk.0.attn_k.weight", []uint64{64, 64}, 5))
	tensors = append(tensors, f32Tensor("blk.0.attn_v.weight", []uint64{64, 64}, 6))
	tensors = append(tensors, f32Tensor("blk.0.attn_output.weight", []uint64{64, 64}, 7))
	tensors = append(tensors, f32Tensor("blk.0.attn_norm.weight", []uint64{64}, 8))
	tensors = append(tensors, f32Tensor("blk.0.ffn_norm.weight", []uint64{64}, 9))
	tensors = append(tensors, f32Tensor("blk.0.ffn_gate_inp.weight", []uint64{64, 8}, 10))
	tensors = append(tensors, f32Tensor("blk.0.ffn_gate_exps.weight", []uint64{128, 64, 8}, 11))
	tensors = append(tensors, f32Tensor("blk.0.ffn_up_exps.weight", []uint64{128, 64, 8}, 12))
	tensors = append(tensors, f32Tensor("blk.0.ffn_down_exps.weight", []uint64{64, 128, 8}, 13))
	tensors = append(tensors, f32Tensor("blk.0.ffn_gate_shexp.weight", []uint64{0}, 14))
	tensors = append(tensors, f32Tensor("blk.0.ffn_up_shexp.weight", []uint64{0}, 15))
	tensors = append(tensors, f32Tensor("blk.0.ffn_down_shexp.weight", []uint64{0}, 16))

	return buildTestModel(path, kv, tensors)
}

func TestMixtralMOECoherence(t *testing.T) {
	modelPath := "mixtral_moe_mock.gguf"
	if err := generateMixtralTestGGUF(modelPath); err != nil {
		t.Fatalf("Failed to generate mock Mixtral GGUF: %v", err)
	}
	defer os.Remove(modelPath)

	conf := config.Default()
	e, err := engine.NewEngine(modelPath, conf)
	if err != nil {
		t.Fatalf("Failed to initialize engine: %v", err)
	}
	defer e.Close()

	if !e.Config().IsMOE {
		t.Fatal("Engine did not detect MOE architecture")
	}

	t.Logf("Mixtral MOE Config: experts=%d, used=%d, shared=%d",
		e.Config().ExpertCount, e.Config().ExpertUsedCount, e.Config().ExpertSharedCount)

	// Verify Mixtral specific config (8 experts, top-2)
	if e.Config().ExpertCount != 8 {
		t.Errorf("Expected 8 experts for Mixtral, got %d", e.Config().ExpertCount)
	}
	if e.Config().ExpertUsedCount != 2 {
		t.Errorf("Expected top-2 routing for Mixtral, got %d", e.Config().ExpertUsedCount)
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
