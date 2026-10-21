package engine

import (
	"fmt"
	"strings"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

// validateModelFile rejects a GGUF file the engine cannot run before any
// tensor is touched: a tensor table whose offsets do not describe the data
// behind it, or a model without the embedding table every backend dereferences
// on its first forward pass.
//
// Without this check a corrupt offset or a missing token_embd.weight surfaces
// as a nil dereference deep in the forward pass; with it the load fails with
// the tensor's name.
func validateModelFile(f *gguf.GGUFFile) error {
	analyzer := gguf.NewMetadataAnalyzer(f)

	issues, err := analyzer.ValidateTensors()
	if err != nil {
		return fmt.Errorf("failed to validate the model's tensor table: %w", err)
	}
	if len(issues) > 0 {
		return fmt.Errorf("model file has an invalid tensor table: %s", issues[0])
	}

	if missing := analyzer.FindMissingTensors([]string{"token_embd.weight"}); len(missing) > 0 {
		return fmt.Errorf("model file is missing a required tensor: %s", strings.Join(missing, ", "))
	}

	return nil
}

func getKV(f *gguf.GGUFFile, keys ...string) (interface{}, bool) {
	// First, try all provided keys directly
	for _, key := range keys {
		if key == "" {
			continue
		}
		if val, ok := f.KV[key]; ok {
			return val, true
		}

		// Try general keys if applicable
		if strings.Contains(key, "llama.") {
			generalKey := strings.Replace(key, "llama.", "general.", 1)
			if val, ok := f.KV[generalKey]; ok {
				return val, true
			}
		}
	}

	// Try dynamic architecture detection across all keys
	if arch, ok := f.KV["general.architecture"].(string); ok && arch != "" {
		for _, key := range keys {
			if strings.Contains(key, "llama.") {
				archKey := strings.Replace(key, "llama.", arch+".", 1)
				if val, ok := f.KV[archKey]; ok {
					return val, true
				}
			}
			// Try prepending arch if key starts with common parameter names
			if !strings.Contains(key, ".") {
				archKey := arch + "." + key
				if val, ok := f.KV[archKey]; ok {
					return val, true
				}
			}
		}
	}

	return nil, false
}

// ExtractModelConfig extracts and normalizes model hyperparameters from a GGUF file
// supporting LLaMA, Mistral, Qwen (qwen2, qwen35), Gemma, and Nemotron.
func ExtractModelConfig(f *gguf.GGUFFile) config.Config {
	cfg := config.Default()
	arch := "unknown"
	if v, ok := f.KV["general.architecture"].(string); ok {
		arch = v
	}
	cfg.Architecture = arch

	// Block Count / Layers
	if val, ok := getKV(f, arch+".block_count", "llama.block_count", "general.block_count"); ok {
		cfg.Layers = int(toFloat64(val))
	}
	if cfg.Layers <= 0 {
		cfg.Layers = 1
	}

	// Embedding Dim
	if val, ok := getKV(f, arch+".embedding_length", arch+".hidden_size", "llama.embedding_length"); ok {
		cfg.Dim = int(toFloat64(val))
	}
	if cfg.Dim <= 0 {
		cfg.Dim = 2048
	}

	// Attention Heads
	if val, ok := getKV(f, arch+".attention.head_count", "llama.attention.head_count"); ok {
		cfg.Heads = int(toFloat64(val))
	}
	if cfg.Heads <= 0 {
		cfg.Heads = 32
	}

	// KV Heads
	if val, ok := getKV(f, arch+".attention.head_count_kv", arch+".attention.kv_head_count", "llama.attention.head_count_kv"); ok {
		cfg.KVHeads = int(toFloat64(val))
	}
	if cfg.KVHeads <= 0 {
		cfg.KVHeads = cfg.Heads
	}

	// HeadDim
	if cfg.Heads > 0 {
		cfg.HeadDim = cfg.Dim / cfg.Heads
	}
	if cfg.HeadDim <= 0 {
		if val, ok := getKV(f, arch+".attention.key_length", arch+".attention.head_dim"); ok {
			cfg.HeadDim = int(toFloat64(val))
		}
	}
	if cfg.HeadDim <= 0 {
		cfg.HeadDim = 128
	}

	// HiddenDim (FFN)
	if val, ok := getKV(f, arch+".feed_forward_length", arch+".intermediate_size", "llama.feed_forward_length"); ok {
		cfg.HiddenDim = int(toFloat64(val))
	}
	if cfg.HiddenDim <= 0 {
		cfg.HiddenDim = 4 * cfg.Dim
	}

	// Context Length / SeqLen
	if val, ok := getKV(f, arch+".context_length", "llama.context_length", "general.context_length"); ok {
		cfg.SeqLen = int(toFloat64(val))
	}
	if cfg.SeqLen <= 0 {
		cfg.SeqLen = 2048
	}

	// RoPE Frequency Base
	if val, ok := getKV(f, arch+".rope.freq_base", "llama.rope.freq_base"); ok {
		cfg.RopeTheta = float32(toFloat64(val))
	}
	if cfg.RopeTheta <= 0 {
		cfg.RopeTheta = 10000.0
	}

	// RMSNorm Epsilon
	if val, ok := getKV(f, arch+".attention.layer_norm_rms_epsilon", "llama.attention.layer_norm_rms_epsilon"); ok {
		cfg.Eps = float32(toFloat64(val))
	}
	if cfg.Eps <= 0 {
		cfg.Eps = 1e-5
	}

	// Vocab Size
	if val, ok := getKV(f, arch+".vocab_size", "llama.vocab_size"); ok {
		cfg.VocabSize = int(toFloat64(val))
	}
	if cfg.VocabSize <= 0 {
		if tokens, ok := f.KV["tokenizer.ggml.tokens"].([]interface{}); ok {
			cfg.VocabSize = len(tokens)
		} else {
			cfg.VocabSize = 32000
		}
	}

	// EOS Token ID
	if val, ok := getKV(f, "tokenizer.ggml.eos_token_id", arch+".eos_token_id"); ok {
		cfg.EOSTokenID = int(toFloat64(val))
	}
	if cfg.EOSTokenID <= 0 {
		cfg.EOSTokenID = 2
	}

	if arch == "gemma4" {
		cfg.IsGemma4 = true
		cfg.Gemma4SlidingWindowSize = 512
		cfg.Gemma4SlidingRoPETheta = 10000.0
		cfg.Gemma4FullRoPETheta = 1000000.0
		cfg.Gemma4PartialRoPEFactor = 0.25
		cfg.Gemma4SlidingHeadDim = 256
		cfg.Gemma4FullHeadDim = 512
		cfg.Gemma4SharedKVLayers = 18
		cfg.FinalLogitSoftcapping = 30.0
		cfg.Eps = 1e-6

		// Read shared KV layers from GGUF metadata
		if val, ok := getKV(f, arch+".attention.shared_kv_layers"); ok {
			cfg.Gemma4SharedKVLayers = int(toFloat64(val))
		}

		// Read sliding window pattern from GGUF metadata
		if val, ok := f.KV[arch+".attention.sliding_window_pattern"]; ok {
			if arr, ok := val.([]interface{}); ok {
				pattern := make([]bool, len(arr))
				for i, v := range arr {
					if f, ok := v.(float64); ok {
						pattern[i] = f != 0
					} else if b, ok := v.(bool); ok {
						pattern[i] = b
					}
				}
				cfg.Gemma4SlidingPattern = pattern
			}
		}

		// Read sliding window size from GGUF metadata
		if val, ok := getKV(f, arch+".attention.sliding_window"); ok {
			cfg.Gemma4SlidingWindowSize = int(toFloat64(val))
		}

		// Read head dimensions from GGUF metadata
		if val, ok := getKV(f, arch+".attention.key_length"); ok {
			cfg.Gemma4FullHeadDim = int(toFloat64(val))
		}
		if val, ok := getKV(f, arch+".attention.key_length_swa"); ok {
			cfg.Gemma4SlidingHeadDim = int(toFloat64(val))
		}

		// Read rope freq bases from GGUF metadata
		if val, ok := getKV(f, arch+".rope.freq_base"); ok {
			cfg.Gemma4FullRoPETheta = float32(toFloat64(val))
		}
		if val, ok := getKV(f, arch+".rope.freq_base_swa"); ok {
			cfg.Gemma4SlidingRoPETheta = float32(toFloat64(val))
		}

		// Read final logit softcapping from GGUF metadata
		if val, ok := getKV(f, arch+".final_logit_softcapping"); ok {
			cfg.FinalLogitSoftcapping = float32(toFloat64(val))
		}

		// Read epsilon from GGUF metadata
		if val, ok := getKV(f, arch+".attention.layer_norm_rms_epsilon"); ok {
			cfg.Eps = float32(toFloat64(val))
		}
	}

	// MOE Metadata
	if val, ok := getKV(f, arch+".expert_count", "llama.expert_count"); ok {
		cfg.ExpertCount = int(toFloat64(val))
		cfg.IsMOE = true
	}
	if val, ok := getKV(f, arch+".expert_used_count", "llama.expert_used_count"); ok {
		cfg.ExpertUsedCount = int(toFloat64(val))
	}
	if val, ok := getKV(f, arch+".expert_shared_count", "llama.expert_shared_count"); ok {
		cfg.ExpertSharedCount = int(toFloat64(val))
	}
	if val, ok := getKV(f, arch+".expert_feed_forward_length", "llama.expert_feed_forward_length"); ok {
		cfg.ExpertFeedForwardLength = int(toFloat64(val))
	}
	if val, ok := getKV(f, arch+".expert_shared_feed_forward_length", "llama.expert_shared_feed_forward_length"); ok {
		cfg.ExpertSharedFeedForwardLength = int(toFloat64(val))
	}
	// Grouped-expert routing is not implemented; see the note in
	// engine.go. The metadata is not copied into the configuration,
	// because a field nothing reads is a lie.
	if arch == "nemo" || arch == "nemotron" {
		cfg.IsMOE = true
	}

	// MLA (Multi-Head Latent Attention) Metadata (DeepSeek V2 / V3)
	if val, ok := getKV(f, arch+".attention.kv_lora_rank", "deepseek2.attention.kv_lora_rank", "deepseek.attention.kv_lora_rank"); ok {
		cfg.KVLoRARank = int(toFloat64(val))
		cfg.IsMLA = true
	}
	if val, ok := getKV(f, arch+".attention.q_lora_rank", "deepseek2.attention.q_lora_rank", "deepseek.attention.q_lora_rank"); ok {
		cfg.QLoRARank = int(toFloat64(val))
	}
	if val, ok := getKV(f, arch+".attention.qk_nope_head_dim", "deepseek2.attention.qk_nope_head_dim", "deepseek.attention.qk_nope_head_dim"); ok {
		cfg.QKNDim = int(toFloat64(val))
	}
	if val, ok := getKV(f, arch+".attention.qk_rope_head_dim", "deepseek2.attention.qk_rope_head_dim", "deepseek.attention.qk_rope_head_dim"); ok {
		cfg.QKRopeDim = int(toFloat64(val))
	}
	if val, ok := getKV(f, arch+".attention.v_head_dim", "deepseek2.attention.v_head_dim", "deepseek.attention.v_head_dim"); ok {
		cfg.VHeadDim = int(toFloat64(val))
	}
	if arch == "deepseek2" || arch == "deepseek" {
		cfg.IsMLA = true
	}

	return cfg
}

// checkMLAQLoRARank compares the low-rank query projection rank the model
// declares (q_lora_rank) with the rank the weights actually have.
//
// The MLA query projection has two stages when the model declares a rank:
// a projection down to q_lora_rank columns, a normalisation of those
// columns, and a projection back up to the head dimension. A model whose
// metadata and weights disagree computes the wrong query, silently, so the
// mismatch is an error rather than a warning.
//
// A rank of zero means the model does not declare one, and a weight with
// no columns means the tensor was not loaded; neither is a mismatch.
func checkMLAQLoRARank(declared, weightCols int) error {
	if declared <= 0 || weightCols <= 0 {
		return nil
	}
	if declared != weightCols {
		return fmt.Errorf("the model declares q_lora_rank %d but the low-rank query projection has %d columns",
			declared, weightCols)
	}
	return nil
}

func toFloat64(v interface{}) float64 {
	switch val := v.(type) {
	case float64:
		return val
	case float32:
		return float64(val)
	case uint64:
		return float64(val)
	case uint32:
		return float64(val)
	case int64:
		return float64(val)
	case int32:
		return float64(val)
	case int:
		return float64(val)
	}
	return 0
}

func isNormWeight(name string) bool {
	return strings.HasSuffix(name, "attn_norm.weight") || strings.HasSuffix(name, "ffn_norm.weight") || name == "output_norm.weight"
}

// ValidateTensorDimensions validates tensor dimensions based on quantization type
func ValidateTensorDimensions(name string, rows, cols int, ggufType gguf.GGMLType) error {
	switch ggufType {
	case gguf.GGMLTypeF32, gguf.GGMLTypeF16:
		if rows <= 0 || cols <= 0 {
			return fmt.Errorf("invalid dimensions: rows=%d, cols=%d", rows, cols)
		}
	case gguf.GGMLTypeQ4_0:
		if cols%32 != 0 {
			return fmt.Errorf("Q4_0 requires cols divisible by 32, got cols=%d", cols)
		}
		if rows <= 0 || cols <= 0 {
			return fmt.Errorf("invalid Q4_0 dimensions: rows=%d, cols=%d", rows, cols)
		}
	case gguf.GGMLTypeQ4_K:
		if cols%256 != 0 {
			return fmt.Errorf("Q4_K requires cols divisible by 256, got cols=%d", cols)
		}
		if rows <= 0 || cols <= 0 {
			return fmt.Errorf("invalid Q4_K dimensions: rows=%d, cols=%d", rows, cols)
		}
	case gguf.GGMLTypeQ6_K:
		if cols%256 != 0 {
			return fmt.Errorf("Q6_K requires cols divisible by 256, got cols=%d", cols)
		}
		if rows <= 0 || cols <= 0 {
			return fmt.Errorf("invalid Q6_K dimensions: rows=%d, cols=%d", rows, cols)
		}
	}
	return nil
}

func isNeededTensor(name string) bool {
	lowerName := strings.ToLower(name)
	// Global weights
	if (strings.HasSuffix(lowerName, "token_embd.weight") ||
		strings.HasSuffix(lowerName, "output_norm.weight") ||
		strings.HasSuffix(lowerName, "output.weight") ||
		strings.HasSuffix(lowerName, "model.embed_tokens.weight") ||
		strings.HasSuffix(lowerName, "model.norm.weight") ||
		strings.HasSuffix(lowerName, "model.lm_head.weight")) && !strings.Contains(lowerName, "blk.") {
		return true
	}

	// Layer weights
	if strings.Contains(lowerName, "blk.") {
		suffixes := []string{
			"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "attn_norm.weight",
			"attn_q_norm.weight", "attn_k_norm.weight", // Gemma4 Q/K normalization
			"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight", "ffn_norm.weight",
			"ssm_a", "ssm_d", "ssm_conv1d.weight", "ssm_conv1d.bias", "ssm_dt.weight", "ssm_dt.bias",
			"ssm_norm.weight", "ssm_norm.bias", "ssm_out.weight", "ssm_in.weight",
			// MOE weights
			"ffn_gate_inp.weight", "exp_probs_b.bias",
			"ffn_down_exps.weight", "ffn_up_exps.weight", "ffn_gate_exps.weight",
			"ffn_down_shexp.weight", "ffn_up_shexp.weight", "ffn_gate_shexp.weight",
		}
		for _, s := range suffixes {
			if strings.HasSuffix(lowerName, s) {
				return true
			}
		}
	}
	return false
}
