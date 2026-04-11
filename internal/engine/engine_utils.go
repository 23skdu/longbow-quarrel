package engine

import (
	"fmt"
	"strings"

	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

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

	// Try dynamic architecture detection
	if arch, ok := f.KV["general.architecture"].(string); ok && len(keys) > 0 {
		// Replace "llama." prefix with architecture name
		primaryKey := keys[0]
		if strings.Contains(primaryKey, "llama.") {
			archKey := strings.Replace(primaryKey, "llama.", arch+".", 1)
			if val, ok := f.KV[archKey]; ok {
				return val, true
			}
		}
	}

	return nil, false
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
