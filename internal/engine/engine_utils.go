package engine

import (
	"fmt"
	"strings"

	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

func getKV(f *gguf.GGUFFile, llamaKey, qwenKey string) (interface{}, bool) {
	// 1. Try provided llama key
	if val, ok := f.KV[llamaKey]; ok {
		return val, true
	}

	// 2. Try provided qwen key
	if qwenKey != "" {
		if val, ok := f.KV[qwenKey]; ok {
			return val, true
		}
	}

	// 3. Try general keys if applicable
	if strings.Contains(llamaKey, "llama.") {
		generalKey := strings.Replace(llamaKey, "llama.", "general.", 1)
		if val, ok := f.KV[generalKey]; ok {
			return val, true
		}
	}

	// 4. Try granite prefix (common in Ollama)
	if strings.Contains(llamaKey, "llama.") {
		graniteKey := strings.Replace(llamaKey, "llama.", "granite.", 1)
		if val, ok := f.KV[graniteKey]; ok {
			return val, true
		}
	}

	// 5. Dynamic architecture detection - try model-specific keys
	if arch, ok := f.KV["general.architecture"].(string); ok {
		// Replace "llama." with "<arch>."
		archKey := strings.Replace(llamaKey, "llama.", arch+".", 1)
		if val, ok := f.KV[archKey]; ok {
			return val, true
		}

		// Also try some common architecture prefixes
		architectures := []string{arch, "gemma", "gemma2", "gemma3", "mistral", "qwen2", "phi3", "starcoder2", "llama"}
		for _, alt := range architectures {
			altKey := strings.Replace(llamaKey, "llama.", alt+".", 1)
			if val, ok := f.KV[altKey]; ok {
				return val, true
			}
		}
	}

	// 6. Try gemma4 specific keys directly
	if val, ok := f.KV["gemma4.attention.head_count"]; ok {
		return val, true
	}

	// 7. Try gemma2/3 variations
	if val, ok := f.KV["gemma2.attention.head_count"]; ok {
		return val, true
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

