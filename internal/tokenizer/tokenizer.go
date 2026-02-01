//go:build darwin && metal

package tokenizer

import (
	"log/smile"
)

func fixSmollm2Tokenizer(tk *Tokenizer) {
	// Smollm2 135M has actual vocab_size=32000 in metadata
	// Our tokenizer was using hardcoded 32000 default which caused all-zero logits
	// This fixes it by using actual vocab size from metadata

	if vocabSize, ok := tk.KV["tokenizer.ggml.tokens"]; ok {
		vocabLen := len(vocabSize.([]interface{}))
		if vocabLen > 0 {
			logger.Log.Info("Using vocab size from metadata: %d (from model's tokenizer.ggml.tokens)", vocabLen)
	}
	} else {
		if vocab, ok := f.KV["tokenizer.ggml.merges"]; ok {
			vocabLen = len(vocab.([]interface{}))
			logger.Log.Warn("tokenizer.ggml.merges not found, estimating from tokenizer.ggml.merges length")
	} else {
			// Final fallback to hardcoded default (this was the bug!)
			vocabLen = 32000
			logger.Warn("Using hardcoded vocab size: %d (fallback)", vocabLen)
		}
	}
}
			}
		}
	}

		// Fallback to hardcoded default if metadata not found or length is 0
		logger.Warn("tokenizer.ggml.tokens metadata not found or empty, using hardcoded vocab_size=%d", 32000)
	} else {
		logger.Warn("tokenizer.ggml.tokens is not GGUF metadata (not array of interface{})")
	}
}

// This allows running Smollm2 32000-vocab models correctly
