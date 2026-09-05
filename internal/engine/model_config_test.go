package engine

import (
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

func TestExtractModelConfig_Qwen35(t *testing.T) {
	f := &gguf.GGUFFile{
		Header: gguf.GGUFHeader{Version: 3},
		KV: map[string]interface{}{
			"general.architecture":                   "qwen35",
			"qwen35.block_count":                     uint32(32),
			"qwen35.embedding_length":                uint32(2560),
			"qwen35.attention.head_count":            uint32(16),
			"qwen35.attention.head_count_kv":         uint32(4),
			"qwen35.feed_forward_length":             uint32(9216),
			"qwen35.context_length":                  uint32(262144),
			"qwen35.rope.freq_base":                  float32(10000000.0),
			"qwen35.attention.layer_norm_rms_epsilon": float32(1e-6),
			"qwen35.vocab_size":                      uint32(248320),
		},
	}

	cfg := ExtractModelConfig(f)

	if cfg.Architecture != "qwen35" {
		t.Errorf("expected architecture qwen35, got %s", cfg.Architecture)
	}
	if cfg.Layers != 32 {
		t.Errorf("expected 32 layers, got %d", cfg.Layers)
	}
	if cfg.Dim != 2560 {
		t.Errorf("expected dim 2560, got %d", cfg.Dim)
	}
	if cfg.Heads != 16 {
		t.Errorf("expected 16 heads, got %d", cfg.Heads)
	}
	if cfg.KVHeads != 4 {
		t.Errorf("expected 4 kv_heads, got %d", cfg.KVHeads)
	}
	if cfg.HeadDim != 160 {
		t.Errorf("expected head_dim 160, got %d", cfg.HeadDim)
	}
	if cfg.HiddenDim != 9216 {
		t.Errorf("expected hidden_dim 9216, got %d", cfg.HiddenDim)
	}
	if cfg.VocabSize != 248320 {
		t.Errorf("expected vocab 248320, got %d", cfg.VocabSize)
	}
}

func TestExtractModelConfig_LlamaFallback(t *testing.T) {
	f := &gguf.GGUFFile{
		Header: gguf.GGUFHeader{Version: 3},
		KV: map[string]interface{}{
			"general.architecture":        "llama",
			"llama.block_count":           uint64(28),
			"llama.embedding_length":      uint64(3072),
			"llama.attention.head_count":  uint64(24),
			"llama.attention.head_count_kv": uint64(8),
			"llama.feed_forward_length":   uint64(8192),
			"llama.context_length":        uint64(8192),
		},
	}

	cfg := ExtractModelConfig(f)

	if cfg.Architecture != "llama" {
		t.Errorf("expected architecture llama, got %s", cfg.Architecture)
	}
	if cfg.Layers != 28 {
		t.Errorf("expected 28 layers, got %d", cfg.Layers)
	}
	if cfg.Dim != 3072 {
		t.Errorf("expected dim 3072, got %d", cfg.Dim)
	}
	if cfg.Heads != 24 {
		t.Errorf("expected 24 heads, got %d", cfg.Heads)
	}
	if cfg.KVHeads != 8 {
		t.Errorf("expected 8 kv_heads, got %d", cfg.KVHeads)
	}
}
