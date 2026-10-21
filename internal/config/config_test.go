package config

import (
	"testing"
)

func TestDefault(t *testing.T) {
	cfg := Default()

	if cfg.SeqLen != 2048 {
		t.Errorf("expected SeqLen 2048, got %d", cfg.SeqLen)
	}
	if cfg.Eps != 1e-5 {
		t.Errorf("expected Eps 1e-5, got %v", cfg.Eps)
	}
	if cfg.RopeTheta != 10000.0 {
		t.Errorf("expected RopeTheta 10000.0, got %v", cfg.RopeTheta)
	}
	if cfg.MaxTokens != 64 {
		t.Errorf("expected MaxTokens 64, got %d", cfg.MaxTokens)
	}
	if cfg.KVCacheType != KVCacheF16 {
		t.Errorf("expected KVCacheType KVCacheF16, got %v", cfg.KVCacheType)
	}
}

func TestValidate(t *testing.T) {
	tests := []struct {
		name    string
		config  Config
		wantErr bool
	}{
		{
			name: "valid config",
			config: Config{
				Dim:       4096,
				HiddenDim: 11008,
				Layers:    32,
				Heads:     32,
				KVHeads:   32,
				HeadDim:   128,
				VocabSize: 32000,
				SeqLen:    2048,
				Eps:       1e-5,
				RopeTheta: 10000.0,
			},
			wantErr: false,
		},
		{
			name: "invalid dim",
			config: Config{
				Dim:       0,
				Layers:    32,
				Heads:     32,
				VocabSize: 32000,
			},
			wantErr: true,
		},
		{
			name: "invalid layers",
			config: Config{
				Dim:       4096,
				Layers:    0,
				Heads:     32,
				VocabSize: 32000,
			},
			wantErr: true,
		},
		{
			name: "invalid heads",
			config: Config{
				Dim:       4096,
				Layers:    32,
				Heads:     0,
				VocabSize: 32000,
			},
			wantErr: true,
		},
		{
			name: "invalid vocab size",
			config: Config{
				Dim:       4096,
				Layers:    32,
				Heads:     32,
				VocabSize: 0,
			},
			wantErr: true,
		},
		{
			name: "negative dim",
			config: Config{
				Dim:       -1,
				Layers:    32,
				Heads:     32,
				VocabSize: 32000,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.config.Validate()
			if (err != nil) != tt.wantErr {
				t.Errorf("Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestKVCacheTypeConstants(t *testing.T) {
	// The KV cache type is selected with -kv-cache-type, so every value
	// has to be distinguishable and reachable.
	seen := map[KVCacheType]bool{}
	for _, kt := range []KVCacheType{KVCacheF32, KVCacheF16, KVCacheTQ1_0, KVCacheTQ2_0, KVCacheFP8, KVCacheQ8_0} {
		if seen[kt] {
			t.Errorf("duplicate KVCacheType value %d", kt)
		}
		seen[kt] = true
	}
	if len(seen) != 6 {
		t.Errorf("expected 6 distinct KV cache types, got %d", len(seen))
	}
}

func TestConfigFields(t *testing.T) {
	cfg := Config{
		Architecture:                  "llama",
		Dim:                           4096,
		HiddenDim:                     14336,
		Layers:                        32,
		Heads:                         32,
		KVHeads:                       8,
		HeadDim:                       128,
		VocabSize:                     32000,
		SeqLen:                        2048,
		Eps:                           1e-5,
		RopeTheta:                     10000.0,
		WindowSize:                    4096,
		KVCacheType:                   KVCacheFP8,
		KVCacheSize:                   8192,
		IsMOE:                         true,
		ExpertCount:                   128,
		ExpertUsedCount:               6,
		ExpertSharedCount:             1,
		ExpertFeedForwardLength:       4096,
		ExpertSharedFeedForwardLength: 14336,
		DebugMemory:                   true,
	}

	if err := cfg.Validate(); err != nil {
		t.Errorf("valid config should not return error: %v", err)
	}

	if cfg.GetArchitecture() != "llama" {
		t.Errorf("expected architecture 'llama', got %q", cfg.GetArchitecture())
	}
	if !cfg.IsLargeModel() {
		t.Error("expected IsLargeModel true for Dim=4096")
	}
	if !cfg.NeedsPagedAttention() {
		t.Error("expected NeedsPagedAttention true for WindowSize=4096")
	}

	cfgSmall := Config{Dim: 2048, WindowSize: 0}
	if cfgSmall.IsLargeModel() {
		t.Error("expected IsLargeModel false for Dim=2048")
	}
	if cfgSmall.NeedsPagedAttention() {
		t.Error("expected NeedsPagedAttention false for WindowSize=0")
	}
}

func TestValidate_EdgeCases(t *testing.T) {
	base := func() Config {
		return Config{
			Dim:       4096,
			HiddenDim: 11008,
			Layers:    32,
			Heads:     32,
			KVHeads:   32,
			HeadDim:   128,
			VocabSize: 32000,
			SeqLen:    2048,
			Eps:       1e-5,
			RopeTheta: 10000.0,
		}
	}

	tests := []struct {
		name    string
		mutate  func(*Config)
		wantErr bool
	}{
		{"invalid kv_heads <= 0", func(c *Config) { c.KVHeads = 0 }, true},
		{"invalid kv_heads > heads", func(c *Config) { c.KVHeads = 64 }, true},
		{"invalid head_dim <= 0", func(c *Config) { c.HeadDim = 0 }, true},
		{"dim mismatch", func(c *Config) { c.Dim = 2048 }, true},
		{"invalid seq_len <= 0", func(c *Config) { c.SeqLen = 0 }, true},
		{"invalid eps <= 0", func(c *Config) { c.Eps = 0 }, true},
		{"invalid rope_theta <= 0", func(c *Config) { c.RopeTheta = 0 }, true},
		{"invalid window_size < 0", func(c *Config) { c.WindowSize = -1 }, true},
		{"invalid hidden_dim <= 0", func(c *Config) { c.HiddenDim = 0 }, true},
		{"moe expert_count <= 0", func(c *Config) {
			c.IsMOE = true
			c.ExpertCount = 0
			c.ExpertUsedCount = 2
			c.ExpertFeedForwardLength = 1024
		}, true},
		{"moe expert_used_count <= 0", func(c *Config) {
			c.IsMOE = true
			c.ExpertCount = 8
			c.ExpertUsedCount = 0
			c.ExpertFeedForwardLength = 1024
		}, true},
		{"moe expert_used_count > expert_count", func(c *Config) {
			c.IsMOE = true
			c.ExpertCount = 4
			c.ExpertUsedCount = 8
			c.ExpertFeedForwardLength = 1024
		}, true},
		{"invalid vocab_size <= 0", func(c *Config) { c.VocabSize = 0 }, true},
		{"valid moe", func(c *Config) {
			c.IsMOE = true
			c.ExpertCount = 8
			c.ExpertUsedCount = 2
			c.ExpertFeedForwardLength = 1024
		}, false},
		{"moe expert_feed_forward_length <= 0", func(c *Config) {
			c.IsMOE = true
			c.ExpertCount = 8
			c.ExpertUsedCount = 2
			c.ExpertFeedForwardLength = 0
		}, true},
		{"invalid tensor_parallel_size < 0", func(c *Config) { c.TensorParallelSize = -1 }, true},
		{"invalid pipeline_stages < 0", func(c *Config) { c.PipelineStages = -1 }, true},
		{"valid gemma4 dim mismatch bypass", func(c *Config) {
			c.IsGemma4 = true
			c.Dim = 2048
		}, false},
		{"valid mla", func(c *Config) {
			c.IsMLA = true
			c.Dim = 2048
			c.KVLoRARank = 512
			c.QKNDim = 128
			c.QKRopeDim = 64
			c.VHeadDim = 128
		}, false},
		{"mla invalid kv_lora_rank", func(c *Config) {
			c.IsMLA = true
			c.Dim = 2048
			c.KVLoRARank = 0
			c.QKNDim = 128
			c.QKRopeDim = 64
			c.VHeadDim = 128
		}, true},
		{"mla invalid qk_nope_head_dim", func(c *Config) {
			c.IsMLA = true
			c.Dim = 2048
			c.KVLoRARank = 512
			c.QKNDim = 0
			c.QKRopeDim = 64
			c.VHeadDim = 128
		}, true},
		{"mla invalid qk_rope_head_dim", func(c *Config) {
			c.IsMLA = true
			c.Dim = 2048
			c.KVLoRARank = 512
			c.QKNDim = 128
			c.QKRopeDim = 0
			c.VHeadDim = 128
		}, true},
		{"mla invalid v_head_dim", func(c *Config) {
			c.IsMLA = true
			c.Dim = 2048
			c.KVLoRARank = 512
			c.QKNDim = 128
			c.QKRopeDim = 64
			c.VHeadDim = 0
		}, true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := base()
			tc.mutate(&cfg)
			if err := cfg.Validate(); (err != nil) != tc.wantErr {
				t.Errorf("Validate() error = %v, wantErr %v", err, tc.wantErr)
			}
		})
	}
}
