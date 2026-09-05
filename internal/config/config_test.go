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
	if cfg.PrecisionMode != PrecisionAuto {
		t.Errorf("expected PrecisionMode PrecisionAuto, got %v", cfg.PrecisionMode)
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

func TestPrecisionModeConstants(t *testing.T) {
	// Verify precision mode constants are ordered correctly
	if PrecisionAuto != 0 {
		t.Errorf("expected PrecisionAuto to be 0, got %d", PrecisionAuto)
	}
	if PrecisionFP16 != 1 {
		t.Errorf("expected PrecisionFP16 to be 1, got %d", PrecisionFP16)
	}
	if PrecisionF32FFN != 2 {
		t.Errorf("expected PrecisionF32FFN to be 2, got %d", PrecisionF32FFN)
	}
	if PrecisionMixed != 3 {
		t.Errorf("expected PrecisionMixed to be 3, got %d", PrecisionMixed)
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
		PrecisionMode:                 PrecisionFP16,
		KVCacheSize:                   8192,
		IsMOE:                         true,
		ExpertCount:                   128,
		ExpertUsedCount:               6,
		ExpertSharedCount:             1,
		ExpertFeedForwardLength:       4096,
		ExpertSharedFeedForwardLength: 14336,
		ExpertGroupCount:              8,
		ExpertGroupUsedCount:          4,
		ExpertWeightsNorm:             true,
		ExpertWeightsScale:            1.0,
		DebugDequant:                  true,
		DebugActivations:              true,
		DebugEmbedding:                true,
		DebugAttention:                true,
		DebugFFN:                      true,
		DebugLayerOutput:              true,
		DebugLogits:                   true,
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

