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
	// Debug flags should be enabled by default
	if !cfg.DebugEmbedding {
		t.Error("expected DebugEmbedding to be true")
	}
	if !cfg.DebugAttention {
		t.Error("expected DebugAttention to be true")
	}
	if !cfg.DebugFFN {
		t.Error("expected DebugFFN to be true")
	}
	if !cfg.DebugLayerOutput {
		t.Error("expected DebugLayerOutput to be true")
	}
	if !cfg.DebugLogits {
		t.Error("expected DebugLogits to be true")
	}
	if !cfg.DebugMemory {
		t.Error("expected DebugMemory to be true")
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
				Layers:    32,
				Heads:     32,
				VocabSize: 32000,
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
}
