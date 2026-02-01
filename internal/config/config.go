package config

import (
	"fmt"
)

type PrecisionMode int

const (
	PrecisionAuto   PrecisionMode = iota // Heuristic based on Dim
	PrecisionFP16                        // Full FP16
	PrecisionF32FFN                      // FP32 FFN for small models
	PrecisionMixed                       // Mixed precision for large models
)

// Config holds all configuration for the engine
type Config struct {
	// Model Architecture
	Architecture string
	Dim          int
	HiddenDim    int
	Layers       int
	Heads        int
	KVHeads      int
	HeadDim      int // Dim / Heads usually
	VocabSize    int
	SeqLen       int
	Eps          float32
	RopeTheta    float32
	WindowSize   int // Sliding window size for attention (4096 for Mistral)

	// Runtime Configuration
	PrecisionMode PrecisionMode
	KVCacheSize   int // Overrides model's default if > 0

	// MOE (Mixture of Experts) Configuration
	ExpertCount                   int     // Total number of experts (e.g., 128 for Nemotron, 8 for Mixtral)
	ExpertUsedCount               int     // Number of experts used per token (e.g., 6 for Nemotron, 2 for Mixtral)
	ExpertSharedCount             int     // Number of shared experts (always active)
	ExpertFeedForwardLength       int     // Hidden dimension for expert-specific FFN
	ExpertSharedFeedForwardLength int     // Hidden dimension for shared expert FFN
	ExpertGroupCount              int     // Number of expert groups (for grouped MOE)
	ExpertGroupUsedCount          int     // Number of groups used per token
	ExpertWeightsNorm             bool    // Whether expert weights are normalized
	ExpertWeightsScale            float32 // Scaling factor for expert weights
	IsMOE                         bool    // Flag to indicate MOE architecture

	// Feature Flags / Toggles
	DebugDequant     bool
	DebugActivations bool

	DebugEmbedding   bool
	DebugAttention   bool
	DebugFFN         bool
	DebugLayerOutput bool
	DebugLogits      bool

	DebugMemory bool
}

// Validate ensures the configuration is sane
func (c *Config) Validate() error {
	if c.Dim <= 0 {
		return fmt.Errorf("invalid dim: %d", c.Dim)
	}
	if c.Layers <= 0 {
		return fmt.Errorf("invalid layers: %d", c.Layers)
	}
	if c.Heads <= 0 {
		return fmt.Errorf("invalid heads: %d", c.Heads)
	}
	if c.VocabSize <= 0 {
		return fmt.Errorf("invalid vocab size: %d", c.VocabSize)
	}
	return nil
}

// Default returns a default configuration
func Default() Config {
	return Config{
		SeqLen:        2048,
		Eps:           1e-5,
		RopeTheta:     10000.0,
		PrecisionMode: PrecisionAuto,

		DebugEmbedding:   true,
		DebugAttention:   true,
		DebugFFN:         true,
		DebugLayerOutput: true,
		DebugLogits:      true,
		DebugMemory:      true,
	}
}
