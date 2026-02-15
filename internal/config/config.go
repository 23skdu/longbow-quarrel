package config

import (
	"fmt"
	"strings"
)

type PrecisionMode int

const (
	PrecisionAuto PrecisionMode = iota
	PrecisionFP16
	PrecisionF32FFN
	PrecisionMixed
)

type Config struct {
	Architecture string
	Dim          int
	HiddenDim    int
	Layers       int
	Heads        int
	KVHeads      int
	HeadDim      int
	VocabSize    int
	SeqLen       int
	Eps          float32
	RopeTheta    float32
	WindowSize   int

	PrecisionMode PrecisionMode
	KVCacheSize   int

	ExpertCount                   int
	ExpertUsedCount               int
	ExpertSharedCount             int
	ExpertFeedForwardLength       int
	ExpertSharedFeedForwardLength int
	ExpertGroupCount              int
	ExpertGroupUsedCount          int
	ExpertWeightsNorm             bool
	ExpertWeightsScale            float32
	IsMOE                         bool

	DebugDequant     bool
	DebugActivations bool

	DebugEmbedding   bool
	DebugAttention   bool
	DebugFFN         bool
	DebugLayerOutput bool
	DebugLogits      bool

	DebugMemory bool
}

func (c *Config) Validate() error {
	if c.Dim <= 0 {
		return fmt.Errorf("invalid dim: %d (must be positive)", c.Dim)
	}
	if c.Layers <= 0 {
		return fmt.Errorf("invalid layers: %d (must be positive)", c.Layers)
	}
	if c.Heads <= 0 {
		return fmt.Errorf("invalid heads: %d (must be positive)", c.Heads)
	}
	if c.KVHeads <= 0 {
		return fmt.Errorf("invalid kv_heads: %d (must be positive)", c.KVHeads)
	}
	if c.KVHeads > c.Heads {
		return fmt.Errorf("invalid kv_heads: %d (must be <= heads: %d)", c.KVHeads, c.Heads)
	}
	if c.HeadDim <= 0 {
		return fmt.Errorf("invalid head_dim: %d (must be positive)", c.HeadDim)
	}
	if c.Dim != c.Heads*c.HeadDim {
		return fmt.Errorf("dim mismatch: %d != heads(%d) * head_dim(%d)", c.Dim, c.Heads, c.HeadDim)
	}
	if c.VocabSize <= 0 {
		return fmt.Errorf("invalid vocab_size: %d (must be positive)", c.VocabSize)
	}
	if c.SeqLen <= 0 {
		return fmt.Errorf("invalid seq_len: %d (must be positive)", c.SeqLen)
	}
	if c.Eps <= 0 {
		return fmt.Errorf("invalid eps: %f (must be positive)", c.Eps)
	}
	if c.RopeTheta <= 0 {
		return fmt.Errorf("invalid rope_theta: %f (must be positive)", c.RopeTheta)
	}
	if c.WindowSize < 0 {
		return fmt.Errorf("invalid window_size: %d (must be non-negative)", c.WindowSize)
	}
	if c.HiddenDim <= 0 {
		return fmt.Errorf("invalid hidden_dim: %d (must be positive)", c.HiddenDim)
	}

	if c.IsMOE {
		if err := c.validateMOE(); err != nil {
			return err
		}
	}

	return nil
}

func (c *Config) validateMOE() error {
	if c.ExpertCount <= 0 {
		return fmt.Errorf("invalid expert_count: %d (must be positive for MOE)", c.ExpertCount)
	}
	if c.ExpertUsedCount <= 0 {
		return fmt.Errorf("invalid expert_used_count: %d (must be positive for MOE)", c.ExpertUsedCount)
	}
	if c.ExpertUsedCount > c.ExpertCount {
		return fmt.Errorf("expert_used_count (%d) > expert_count (%d)", c.ExpertUsedCount, c.ExpertCount)
	}
	if c.ExpertFeedForwardLength <= 0 {
		return fmt.Errorf("invalid expert_feed_forward_length: %d (must be positive for MOE)", c.ExpertFeedForwardLength)
	}
	return nil
}

func (c *Config) GetArchitecture() string {
	return strings.ToLower(c.Architecture)
}

func (c *Config) IsLargeModel() bool {
	return c.Dim >= 4096
}

func (c *Config) NeedsPagedAttention() bool {
	return c.WindowSize > 0
}

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
