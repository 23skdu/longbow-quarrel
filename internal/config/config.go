package config

import (
	"fmt"
	"strings"
)

// KVCacheType selects the precision of the KV cache pools.
//
// The paged cache reads this when it allocates its pools, so every value
// here is reachable from a shipped binary: the CLI sets it with
// -kv-cache-type and the engines honour it.
//
// The zero value is F16, the precision every backend can read and write.
type KVCacheType int

const (
	KVCacheF16 KVCacheType = iota
	KVCacheF32
	KVCacheTQ1_0
	KVCacheTQ2_0
	// KVCacheFP8 stores the KV cache as FP8-E4M3 with a per-block scale.
	// It halves the cache again relative to F16, at the cost of the
	// quantisation error of a 4-bit exponent format.
	KVCacheFP8
	// KVCacheQ8_0 stores the KV cache as block-quantised 8-bit integers
	// with a per-block scale.
	KVCacheQ8_0
)

// String is the name of the KV cache type as it is spelled on the command
// line.
func (t KVCacheType) String() string {
	switch t {
	case KVCacheF16:
		return "f16"
	case KVCacheF32:
		return "f32"
	case KVCacheTQ1_0:
		return "tq1_0"
	case KVCacheTQ2_0:
		return "tq2_0"
	case KVCacheFP8:
		return "fp8"
	case KVCacheQ8_0:
		return "q8_0"
	default:
		return "unknown"
	}
}

type Config struct {
	Architecture string
	Dim          int
	HiddenDim    int
	Layers       int
	NumGPULayers int // Number of layers offloaded to GPU (-1 or >= Layers for full GPU offload, >0 for partial)
	Heads        int
	KVHeads      int
	HeadDim      int
	VocabSize    int
	SeqLen       int
	// MaxTokens is the number of tokens to generate when a request does
	// not ask for a specific number. It is the default the REST API uses
	// when "max_tokens" is absent or zero.
	MaxTokens  int
	Eps        float32
	RopeTheta  float32
	WindowSize int
	EOSTokenID int
	// StopTokens are the token ids that end a generation. The CLI sets
	// them with -stop-tokens and the REST API accepts them per request.
	StopTokens []int

	KVCacheType KVCacheType
	KVCacheSize int

	ExpertCount                   int
	ExpertUsedCount               int
	ExpertSharedCount             int
	ExpertFeedForwardLength       int
	ExpertSharedFeedForwardLength int
	IsMOE                         bool
	IsHybrid                      bool
	MambaLayerPattern             string

	// DeepSeek MLA (Multi-Head Latent Attention) configuration
	IsMLA bool
	// QLoRARank is the rank of the low-rank query projection (q_lora_rank).
	// When it is greater than zero the model declares a two-stage query
	// projection, and the engine checks the weights against it.
	QLoRARank  int // q_lora_rank
	KVLoRARank int // kv_lora_rank
	QKNDim     int // qk_nope_head_dim
	QKRopeDim  int // qk_rope_head_dim
	VHeadDim   int // v_head_dim

	// Gemma4-specific configuration
	IsGemma4                bool
	Gemma4SlidingWindowSize int     // Default 512 for sliding window layers
	Gemma4SlidingRoPETheta  float32 // Default 10000 for sliding window layers
	Gemma4FullRoPETheta     float32 // Default 1000000 for full attention layers
	Gemma4PartialRoPEFactor float32 // Default 0.25 for full attention layers
	Gemma4SlidingHeadDim    int     // Default 256 for sliding window layers
	Gemma4FullHeadDim       int     // Default 512 for full attention layers
	Gemma4SharedKVLayers    int     // Number of KV layers that are shared (default 18)
	Gemma4SlidingPattern    []bool  // Per-layer sliding window pattern (true=sliding, false=full)
	FinalLogitSoftcapping   float32 // e.g. 30.0 for Gemma4

	// DebugMemory makes the CPU engine report its allocator statistics
	// while it runs.
	DebugMemory bool

	// Phase 2: Continuous Batching & Paged Attention
	MaxBatchSize int
	BlockSize    int
	TotalBlocks  int

	// Memory limits (0 = no limit)
	MaxMemoryMB int64 // Soft limit for total process memory in MB; triggers GC+OOM error

	// Multi-GPU / Distributed Parallelism
	Devices            []int
	TensorParallelSize int
	PipelineStages     int

	// ActivationLogPath is the file the activation log is written to. When
	// it is empty, the activation log is off.
	//
	// The activation log records the embedding, the per-layer
	// activations and the final logits of a request, which is what you
	// need to find the layer that broke.
	ActivationLogPath string

	// Speculative decoding. A smaller "draft" model proposes tokens and the
	// target model verifies them, which trades extra draft compute for
	// fewer target forward passes.
	//
	// SpeculativeDraftModel is the path to the draft model. When it is
	// empty, speculative decoding is off.
	SpeculativeDraftModel  string
	SpeculativeDraftLength int // How many tokens the draft model proposes per step
	SpeculativeNumPaths    int // How many parallel draft paths to generate per step
}

type Gemma4Config struct {
	IsGemma4             bool
	IsSlidingWindowLayer bool
	SlidingWindowSize    int
	SlidingRoPETheta     float32
	FullRoPETheta        float32
	PartialRoPEFactor    float32
	SlidingHeadDim       int
	FullHeadDim          int
	SharedKVLayers       int
	SlidingPattern       []bool
}

func (c *Config) Validate() error {
	if c.TensorParallelSize < 0 {
		return fmt.Errorf("invalid tensor_parallel_size: %d (must be non-negative)", c.TensorParallelSize)
	}
	if c.PipelineStages < 0 {
		return fmt.Errorf("invalid pipeline_stages: %d (must be non-negative)", c.PipelineStages)
	}
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
	if !c.IsGemma4 && !c.IsMLA && c.Dim != c.Heads*c.HeadDim {
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

	if c.IsMLA {
		if err := c.validateMLA(); err != nil {
			return err
		}
	}

	return nil
}

func (c *Config) validateMLA() error {
	if c.KVLoRARank <= 0 {
		return fmt.Errorf("invalid kv_lora_rank: %d (must be positive for MLA)", c.KVLoRARank)
	}
	if c.QKNDim <= 0 {
		return fmt.Errorf("invalid qk_nope_head_dim: %d (must be positive for MLA)", c.QKNDim)
	}
	if c.QKRopeDim <= 0 {
		return fmt.Errorf("invalid qk_rope_head_dim: %d (must be positive for MLA)", c.QKRopeDim)
	}
	if c.VHeadDim <= 0 {
		return fmt.Errorf("invalid v_head_dim: %d (must be positive for MLA)", c.VHeadDim)
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
		SeqLen:      2048,
		MaxTokens:   64,
		Eps:         1e-5,
		RopeTheta:   10000.0,
		EOSTokenID:  2,
		KVCacheType: KVCacheF16,
		DebugMemory: false,

		// Speculative decoding is off unless a draft model is given.
		SpeculativeDraftLength: 4,
		SpeculativeNumPaths:    2,
	}
}
