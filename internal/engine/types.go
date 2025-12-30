package engine

import (
	"github.com/23skdu/longbow-quarrel/internal/device"

	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

type LlamaConfig struct {
	Dim        int
	HiddenDim  int
	Layers     int
	Heads      int
	KVHeads    int
	HeadDim    int // Dim / Heads usually
	VocabSize  int
	SeqLen     int
	Eps        float32
	RopeTheta  float32
	WindowSize int // Sliding window size for attention (4096 for Mistral)
	
	// Debug Flags
	DebugDequant bool
}

type LlamaWeights struct {
	TokenEmb   *device.Tensor // vocab x dim
	
	// Layers
	AttnQ      []*device.Tensor
	AttnK      []*device.Tensor
	AttnV      []*device.Tensor
	AttnO      []*device.Tensor
	
	// AttnNorm   []*device.Tensor
	AttnNorm   []*device.Tensor // Re-added just in case
	
	FfnGate    []*device.Tensor
	FfnDown    []*device.Tensor
	FfnUp      []*device.Tensor
	
	FfnNorm    []*device.Tensor
	
	// Final
	OutputNorm *device.Tensor
	Output     *device.Tensor // vocab x dim (often shared with TokenEmb?)
}

type Engine struct {
	Ctx     *device.Context
	Model   *gguf.GGUFFile
	Config  LlamaConfig
	Weights *LlamaWeights
	
	// KV Cache
	KVCacheK []*device.Tensor // layers x (seq_len x dim) ? No, pre-allocated buffer
	KVCacheV []*device.Tensor
	
	// Cache State
	CachePos int
	
	// Tokenizer
	Tokenizer interface{} // Will be *tokenizer.Tokenizer
	
	// Activation Logger
	ActLogger *ActivationLogger

	// Heuristic Global Scale (1.0 default, 100.0 if detected underscaling)
	GlobalScale float32
}


