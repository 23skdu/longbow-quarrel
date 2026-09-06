//go:build metal



package engine

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
	"github.com/23skdu/longbow-quarrel/internal/logger"
	"github.com/23skdu/longbow-quarrel/internal/metrics"
	"github.com/23skdu/longbow-quarrel/internal/telemetry"
	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
)

var (
	ErrModelSwapped = errors.New("inference interrupted: engine model was swapped")
)

func init() {
	RegisterEngine("metal", NewMetalEngine)
}

func NewMetalEngine(modelPath string, config config.Config) (Engine, error) {
	ctx := device.NewContext()

	e := &metalEngine{
		ctx:       ctx,
		config:    config,
		weights:   &LlamaWeights{},
		ActLogger: NewActivationLogger(),
		SeqMgr:    NewSequenceManager(),
		LoRA:      NewLoRAManager(),
	}

	e.TraceTracker = NewActivationTraceTracker(e.config.Layers)
	e.BatchManager = NewContinuousBatchManager()
	e.PromptCache = NewPromptCache()
	
	e.stopChan = make(chan struct{})
	e.doneChan = make(chan struct{})

	if err := e.loadModel(modelPath); err != nil {
		ctx.Free()
		return nil, err
	}

	e.cache = &PagedKVCache{}
	e.cache.Init(ctx, e.config)

	go e.runBatchLoop()

	return e, nil
}

// Internal load method
// Internal load method

func (e *metalEngine) loadModel(path string) error {
	logger.Log.Debug("loadModel starting", "path", path, "ctx_is_nil", e.ctx == nil, "config", fmt.Sprintf("%+v", e.config))
	f, err := gguf.LoadFile(path)
	if err != nil {
		return err
	}
	e.model = f

	// Direct lookup from KV
	archVal := f.KV["general.architecture"]
	embVal := f.KV["gemma4.embedding_length"]
	headVal := f.KV["gemma4.attention.head_count"]
	logger.Log.Debug("KV metadata discovered", "arch", archVal, "emb", embVal, "head", headVal)
	tok, err := tokenizer.NewFromGGUF(f)
	if err != nil {
		logger.Log.Warn("Failed to initialize tokenizer from GGUF", "error", err)
	} else {
		e.Tokenizer = tok
	}

	// ... (metadata loading remains same)

	// Read Metadata
	// Block Count
	if val, ok := getKV(f,
		"llama.block_count",
		"qwen3moe.block_count",
		"gemma4.block_count",
		"qwen2.block_count",
	); ok {
		e.config.Layers = int(toFloat64(val)) // GGUF numbers can be various types
	} else {
		e.config.Layers = 1 // Default for test
	}

	// 1. Embedding Dim
	if val, ok := getKV(f, "llama.embedding_length", "gemma4.embedding_length", "qwen2.embedding_length"); ok {
		e.config.Dim = int(toFloat64(val))
	}
	if e.config.Dim <= 0 {
		e.config.Dim = 4096 // Fallback
	}

	// 2. Heads (Attention)
	if val, ok := getKV(f, "llama.attention.head_count", "gemma4.attention.head_count", "qwen2.attention.head_count"); ok {
		e.config.Heads = int(toFloat64(val))
	}
	if e.config.Heads <= 0 {
		e.config.Heads = 32 // Fallback
	}

	// 3. KV Heads (GQA/MHA)
	if val, ok := getKV(f, "llama.attention.head_count_kv", "gemma4.attention.head_count_kv", "gemma4.attention.kv_head_count"); ok {
		if arr, ok := val.([]interface{}); ok {
			maxVal := 0
			for _, v := range arr {
				iv := int(toFloat64(v))
				if iv > maxVal { maxVal = iv }
			}
			e.config.KVHeads = maxVal
		} else {
			e.config.KVHeads = int(toFloat64(val))
		}
	}
	if e.config.KVHeads <= 0 {
		e.config.KVHeads = e.config.Heads // Default to MHA
	}

	// 4. Derived Dimensions
	if e.config.Heads > 0 {
		e.config.HeadDim = e.config.Dim / e.config.Heads
	}
	if e.config.HeadDim <= 0 {
		e.config.HeadDim = 128 // Fallback
	}

	// 5. Hidden Dim (FFN)
	if val, ok := getKV(f, "llama.feed_forward_length", "gemma4.feed_forward_length", "qwen2.feed_forward_length"); ok {
		e.config.HiddenDim = int(toFloat64(val))
	}
	if e.config.HiddenDim <= 0 {
		e.config.HiddenDim = 4 * e.config.Dim // Fallback
	}

	logger.Log.Info("Model dimensions loaded", "dim", e.config.Dim, "heads", e.config.Heads, "kv_heads", e.config.KVHeads, "head_dim", e.config.HeadDim, "hidden_dim", e.config.HiddenDim)

	// Seq Len (Context)
	if val, ok := getKV(f,
		"llama.context_length",
		"qwen3moe.context_length",
		"gemma4.context_length",
		"qwen2.context_length",
	); ok {
		e.config.SeqLen = int(toFloat64(val))
	} else {
		e.config.SeqLen = 2048 // default
	}
	logger.Log.Info("Model sequence length loaded", "seq_len", e.config.SeqLen)

	// RoPE Freq
	if val, ok := getKV(f,
		"llama.rope.freq_base",
		"qwen3moe.rope.freq_base",
		"gemma4.rope.freq_base",
		"gemma4.rope.freq_base_swa",
		"qwen2.rope.freq_base",
	); ok {
		e.config.RopeTheta = float32(toFloat64(val))
		logger.Log.Info("Model RoPE theta loaded", "theta", e.config.RopeTheta)
	} else {
		// Default to 10k for Llama 2, but Mistral v0.3 uses 1M
		// We'll set it properly based on architecture below
		e.config.RopeTheta = 10000.0
	}

	// Global Scale
	e.GlobalScale = 1.0

	// RMS Norm Eps
	var eps float32
	if val, ok := getKV(f, "llama.attention.layer_norm_rms_epsilon", "qwen3moe.attention.layer_norm_rms_epsilon"); ok {
		eps = float32(toFloat64(val))
	} else {
		eps = 1e-5 // default
	}
	e.config.Eps = eps

	// Sliding Window Size (for Mistral)
	// Mistral uses 4096-token sliding window attention
	// If not specified in GGUF, default to 4096 for Mistral, 0 (disabled) for others
	if val, ok := getKV(f, "llama.attention.sliding_window", ""); ok {
		e.config.WindowSize = int(toFloat64(val))
		logger.Log.Info("Model sliding window size loaded", "window_size", e.config.WindowSize)
	} else {
		// Check if this is Mistral (heuristic: has specific architecture name)
		if arch, ok := f.KV["general.architecture"].(string); ok && arch == "llama" {
			// For Mistral models, default to 4096
			// For other models (Llama, etc.), use 0 (full attention)
			e.config.WindowSize = 4096
			logger.Log.Info("Model heuristic: using Mistral sliding window default", "window_size", e.config.WindowSize)
		} else {
			e.config.WindowSize = 0 // Full attention
		}
	}

	// Log Model Architecture
	if arch, ok := f.KV["general.architecture"].(string); ok {
		logger.Log.Info("Model architecture detected", "arch", arch)

		// Heuristic: If it's llama or mistral and WindowSize not set,
		// check if we should assume Mistral SWA.
		if strings.Contains(strings.ToLower(arch), "llama") || strings.Contains(strings.ToLower(arch), "mistral") {
			if e.config.WindowSize == 0 {
				// Most Mistral models in GGUF don't have the sliding_window key set correctly,
				// but they still require it for correctness if they are v0.3+.
				// However, Llama 2 also uses "llama" arch.
				// Heuristic: If RopeTheta is 1M, it's likely Mistral v0.3.
				if e.config.RopeTheta >= 1000000.0 {
					e.config.WindowSize = 4096
					logger.Log.Info("Model heuristic: Mistral v0.3 detected, using 4096 SWA")
				}
			}
		}
	}
	if val, ok := getKV(f, "llama.vocab_size", ""); ok {
		e.config.VocabSize = int(toFloat64(val))
		logger.Log.Info("Model vocab size loaded", "vocab_size", e.config.VocabSize)
	} else {
		// Fallback for Smollm2 / Llama3 if missing
		e.config.VocabSize = 49152
		logger.Log.Info("Model vocab size default", "vocab_size", e.config.VocabSize)
	}

	// MOE Metadata (Mixture of Experts)
	if val, ok := getKV(f, "llama.expert_count", ""); ok {
		e.config.ExpertCount = int(toFloat64(val))
		e.config.IsMOE = true
		logger.Log.Info("MOE architecture detected", "expert_count", e.config.ExpertCount)
	}

	if val, ok := getKV(f, "llama.expert_used_count", ""); ok {
		e.config.ExpertUsedCount = int(toFloat64(val))
		logger.Log.Info("MOE expert usage", "expert_used_count", e.config.ExpertUsedCount)
	}

	if val, ok := getKV(f, "llama.expert_shared_count", ""); ok {
		e.config.ExpertSharedCount = int(toFloat64(val))
		logger.Log.Info("MOE shared experts", "expert_shared_count", e.config.ExpertSharedCount)
	}

	if val, ok := getKV(f, "llama.expert_feed_forward_length", ""); ok {
		e.config.ExpertFeedForwardLength = int(toFloat64(val))
		logger.Log.Info("MOE expert FFN dimension", "expert_feed_forward_length", e.config.ExpertFeedForwardLength)
	}

	if val, ok := getKV(f, "llama.expert_shared_feed_forward_length", ""); ok {
		e.config.ExpertSharedFeedForwardLength = int(toFloat64(val))
		logger.Log.Info("MOE shared expert FFN dimension", "expert_shared_feed_forward_length", e.config.ExpertSharedFeedForwardLength)
	}

	if val, ok := getKV(f, "llama.expert_group_count", ""); ok {
		e.config.ExpertGroupCount = int(toFloat64(val))
	}

	if val, ok := getKV(f, "llama.expert_group_used_count", ""); ok {
		e.config.ExpertGroupUsedCount = int(toFloat64(val))
	}

	if val, ok := getKV(f, "llama.expert_weights_norm", ""); ok {
		if norm, ok := val.(bool); ok {
			e.config.ExpertWeightsNorm = norm
		}
	}

	if val, ok := getKV(f, "llama.expert_weights_scale", ""); ok {
		e.config.ExpertWeightsScale = float32(toFloat64(val))
	}

	// Log MOE configuration summary if MOE is detected
	if e.config.IsMOE {
		logger.Log.Info("MOE configuration summary",
			"expert_count", e.config.ExpertCount,
			"expert_used_count", e.config.ExpertUsedCount,
			"expert_shared_count", e.config.ExpertSharedCount,
			"expert_ffn_dim", e.config.ExpertFeedForwardLength,
			"expert_shared_ffn_dim", e.config.ExpertSharedFeedForwardLength)
	}

	// Set Precision Mode based on model dimensions (explicit configuration instead of heuristic)
	// Architecture detection
	if arch, ok := f.KV["general.architecture"].(string); ok {
		e.config.Architecture = arch
		if arch == "nemo" || arch == "nemotron" {
			e.config.IsMOE = true
			logger.Log.Debug("Architecture detected as MOE (Nemotron)", "arch", arch)
		}
		// Gemma4 detection
		if arch == "gemma4" {
			e.config.IsGemma4 = true
			e.config.Gemma4SlidingWindowSize = 512
			e.config.Gemma4SlidingRoPETheta = 10000.0
			e.config.Gemma4FullRoPETheta = 1000000.0
			e.config.Gemma4PartialRoPEFactor = 0.25
			e.config.Gemma4SlidingHeadDim = 256
			e.config.Gemma4FullHeadDim = 512
			e.config.FinalLogitSoftcapping = 30.0
			e.config.Eps = 1e-6
			logger.Log.Info("Gemma4 architecture detected", "arch", arch)
		}
		logger.Log.Info("Model architecture confirmed", "arch", arch)
	}
	// MOE-specific metadata
	if val, ok := f.KV["llama.expert_count"].(uint32); ok {
		e.config.ExpertCount = int(val)
		e.config.IsMOE = true
		logger.Log.Debug("Architecture detected as MOE (llama.expert_count)", "count", val)
	}

	// Mamba layer detection for hybrid models
	e.detectMambaLayers(f, *logger.Log)

	if val, ok := f.KV["llama.attention.precision"]; ok {
		if prec, ok := val.(string); ok {
			switch prec {
			case "f16":
				e.config.PrecisionMode = config.PrecisionFP16
				logger.Log.Info("Model precision mode set (metadata)", "mode", "FP16")
			case "f32":
				e.config.PrecisionMode = config.PrecisionF32FFN
				logger.Log.Info("Model precision mode set (metadata)", "mode", "F32_FFN")
			case "mixed":
				e.config.PrecisionMode = config.PrecisionMixed
				logger.Log.Info("Model precision mode set (metadata)", "mode", "Mixed")
			default:
				logger.Log.Info("Model unknown precision mode, using auto", "mode", prec)
			}
		} else {
			e.config.PrecisionMode = config.PrecisionAuto
		}
	} else {
		// Auto-detect based on dimension
		if e.config.Dim < 1024 {
			e.config.PrecisionMode = config.PrecisionF32FFN
			logger.Log.Info("Model precision mode auto-detected", "mode", "F32_FFN", "dim", e.config.Dim)
		} else if e.config.Dim >= 4096 {
			e.config.PrecisionMode = config.PrecisionMixed
			logger.Log.Info("Model precision mode auto-detected", "mode", "Mixed", "dim", e.config.Dim)
		} else {
			e.config.PrecisionMode = config.PrecisionFP16
			logger.Log.Info("Model precision mode auto-detected", "mode", "FP16", "dim", e.config.Dim)
		}
	}

	if val, ok := f.KV["tokenizer.ggml.bos_token_id"]; ok {
		logger.Log.Info("Model BOS token loaded", "id", int(toFloat64(val)))
	}
	if val, ok := f.KV["tokenizer.ggml.eos_token_id"]; ok {
		logger.Log.Info("Model EOS token loaded", "id", int(toFloat64(val)))
	}

	// Initialize KV Cache (now that we have dimensions)
	if err := e.initKVCache(); err != nil {
		return err
	}

	logger.Log.Info("Model configuration summary",
		"layers", e.config.Layers,
		"dim", e.config.Dim,
		"hidden_dim", e.config.HiddenDim,
		"heads", e.config.Heads,
		"kv_heads", e.config.KVHeads,
		"head_dim", e.config.HeadDim,
		"eps", e.config.Eps,
		"rope_theta", e.config.RopeTheta)

	// Initialize Weights Slices
	layers := e.config.Layers
	e.weights.AttnQ = make([]*device.Tensor, layers)
	e.weights.AttnK = make([]*device.Tensor, layers)
	e.weights.AttnV = make([]*device.Tensor, layers)
	e.weights.AttnO = make([]*device.Tensor, layers)
	e.weights.AttnNorm = make([]*device.Tensor, layers)
	e.weights.FfnGate = make([]*device.Tensor, layers)
	e.weights.FfnDown = make([]*device.Tensor, layers)
	e.weights.FfnUp = make([]*device.Tensor, layers)
	e.weights.FfnNorm = make([]*device.Tensor, layers)
	e.weights.AttnQNorm = make([]*device.Tensor, layers)
	e.weights.AttnKNorm = make([]*device.Tensor, layers)
	e.weights.Mamba = make([]*MambaWeights, layers) // Initialize Mamba/SSM layer storage

	// Initialize MOE weight containers if MOE architecture detected
	if e.config.IsMOE {
		e.weights.MOE = make([]*MOELayerWeights, layers)
		logger.Log.Info("Initialized MOE weight containers", "layers", layers)
	}

	// Map tensors
	for _, t := range f.Tensors {
		if !isNeededTensor(t.Name) {
			continue
		}
		logger.Log.Debug("Processing tensor", "name", t.Name, "engine_ptr", fmt.Sprintf("%p", e), "ctx_ptr", fmt.Sprintf("%p", e.ctx))

		cols := int(t.Dimensions[0])
		rows := 1
		if len(t.Dimensions) > 1 {
			rows = int(t.Dimensions[1])
		}
		for i := 2; i < len(t.Dimensions); i++ {
			rows *= int(t.Dimensions[i])
		}

		numElements := rows * cols
		var mt *device.Tensor

		switch t.Type {

		case gguf.GGMLTypeF32:
			mt = e.ctx.NewTensorFP32(rows, cols)
			dataBytes := numElements * 4
			if uint64(len(t.Data)) < uint64(dataBytes) {
				return fmt.Errorf("tensor %s data truncated", t.Name)
			}
			if err := mt.LoadFromRaw(t.Data[:dataBytes]); err != nil {
				return err
			}
			e.GlobalScale = 1.0

		case gguf.GGMLTypeF16:
			if isNormWeight(t.Name) {
				// Promote Norm weights to FP32 for precision and kernel compatibility
				f32Data := gguf.DequantizeF16(t.Data, numElements)
				mt = e.ctx.NewTensorFP32(rows, cols)
				if err := mt.LoadFrom(f32Data); err != nil {
					return err
				}

			} else {
				mt = e.ctx.NewTensorWithType(rows, cols, device.DataTypeF16)
				// ... F16 load ...
				dataBytes := numElements * 2
				if uint64(len(t.Data)) < uint64(dataBytes) {
					return fmt.Errorf("tensor %s data truncated", t.Name)
				}
				if err := mt.LoadFromRaw(t.Data[:dataBytes]); err != nil {
					return err
				}
			}
		case gguf.GGMLTypeQ4_K:
			// Type 12 (Q4_K).
			if e.config.Dim < 1024 {
				f32Data := gguf.DequantizeQ4K(t.Data, numElements)
				mt = e.ctx.NewTensorWithType(rows, cols, device.DataTypeF16)
				mt.LoadFrom(f32Data)
			} else {
				// Large Models: Use Q4K Tensor and Kernels
				if e.ctx == nil {
					return fmt.Errorf("engine context is nil before creating Q4K tensor %s", t.Name)
				}
				var err error
				mt, err = e.ctx.NewQ4KTensor(rows, cols)
				if err != nil {
					return fmt.Errorf("failed to create Q4K tensor %s: %w", t.Name, err)
				}
				dataBytes := (numElements / 256) * 144
				// Check size matches (handle truncated data)
				if uint64(len(t.Data)) < uint64(dataBytes) {
					return fmt.Errorf("tensor %s data truncated (Need %d, Has %d)", t.Name, dataBytes, len(t.Data))
				}

				if err := mt.LoadFromRaw(t.Data[:dataBytes]); err != nil {
					return err
				}
			}
		case gguf.GGMLTypeQ4_0:
			// Type 2 (Q4_0).
			// Always use native Q4_0 kernel
			// rows * cols elements.
			// Block size 32. 18 bytes per block.
			// Check alignment
			if numElements%32 != 0 {
				return fmt.Errorf("Q4_0 tensor %s size %d not divisible by 32", t.Name, numElements)
			}
			mt = e.ctx.NewTensorWithType(rows, cols, device.DataTypeQ4_0)
			dataBytes := (numElements / 32) * 18
			if uint64(len(t.Data)) < uint64(dataBytes) {
				return fmt.Errorf("tensor %s data truncated (Need %d, Has %d)", t.Name, dataBytes, len(t.Data))
			}
			if err := mt.LoadFromRaw(t.Data[:dataBytes]); err != nil {
				return err
			}

		case gguf.GGMLTypeQ8_0:
			// Type 8 (Q8_0).
			// Dequantize to F16 for use in engine
			f32Data := gguf.DequantizeQ8_0(t.Data, numElements)
			mt = e.ctx.NewTensorWithType(rows, cols, device.DataTypeF16)
			if err := mt.LoadFrom(f32Data); err != nil {
				return err
			}

		case gguf.GGMLTypeQ6_K:
			// Type 14 (Q6_K).
			if t.Name == "token_embd.weight" {
				logger.Log.Debug("token_embd.weight dequantizing to FP16")
				f32Data := gguf.DequantizeQ6K(t.Data, numElements)
				mt = e.ctx.NewTensorWithType(rows, cols, device.DataTypeF16)
				if err := mt.LoadFrom(f32Data); err != nil {
					return err
				}
			} else if t.Name == "output.weight" || e.config.Dim >= 1024 {
				// Use Native Q6K
				mt = e.ctx.NewTensorWithType(rows, cols, device.DataTypeQ6K)
				dataBytes := (numElements / 256) * 210
				if uint64(len(t.Data)) < uint64(dataBytes) {
					return fmt.Errorf("tensor %s data truncated", t.Name)
				}
				if err := mt.LoadFromRaw(t.Data[:dataBytes]); err != nil {
					return err
				}
			} else {
				f32Data := gguf.DequantizeQ6K(t.Data, numElements)
				mt = e.ctx.NewTensorWithType(rows, cols, device.DataTypeF16)
				if err := mt.LoadFrom(f32Data); err != nil {
					return err
				}
			}

		case gguf.GGMLTypeQ4_K_S: // 99 Unused
			mt = e.ctx.NewTensor(rows, cols) // fallback
		case gguf.GGMLTypeQ5_0:
			// Type 6 (Q5_0). Dequantize to F16 for use in engine
			f32Data := gguf.DequantizeQ5_0(t.Data, numElements)
			mt = e.ctx.NewTensorWithType(rows, cols, device.DataTypeF16)
			if err := mt.LoadFrom(f32Data); err != nil {
				return err
			}
		case gguf.GGMLTypeQ5_K:
			// Type 13 (Q5_K). Dequantize to F16 for use in engine (use Q5_0 as fallback)
			f32Data := gguf.DequantizeQ5_0(t.Data, numElements)
			mt = e.ctx.NewTensorWithType(rows, cols, device.DataTypeF16)
			if err := mt.LoadFrom(f32Data); err != nil {
				return err
			}
		default:
			continue
		}

		if mt != nil {
			if t.Type == gguf.GGMLTypeF16 || t.Type == gguf.GGMLTypeF32 || t.Type == gguf.GGMLTypeQ6_K || t.Name == "token_embd.weight" {

			} else if t.Type == gguf.GGMLTypeQ4_K {
				mt.ScanQ4KScales(t.Name)
			}
		}

		// Mapping Logic
		name := t.Name
		logger.Log.Debug("loading tensor", "name", name, "dims", t.Dimensions, "type", t.Type, "offset", t.Offset)

		// DEBUG: Check if this is a Gemma4 norm tensor
		if strings.Contains(name, "attn_q_norm") || strings.Contains(name, "attn_k_norm") {
			logger.Log.Debug("found Gemma4 Q/K norm tensor", "name", name)
		}

		// 1. Global weights (supporting prefixes like nemotron., model., etc.)
		// Strict check to avoid matching blk.N.attn_output.weight to global output.weight
		lowerName := strings.ToLower(name)
		if (strings.HasSuffix(lowerName, "token_embd.weight") ||
			strings.HasSuffix(lowerName, "embed_tokens.weight") ||
			lowerName == "model.embed_tokens.weight") && !strings.Contains(lowerName, "blk.") {
			e.weights.TokenEmb = mt
			continue
		}
		if (strings.HasSuffix(name, "output_norm.weight") || strings.HasSuffix(name, "model.norm.weight")) && !strings.Contains(name, "blk.") {
			e.weights.OutputNorm = mt
			continue
		}
		if (strings.HasSuffix(name, "output.weight") || name == "model.lm_head.weight") && !strings.Contains(name, "blk.") {
			e.weights.Output = mt
			continue
		}

		// 2. Layer weights: blk.N.suffix
		if strings.HasPrefix(name, "blk.") {
			parts := strings.Split(name, ".")
			if len(parts) < 3 {
				continue
			}

			// Parse N
			layerIdx := 0
			if n, err := fmt.Sscanf(parts[1], "%d", &layerIdx); n != 1 || err != nil {
				continue
			}
			if layerIdx >= layers {
				continue
			}

			suffix := strings.Join(parts[2:], ".")

			switch suffix {
			case "attn_q.weight":
				e.weights.AttnQ[layerIdx] = mt
			case "attn_k.weight":
				e.weights.AttnK[layerIdx] = mt
			case "attn_v.weight":
				e.weights.AttnV[layerIdx] = mt
			case "attn_output.weight":
				e.weights.AttnO[layerIdx] = mt
			case "attn_norm.weight":
				e.weights.AttnNorm[layerIdx] = mt

			// Gemma-specific: attn_q_norm, attn_k_norm (RMSNorm before Q/K)
			case "attn_q_norm.weight":
				logger.Log.Debug("setting AttnQNorm", "layer", layerIdx, "name", name)
				if e.weights.AttnQNorm == nil || len(e.weights.AttnQNorm) <= layerIdx {
					newSlice := make([]*device.Tensor, layerIdx+1)
					copy(newSlice, e.weights.AttnQNorm)
					e.weights.AttnQNorm = newSlice
				}
				e.weights.AttnQNorm[layerIdx] = mt
			case "attn_k_norm.weight":
				logger.Log.Debug("setting AttnKNorm", "layer", layerIdx, "name", name)
				if e.weights.AttnKNorm == nil || len(e.weights.AttnKNorm) <= layerIdx {
					newSlice := make([]*device.Tensor, layerIdx+1)
					copy(newSlice, e.weights.AttnKNorm)
					e.weights.AttnKNorm = newSlice
				}
				e.weights.AttnKNorm[layerIdx] = mt

			case "ffn_gate.weight":
				e.weights.FfnGate[layerIdx] = mt
				if layerIdx == 0 {
					e.config.HiddenDim = rows
				}
				continue

			// Gemma-specific: inp_gate (input gating)
			case "inp_gate.weight":
				e.weights.FfnGate[layerIdx] = mt

			// Gemma-specific: proj (output projection)
			case "proj.weight":
				e.weights.AttnO[layerIdx] = mt

			case "ffn_down.weight":
				e.weights.FfnDown[layerIdx] = mt
				if layerIdx <= 5 {
					if mt.DataType() == device.DataTypeQ4K {
						logger.Log.Debug("FFN Down weight quantized", "layer", layerIdx, "type", "Q4K")
						mt.ScanQ4KScales(fmt.Sprintf("blk.%d.ffn_down", layerIdx))
					} else {
						logger.Log.Debug("FFN Down weight loaded", "layer", layerIdx, "type", "F16", "rows", mt.Rows(), "cols", mt.Cols())
					}
				}
			case "ffn_up.weight":
				e.weights.FfnUp[layerIdx] = mt
			case "ffn_norm.weight":
				e.weights.FfnNorm[layerIdx] = mt

			// Gemma-specific: post_*_norm weights
			case "post_attention_norm.weight":
				e.weights.AttnNorm[layerIdx] = mt
			case "post_ffw_norm.weight":
				e.weights.FfnNorm[layerIdx] = mt
			case "post_norm.weight":
				e.weights.FfnNorm[layerIdx] = mt

			// Mamba/SSM Tensors
			case "ssm_a":
				if e.weights.Mamba[layerIdx] == nil {
					e.weights.Mamba[layerIdx] = &MambaWeights{}
				}
				e.weights.Mamba[layerIdx].A = mt
			case "ssm_d":
				if e.weights.Mamba[layerIdx] == nil {
					e.weights.Mamba[layerIdx] = &MambaWeights{}
				}
				e.weights.Mamba[layerIdx].D = mt
			case "ssm_conv1d.weight":
				if e.weights.Mamba[layerIdx] == nil {
					e.weights.Mamba[layerIdx] = &MambaWeights{}
				}
				e.weights.Mamba[layerIdx].Conv1dWeight = mt
			case "ssm_conv1d.bias":
				if e.weights.Mamba[layerIdx] == nil {
					e.weights.Mamba[layerIdx] = &MambaWeights{}
				}
				e.weights.Mamba[layerIdx].Conv1dBias = mt
			case "ssm_dt.weight":
				if e.weights.Mamba[layerIdx] == nil {
					e.weights.Mamba[layerIdx] = &MambaWeights{}
				}
				e.weights.Mamba[layerIdx].DTWeight = mt
			case "ssm_dt.bias":
				if e.weights.Mamba[layerIdx] == nil {
					e.weights.Mamba[layerIdx] = &MambaWeights{}
				}
				e.weights.Mamba[layerIdx].DTBias = mt
			case "ssm_norm.weight":
				if e.weights.Mamba[layerIdx] == nil {
					e.weights.Mamba[layerIdx] = &MambaWeights{}
				}
				e.weights.Mamba[layerIdx].NormWeight = mt
			case "ssm_norm.bias":
				if e.weights.Mamba[layerIdx] == nil {
					e.weights.Mamba[layerIdx] = &MambaWeights{}
				}
				e.weights.Mamba[layerIdx].NormBias = mt
			case "ssm_out.weight":
				if e.weights.Mamba[layerIdx] == nil {
					e.weights.Mamba[layerIdx] = &MambaWeights{}
				}
				e.weights.Mamba[layerIdx].OutWeight = mt
			case "ssm_in.weight": // Hypothetical, not seen in logs yet
				if e.weights.Mamba[layerIdx] == nil {
					e.weights.Mamba[layerIdx] = &MambaWeights{}
				}
				e.weights.Mamba[layerIdx].InWeight = mt

			// MOE Router Weights
			case "ffn_gate_inp.weight":
				if e.weights.MOE[layerIdx] == nil {
					e.weights.MOE[layerIdx] = &MOELayerWeights{}
				}
				if e.weights.MOE[layerIdx].Router == nil {
					e.weights.MOE[layerIdx].Router = &MOERouterWeights{}
				}
				e.weights.MOE[layerIdx].Router.GateInput = mt
				logger.Log.Debug("Loaded MOE router gate", "layer", layerIdx, "shape", fmt.Sprintf("[%d, %d]", mt.Rows(), mt.Cols()))

			case "exp_probs_b.bias":
				if e.weights.MOE[layerIdx] == nil {
					e.weights.MOE[layerIdx] = &MOELayerWeights{}
				}
				if e.weights.MOE[layerIdx].Router == nil {
					e.weights.MOE[layerIdx].Router = &MOERouterWeights{}
				}
				e.weights.MOE[layerIdx].Router.ExpertProbBias = mt

			// MOE Expert Weights (3D tensors)
			case "ffn_down_exps.weight":
				if e.weights.MOE[layerIdx] == nil {
					e.weights.MOE[layerIdx] = &MOELayerWeights{}
				}
				if e.weights.MOE[layerIdx].Experts == nil {
					e.weights.MOE[layerIdx].Experts = &MOEExpertWeights{}
				}
				e.weights.MOE[layerIdx].Experts.FfnDownExperts = mt
				// Extract 3D metadata from tensor dimensions [hidden_dim, dim, num_experts]
				if len(t.Dimensions) == 3 {
					e.weights.MOE[layerIdx].Experts.HiddenDim = int(t.Dimensions[0])
					e.weights.MOE[layerIdx].Experts.Dim = int(t.Dimensions[1])
					e.weights.MOE[layerIdx].Experts.NumExperts = int(t.Dimensions[2])
					logger.Log.Debug("Loaded MOE expert down weights", "layer", layerIdx,
						"hidden_dim", t.Dimensions[0], "dim", t.Dimensions[1], "num_experts", t.Dimensions[2])
				} else {
					logger.Log.Debug("Loaded MOE expert down weights", "layer", layerIdx, "shape", t.Dimensions)
				}

			case "ffn_up_exps.weight":
				if e.weights.MOE[layerIdx] == nil {
					e.weights.MOE[layerIdx] = &MOELayerWeights{}
				}
				if e.weights.MOE[layerIdx].Experts == nil {
					e.weights.MOE[layerIdx].Experts = &MOEExpertWeights{}
				}
				e.weights.MOE[layerIdx].Experts.FfnUpExperts = mt
				// Extract 3D metadata if not already set
				if len(t.Dimensions) == 3 && e.weights.MOE[layerIdx].Experts.NumExperts == 0 {
					e.weights.MOE[layerIdx].Experts.HiddenDim = int(t.Dimensions[0])
					e.weights.MOE[layerIdx].Experts.Dim = int(t.Dimensions[1])
					e.weights.MOE[layerIdx].Experts.NumExperts = int(t.Dimensions[2])
					logger.Log.Debug("Loaded MOE expert up weights", "layer", layerIdx,
						"hidden_dim", t.Dimensions[0], "dim", t.Dimensions[1], "num_experts", t.Dimensions[2])
				} else {
					logger.Log.Debug("Loaded MOE expert up weights", "layer", layerIdx, "shape", t.Dimensions)
				}

			case "ffn_gate_exps.weight":
				if e.weights.MOE[layerIdx] == nil {
					e.weights.MOE[layerIdx] = &MOELayerWeights{}
				}
				if e.weights.MOE[layerIdx].Experts == nil {
					e.weights.MOE[layerIdx].Experts = &MOEExpertWeights{}
				}
				e.weights.MOE[layerIdx].Experts.FfnGateExperts = mt

			// MOE Shared Expert Weights
			case "ffn_down_shexp.weight":
				if e.weights.MOE[layerIdx] == nil {
					e.weights.MOE[layerIdx] = &MOELayerWeights{}
				}
				if e.weights.MOE[layerIdx].Shared == nil {
					e.weights.MOE[layerIdx].Shared = &MOESharedWeights{}
				}
				e.weights.MOE[layerIdx].Shared.FfnDownShared = mt
				logger.Log.Debug("Loaded MOE shared expert down weights", "layer", layerIdx, "shape", fmt.Sprintf("[%d, %d]", mt.Rows(), mt.Cols()))

			case "ffn_up_shexp.weight":
				if e.weights.MOE[layerIdx] == nil {
					e.weights.MOE[layerIdx] = &MOELayerWeights{}
				}
				if e.weights.MOE[layerIdx].Shared == nil {
					e.weights.MOE[layerIdx].Shared = &MOESharedWeights{}
				}
				e.weights.MOE[layerIdx].Shared.FfnUpShared = mt
				logger.Log.Debug("Loaded MOE shared expert up weights", "layer", layerIdx, "shape", fmt.Sprintf("[%d, %d]", mt.Rows(), mt.Cols()))

			case "ffn_gate_shexp.weight":
				if e.weights.MOE[layerIdx] == nil {
					e.weights.MOE[layerIdx] = &MOELayerWeights{}
				}
				if e.weights.MOE[layerIdx].Shared == nil {
					e.weights.MOE[layerIdx].Shared = &MOESharedWeights{}
				}
				e.weights.MOE[layerIdx].Shared.FfnGateShared = mt
			}
		}
	}

	// Fallback: many models share token_embd.weight with output.weight (Tied Embeddings)
	// Case 1: TokenEmb exists, Output missing -> Use TokenEmb as Output
	if e.weights.TokenEmb != nil && e.weights.Output == nil {
		e.weights.Output = e.weights.TokenEmb
		logger.Log.Debug("Tied output.weight to token_embd.weight")
	}
	// Case 2: Output exists, TokenEmb missing (e.g. Nemotron) -> Use Output as TokenEmb
	if e.weights.TokenEmb == nil && e.weights.Output != nil {
		e.weights.TokenEmb = e.weights.Output
		logger.Log.Info("Using output.weight as token_embd.weight (tied embeddings recovery)")
	}

	// Update VocabSize based on actual tensor rows (the source of truth)
	if e.weights.TokenEmb != nil {
		actualVocab := e.weights.TokenEmb.Rows()
		if actualVocab != e.config.VocabSize {
			logger.Log.Warn("Correcting Vocab Size (from embedding)", "configured", e.config.VocabSize, "actual", actualVocab)
			e.config.VocabSize = actualVocab
		}
	}

	// Case 3: Gap recovery for missing Mamba ssm_in tensors (Nemotron specific)
	if e.weights.Mamba != nil {
		gaps := f.GetGapTensors()
		logger.Log.Debug("Gap recovery triggered", "num_gaps", len(gaps))
		for idx, gap := range gaps {
			logger.Log.Debug("Found GGUF gap", "idx", idx, "offset", gap.Offset, "size", len(gap.Data))
		}
		gapIdx := 0
		for i := 0; i < e.config.Layers; i++ {
			mw := e.weights.Mamba[i]
			if mw != nil && mw.InWeight == nil {
				// Search for a gap that matches the expected size (approx)
				for gapIdx < len(gaps) {
					gap := gaps[gapIdx]
					gapIdx++
					if len(gap.Data) < 10*1024*1024 {
						continue
					}
					rows := 6144
					cols := 2688
					logger.Log.Info("Recovering missing ssm_in weight from GGUF gap", "layer", i, "offset", gap.Offset, "size", len(gap.Data))
					mt, err := e.ctx.NewTensorFromData(rows, cols, device.DataTypeQ8_0, gap.Data)
					if err != nil {
						logger.Log.Warn("Failed to recover ssm_in weight from gap", "error", err)
						continue
					}
					mw.InWeight = mt
					logger.Log.Info("Successfully recovered ssm_in weight from gap", "layer", i)
					break // Found one for this layer
				}
			}
		}
	}

	// Initialize Mamba Layers and Cache
	e.MambaLayers = make([]*MambaLayer, e.config.Layers)
	e.SSMCache = make([]*MambaState, e.config.Layers)

	for i := 0; i < e.config.Layers; i++ {
		if e.weights.Mamba[i] != nil {
			// Found Mamba weights for this layer
			layer := &MambaLayer{
				Index:   i,
				Weights: e.weights.Mamba[i],
			}
			e.MambaLayers[i] = layer

			dConv := 6144
			kernelSize := 4
			dInner := 4096
			dState := 64

			if layer.Weights.Conv1dWeight != nil {
				dConv = layer.Weights.Conv1dWeight.Rows()
				kernelSize = layer.Weights.Conv1dWeight.Cols()
			}
			if layer.Weights.OutWeight != nil {
				dInner = layer.Weights.OutWeight.Cols()
			}

			convState := e.ctx.NewTensorPooled(dConv, kernelSize)
			ssmState := e.ctx.NewTensorPooled(dInner, dState)

			convState.ZeroInit()
			ssmState.ZeroInit()

			e.SSMCache[i] = &MambaState{
				ConvState: convState,
				SSMState:  ssmState,
			}
		}
	}

	return nil
}

// InferString is a convenience method that takes a string prompt and returns generated text
func (e *metalEngine) InferString(prompt string, tokensToGenerate int) (string, error) {
	samplerConfig := SamplerConfig{
		Temperature:      0.7,
		TopK:             40,
		TopP:             0.9,
		RepPenalty:       1.0,
		Seed:             0,
		DebugActivations: false,
		QualityMode:      false,
	}

	inputTokens := e.Tokenizer.Encode(prompt)

	tokens, err := e.Infer(inputTokens, tokensToGenerate, samplerConfig)
	if err != nil {
		return "", err
	}

	return e.Tokenizer.Decode(tokens), nil
}

// Infer generates tokens and returns them all at once
func (e *metalEngine) Infer(inputTokens []int, tokensToGenerate int, samplerConfig SamplerConfig) ([]int, error) {
	return e.InferWithCallback(inputTokens, tokensToGenerate, samplerConfig, nil)
}

// InferWithLogits generates tokens and returns them along with the logits of the last token
func (e *metalEngine) InferWithLogits(inputTokens []int, tokensToGenerate int, samplerConfig SamplerConfig) ([]int, []float32, error) {
	var lastLogits []float32
	tokens, err := e.InferWithCallbackLogits(inputTokens, tokensToGenerate, samplerConfig, nil, func(logits []float32) {
		lastLogits = make([]float32, len(logits))
		copy(lastLogits, logits)
	})
	return tokens, lastLogits, err
}

// InferWithCallbackLogits is like InferWithCallback but also provides logits for each generated token
func (e *metalEngine) InferWithCallbackLogits(inputTokens []int, tokensToGenerate int, samplerConfig SamplerConfig, tokenCallback func(int), logitsCallback func([]float32)) ([]int, error) {
	return e.inferInternal(inputTokens, tokensToGenerate, samplerConfig, tokenCallback, logitsCallback)
}

// InferWithCallback generates tokens with optional streaming callback
// If callback is provided, it's called for each generated token
func (e *metalEngine) InferWithCallback(inputTokens []int, tokensToGenerate int, samplerConfig SamplerConfig, callback func(token int)) ([]int, error) {
	_, span := telemetry.StartSpan(context.Background(), "InferWithCallback") // Use Background since Engine methods might not have ctx
	defer span.End()
	return e.inferInternal(inputTokens, tokensToGenerate, samplerConfig, callback, nil)
}

func (e *metalEngine) ForwardBatch(desc *BatchDescriptor) ([]*device.Tensor, error) {
	batchSize := len(desc.Sequences)
	numTokens := len(desc.Tokens)
	if batchSize == 0 {
		return nil, nil
	}

	// 1. Lookup Embeddings for all packed tokens
	if e.weights.TokenEmb == nil {
		return nil, fmt.Errorf("missing embedding weights")
	}
	inputT := e.weights.TokenEmb.EmbeddingLookupBatch(desc.Tokens, 1.0)
	defer inputT.Free()

	// 1.1 Support for Multimodal Injection Tokens (Phase 5)
	if len(desc.VisionTensors) > 0 {
		// In a production engine, we would perform a zero-copy "Scatter-Gather" 
		// to interleave vision features into the hidden state.
		// For the Phase 5 implementation, we prepended vision features to the prompt.
		for seqIdx, visionT := range desc.VisionTensors {
			if vt, ok := visionT.(*device.Tensor); ok {
				// Stub: Simplified concatenation logic for vision prefill
				// In Gemma 4 / PaliGemma, vision tokens typically take the first N positions.
				_ = vt
			}
		}
	}

	// 2. Scratch Allocation
	// Note: We allocate scratch based on numTokens (ragged)
	qNormDim := 512
	kNormDim := 512
	scratch := e.ctx.NewLayerScratch(numTokens, e.config.Dim, e.config.HiddenDim,
		e.config.Heads, e.config.KVHeads, e.config.HeadDim, e.config.SeqLen, e.config.VocabSize, qNormDim, kNormDim)
	defer scratch.Free()

	// 3. Metadata for Paged Cache (Resource-aware)
	seqIDs := make([]string, batchSize)
	for i, seq := range desc.Sequences {
		seqIDs[i] = fmt.Sprintf("seq-%d", seq.ID)
	}

	// 3.1 Token-specific metadata for RoPE and Attention
	tokenPosData := make([]float32, numTokens)
	tokenSeqData := make([]float32, numTokens)
	for i, seqIdx := range desc.TokenToSeq {
		// Calculate internal offset within the sequence's current chunk
		// This is tricky: we need the relative position from the chunk start.
		// Actually, desc.TokenToSeq and desc.Offsets tell us.
		chunkStartTokenIdx := desc.Offsets[seqIdx]
		offsetInChunk := i - chunkStartTokenIdx
		tokenPosData[i] = float32(desc.ContextLens[seqIdx] + offsetInChunk)
		tokenSeqData[i] = float32(seqIdx)
	}

	tokenPositions := e.ctx.NewTensorFP32(1, numTokens)
	if err := tokenPositions.LoadFrom(tokenPosData); err != nil {
		return nil, err
	}
	defer tokenPositions.Free()

	tokenToSeq := e.ctx.NewTensorFP32(1, numTokens)
	if err := tokenToSeq.LoadFrom(tokenSeqData); err != nil {
		return nil, err
	}
	defer tokenToSeq.Free()

	current := inputT
	// 4. Transformer Layers
	for l := 0; l < e.config.Layers; l++ {
		// Use positions at the START of this chunk (ContextLens) to get Block Tables
		view := e.cache.GetBatch(seqIDs, desc.ContextLens, l)
		
		current.LayerBatch(l,
			e.weights.AttnNorm[l], e.weights.AttnQ[l], e.weights.AttnK[l], e.weights.AttnV[l], e.weights.AttnO[l],
			e.weights.FfnNorm[l], e.weights.FfnGate[l], e.weights.FfnUp[l], e.weights.FfnDown[l],
			view.KPools[l], view.VPools[l],
			scratch,
			tokenPositions, tokenToSeq, view.BlockTables, view.MaxBlocks,
			e.config.Heads, e.config.KVHeads, e.config.HeadDim,
			e.config.RopeTheta, e.config.Eps, e.config.HiddenDim,
			view.BlockSize, numTokens, 1.0,
			func(k, v *device.Tensor) {
				// k/v here are [numTokens, kvDim]
				// We need to update the paged cache for each token's position
				updateItems := make([]struct {
					SeqID string
					Pos   int
					K     *device.Tensor
					V     *device.Tensor
				}, numTokens)
				
				tokenIdx := 0
				for i := range desc.Sequences {
					chunkLen := 1
					if i < len(desc.Offsets)-1 {
						chunkLen = desc.Offsets[i+1] - desc.Offsets[i]
					} else {
						chunkLen = numTokens - desc.Offsets[i]
					}
					
					for j := 0; j < chunkLen; j++ {
						updateItems[tokenIdx].SeqID = seqIDs[i]
						updateItems[tokenIdx].Pos = desc.ContextLens[i] + j
						updateItems[tokenIdx].K = k.Slice(tokenIdx, 1)
						updateItems[tokenIdx].V = v.Slice(tokenIdx, 1)
						tokenIdx++
					}
				}
				e.cache.UpdateBatch(l, updateItems)
			})

		// Apply LoRA to this layer's projections if active
		e.applyLoRA(l, current, scratch, desc)

		view.BlockTables.Free()
		view.BatchPositions.Free()
	}

	// 5. Final Output
	normed := current.RMSNorm(e.weights.OutputNorm, e.config.Eps)
	defer normed.Free()

	// logits [numTokens, VocabSize]
	logitsAll := e.ctx.NewTensorWithType(numTokens, e.config.VocabSize, device.DataTypeF32)
	normed.LinearInto(e.weights.Output, logitsAll, 1.0)
	defer logitsAll.Free()

	// 6. Extract Logits for the LAST token of each sequence in the batch
	results := make([]*device.Tensor, batchSize)
	for i := range desc.Sequences {
		lastTokenIdx := 0
		if i < len(desc.Offsets)-1 {
			lastTokenIdx = desc.Offsets[i+1] - 1
		} else {
			lastTokenIdx = numTokens - 1
		}
		
		// Copy single row to a new tensor
		row := logitsAll.Slice(lastTokenIdx, 1)
		rowCopy := e.ctx.NewTensorWithType(1, e.config.VocabSize, device.DataTypeF32)
		row.CopyF32Into(rowCopy)
		results[i] = rowCopy
	}

	return results, nil
}

func (e *metalEngine) applyLoRA(layerIdx int, input *device.Tensor, scratch *device.LayerScratch, desc *BatchDescriptor) {
	// Group tokens by adapter
	adapterToTokens := make(map[string][]int)
	for i, seqIdx := range desc.TokenToSeq {
		adapterID := desc.AdapterIDs[seqIdx]
		if adapterID != "" {
			adapterToTokens[adapterID] = append(adapterToTokens[adapterID], i)
		}
	}

	if len(adapterToTokens) == 0 {
		return
	}

	// For each adapter, apply it to the corresponding projection results
	for adapterID, tokenIndices := range adapterToTokens {
		// 1. Attention Projections (Input: scratch.Normed)
		e.applyLoRAtoLayer(adapterID, fmt.Sprintf("blk.%d.attn_q", layerIdx), scratch.Normed, scratch.QPart, tokenIndices)
		e.applyLoRAtoLayer(adapterID, fmt.Sprintf("blk.%d.attn_k", layerIdx), scratch.Normed, scratch.KPart, tokenIndices)
		e.applyLoRAtoLayer(adapterID, fmt.Sprintf("blk.%d.attn_v", layerIdx), scratch.Normed, scratch.VPart, tokenIndices)

		// 2. Attention Output (Input: scratch.AttOut)
		e.applyLoRAtoLayer(adapterID, fmt.Sprintf("blk.%d.attn_output", layerIdx), scratch.AttOut, scratch.ResAtt, tokenIndices)

		// 3. FFN Projections (Input: scratch.NormedFFN)
		e.applyLoRAtoLayer(adapterID, fmt.Sprintf("blk.%d.ffn_gate", layerIdx), scratch.NormedFFN, scratch.GatePart, tokenIndices)
		e.applyLoRAtoLayer(adapterID, fmt.Sprintf("blk.%d.ffn_up", layerIdx), scratch.NormedFFN, scratch.UpPart, tokenIndices)

		// 4. FFN Output (Input: scratch.SwiOut - derived from gate/up)
		// swiOut is usually temp, but we might need to apply lora to ResFFN
		e.applyLoRAtoLayer(adapterID, fmt.Sprintf("blk.%d.ffn_down", layerIdx), scratch.SwiOut, scratch.ResFFN, tokenIndices)
	}
}

func (e *metalEngine) applyLoRAtoLayer(adapterID, name string, input, output *device.Tensor, tokenIndices []int) {
	w, ok := e.LoRA.GetWeights(adapterID, name)
	if !ok {
		return
	}

	// LoRA kernel expects F16. If input is F32 (e.g., SwiOut), convert it.
	var finalInput *device.Tensor
	isTempInput := false
	if input.DataType() == device.DataTypeF32 {
		finalInput = e.ctx.NewTensorPooled(input.Rows(), input.Cols())
		input.CopyToF16_Into(finalInput)
		isTempInput = true
	} else {
		finalInput = input
	}
	
	// Slice the rows of input and output that belong to this adapter
	for _, rowIdx := range tokenIndices {
		rowIn := finalInput.Slice(rowIdx, 1)
		rowOut := output.Slice(rowIdx, 1)
		e.ctx.LinearLoRAAdd(rowIn, w.A, w.B, rowOut, w.Alpha/float32(w.Rank))
	}

	if isTempInput {
		finalInput.ReturnToPool()
	}
}

func (e *metalEngine) runBatchLoop() {
	defer close(e.doneChan)
	for {
		select {
		case <-e.stopChan:
			return
		default:
		}
		// Pull active sequences from the manager (desc contains packed tokens)
		desc, _ := e.BatchManager.Step(16, e.cache, e.PromptCache)
		if desc == nil || len(desc.Sequences) == 0 {
			time.Sleep(10 * time.Millisecond)
			continue
		}

		// Forward Pass
		results, err := e.ForwardBatch(desc)
		if err != nil {
			for _, seq := range desc.Sequences {
				select {
				case seq.Err <- err:
				default:
				}
			}
			continue
		}

		// Sampling & Update
		for i, seq := range desc.Sequences {
			if seq.Speculative && e.SpeculativeMgr != nil && !seq.PrefillCompleted {
				// Use Speculative Decoding for this sequence
				err := e.SpeculativeMgr.GenerateSpeculativeMultiPath(context.Background(), seq)
				if err != nil {
					logger.Log.Error("Speculative generation failed", "seq", seq.ID, "error", err)
					// Fallback to regular sampling
				} else {
					// Speculative generation succeeded (possibly accepted 0-K tokens).
					// If we accepted >0 tokens, results[i] is already handled or unused.
					results[i].Free()
					continue
				}
			}

			logits := results[i].ToHostF32()
			results[i].Free()

			if seq.LogitsCallback != nil {
				seq.LogitsCallback(logits)
			}

			sampler := NewSampler(seq.Config)
			token := sampler.Sample(logits, seq.Tokens)

			// Update Sequence State
			// Important: If we were prefilling, we consumed multiple tokens.
			// But Sampling only happens for the LAST token of the chunk.
			chunkLen := 1
			if i < len(desc.Offsets)-1 {
				chunkLen = desc.Offsets[i+1] - desc.Offsets[i]
			} else {
				chunkLen = len(desc.Tokens) - desc.Offsets[i]
			}

			seq.Tokens = append(seq.Tokens, token)
			seq.Pos += chunkLen // Advance by number of tokens processed

			if seq.TokenCallback != nil {
				seq.TokenCallback(token)
			}

			// Check for completion
			if seq.PrefillCompleted {
				// We just finished prefill, cache the resulting blocks
				blocks := e.cache.GetSequenceBlocks(fmt.Sprintf("seq-%d", seq.ID))
				if blocks != nil {
					e.PromptCache.Insert(seq.Tokens[:seq.PromptLen], blocks)
				}
				seq.PrefillCompleted = false
			}

			if token == 2 || len(seq.Tokens) >= seq.MaxTokens {
				select {
				case seq.Result <- seq.Tokens:
				default:
				}
				e.BatchManager.CompleteSequence(seq.ID, e.cache)
			}
		}
	}
}

func (e *metalEngine) inferInternal(inputTokens []int, tokensToGenerate int, samplerConfig SamplerConfig, tokenCallback func(int), logitsCallback func([]float32)) ([]int, error) {
	// Create channels for async completion
	resChan := make(chan []int, 1)
	errChan := make(chan error, 1)

	// Create and submit request
	req := &InferenceRequest{
		ID:             uint64(time.Now().UnixNano()),
		Prompt:         inputTokens,
		MaxTokens:      len(inputTokens) + tokensToGenerate,
		Config:         samplerConfig,
		Result:         resChan,
		Err:            errChan,
		TokenCallback:  tokenCallback,
		LogitsCallback: logitsCallback,
	}

	e.BatchManager.Submit(req)

	// Wait for completion (blocking call for sync API compatibility)
	select {
	case tokens := <-resChan:
		// Return only the newly generated tokens
		if len(tokens) > len(inputTokens) {
			return tokens[len(inputTokens):], nil
		}
		return []int{}, nil
	case err := <-errChan:
		return nil, err
	}
}

func (e *metalEngine) initKVCache() error {
	if err := e.initTurboQuant(); err != nil {
		return err
	}

	// Initialize Paged KV Cache
	pagedCache := &PagedKVCache{}
	if err := pagedCache.Init(e.ctx, e.config); err != nil {
		return err
	}
	e.cache = pagedCache

	// Initialize Batch Manager
	if e.BatchManager == nil {
		e.BatchManager = NewContinuousBatchManager()
	}

	// Loop management moved to NewMetalEngine and SwapModel

	return nil
}

func (e *metalEngine) initTurboQuant() error {
	if e.config.KVCacheType != config.KVCacheTQ1_0 && e.config.KVCacheType != config.KVCacheTQ2_0 {
		return nil
	}

	headDim := e.config.HeadDim
	qjlRows := 64 // Fixed for now

	var rotData []float32
	var qjlData []float32

	// 1. Try Loading from Model (GGUF)
	if e.model != nil {
		rot, qjl, err := e.model.GetTurboQuantMatrices()
		if err == nil {
			rotData = rot
			qjlData = qjl
			logger.Log.Info("TurboQuant matrices loaded from model", "head_dim", headDim)
		} else {
			logger.Log.Debug("TurboQuant matrices not found in model, using fallbacks", "error", err)
		}
	}

	// 2. Deterministic Fallbacks
	if rotData == nil {
		rotData = device.GetPrecomputedRotation(headDim)
	}
	if qjlData == nil {
		qjlData = device.GetPrecomputedQJLSigns(qjlRows * headDim)
	}

	logger.Log.Info("TurboQuant matrices initialized", "head_dim", headDim, "qjl_rows", qjlRows, "source", "loaded/precomputed")

	// 3. Tensor Allocation
	rot := e.ctx.NewTensorFP32(headDim, headDim)
	if rot == nil {
		return fmt.Errorf("failed to allocate TQRotation tensor")
	}
	rot.LoadFromF32(rotData)
	e.ctx.TQRotation = rot

	qjl := e.ctx.NewTensorFP32(qjlRows, headDim)
	if qjl == nil {
		return fmt.Errorf("failed to allocate TQQJL tensor")
	}
	qjl.LoadFromF32(qjlData)
	e.ctx.TQQJL = qjl

	logger.Log.Info("TurboQuant matrices initialized", "head_dim", headDim, "qjl_rows", qjlRows, "source", "loaded/generated")
	return nil
}

func (e *metalEngine) Close() {
	if e.stopChan != nil {
		close(e.stopChan)
		<-e.doneChan // Wait for loop to exit
	}

	if e.weights != nil {
		e.weights.Free()
	}
	if e.cache != nil {
		e.cache.Free()
	}
	for _, s := range e.SSMCache {
		if s != nil {
			s.Free()
		}
	}
	me := e
	if me.ctx != nil {
		me.ctx.Free()
	}
}

// GetEmbedding returns the embedding vector for a single token as a float32 slice
func (e *metalEngine) GetEmbedding(token int) ([]float32, error) {
	if e.weights.TokenEmb == nil {
		return nil, errors.New("token embedding weights not loaded")
	}
	if token < 0 || token >= e.weights.TokenEmb.Rows() {
		return nil, fmt.Errorf("token %d out of vocab range [0, %d)", token, e.weights.TokenEmb.Rows())
	}

	emb := e.weights.TokenEmb.EmbeddingLookup(token, e.GlobalScale)
	defer emb.Free()

	return emb.ToHost(), nil
}



// GetEmbeddings returns embedding vectors for multiple tokens
func (e *metalEngine) GetEmbeddings(tokens []int) ([][]float32, error) {
	embeddings := make([][]float32, len(tokens))
	for i, token := range tokens {
		emb, err := e.GetEmbedding(token)
		if err != nil {
			return nil, fmt.Errorf("error embedding token %d: %w", i, err)
		}
		embeddings[i] = emb
	}
	return embeddings, nil
}

// TextToEmbedding tokenizes the input text and returns embeddings for all tokens
func (e *metalEngine) TextToEmbedding(text string) ([][]float32, error) {
	if e.Tokenizer == nil {
		return nil, errors.New("tokenizer not available")
	}

	tokens := e.Tokenizer.Encode(text)
	if len(tokens) == 0 {
		return nil, errors.New("text tokenized to empty sequence")
	}

	return e.GetEmbeddings(tokens)
}

// EmbeddingDim returns the dimension of the embedding vectors
func (e *metalEngine) EmbeddingDim() int {
	if e.weights.TokenEmb == nil {
		return 0
	}
	return e.weights.TokenEmb.Cols()
}

// LoadWeightFromGGUF decodes weights to F32 for CPU reference
func LoadWeightFromGGUF(e *metalEngine, name string) []float32 {
	var t *gguf.TensorInfo
	for _, tensor := range e.model.Tensors {
		if tensor.Name == name {
			t = tensor
			break
		}
	}
	if t == nil {
		panic(fmt.Sprintf("Tensor %s not found in GGUF", name))
	}

	numElements := int(t.Dimensions[0])
	for i := 1; i < len(t.Dimensions); i++ {
		numElements *= int(t.Dimensions[i])
	}

	switch t.Type {
	case gguf.GGMLTypeQ4_K:
		return gguf.DequantizeQ4K(t.Data, numElements)
	case gguf.GGMLTypeQ6_K:
		return gguf.DequantizeQ6K(t.Data, numElements)
	case gguf.GGMLTypeF32:
		out := make([]float32, numElements)
		for i := 0; i < numElements; i++ {
			bits := binary.LittleEndian.Uint32(t.Data[i*4 : (i+1)*4])
			out[i] = math.Float32frombits(bits)
		}
		return out
	case gguf.GGMLTypeF16:
		out := make([]float32, numElements)
		for i := 0; i < numElements; i++ {
			bits := binary.LittleEndian.Uint16(t.Data[i*2 : (i+1)*2])
			out[i] = device.Float16ToFloat32(bits)
		}
		return out
	}
	panic(fmt.Sprintf("Unsupported type %d for %s", t.Type, name))
}

// SwapModel safely replaces the currently loaded model with a new one
// It blocks new inferences, waits for ongoing ones (via RWMutex), frees the old weights, and loads the new ones.
func (e *metalEngine) SwapModel(newModelPath string, newConfig config.Config) error {
	startTime := time.Now()
	success := false

	defer func() {
		metrics.RecordModelHotSwap(time.Since(startTime), success)
	}()

	// 1. Signal background loop to stop
	if e.stopChan != nil {
		close(e.stopChan)
		<-e.doneChan // Wait for loop to exit completely
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	// 2. Abort all active sequences to prevent caller deadlocks
	if e.BatchManager != nil {
		e.BatchManager.AbortAll(ErrModelSwapped)
	}

	// 3. Free existing cache and weights
	if e.cache != nil {
		e.cache.Free()
		e.cache = nil
	}
	if e.weights != nil {
		e.weights.Free()
		e.weights = nil
	}

	// 4. Update config
	e.config = newConfig

	// 5. Pre-allocate weights container for metadata detection
	e.weights = &LlamaWeights{}

	// 6. Load the new model
	if err := e.loadModel(newModelPath); err != nil {
		return err
	}

	// 7. Restart background loop
	e.stopChan = make(chan struct{})
	e.doneChan = make(chan struct{})
	go e.runBatchLoop()

	success = true
	return nil
}

func (e *metalEngine) ForwardDraft(tokens []int) ([][]float32, error) {
	// Treat as a single-sequence prefill batch for verification
	desc := &BatchDescriptor{
		Sequences:   []*Sequence{{ID: 0, Tokens: tokens}},
		Tokens:      tokens,
		Offsets:     []int{0},
		ContextLens: []int{0},
		TokenToSeq:  make([]int, len(tokens)),
		AdapterIDs:  []string{""},
		IsDecode:    make([]bool, len(tokens)),
	}
	
	results, err := e.ForwardBatch(desc)
	if err != nil {
		return nil, err
	}
	
	// Convert device tensors to host logits
	hostResults := make([][]float32, len(results))
	for i, res := range results {
		hostResults[i] = res.ToHostF32()
		res.Free()
	}
	
	return hostResults, nil
}

func (e *metalEngine) RollbackKV(seqID string, newPos int) error {
	return e.cache.RollbackKV(seqID, newPos)
}

func (e *metalEngine) LoadAdapter(path, id string) error {
	return e.LoRA.LoadAdapter(e.ctx, path, id)
}
