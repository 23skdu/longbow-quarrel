//go:build darwin && metal

package engine

import (
	"encoding/binary"
	"errors"
	"fmt"
	"log"
	"math"
	"runtime"
	"sort"

	"time"

	"strings"

	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
	"github.com/23skdu/longbow-quarrel/internal/metrics"
)

// Helper to get KV value with architecture-aware fallback
func getKV(f *gguf.GGUFFile, llamaKey, qwenKey string) (interface{}, bool) {
	// Try llama key first
	if val, ok := f.KV[llamaKey]; ok {
		return val, true
	}
	// Fallback to qwen key if provided
	if qwenKey != "" {
		if val, ok := f.KV[qwenKey]; ok {
			return val, true
		}
	}
	return nil, false
}

func NewEngine(modelPath string, debugDequant bool) (*Engine, error) {
	ctx := device.NewContext()

	e := &Engine{
		Ctx:        ctx,
		Weights:    &LlamaWeights{},
		ActLogger:  NewActivationLogger(),
		LastLogits: nil,
	}
	e.Config.DebugDequant = debugDequant

	if err := e.loadModel(modelPath); err != nil {
		ctx.Free()
		return nil, err
	}

	e.TraceTracker = NewActivationTraceTracker(e.Config.Layers)

	return e, nil
}

// Internal load method
// Internal load method
func (e *Engine) loadModel(path string) error {
	f, err := gguf.LoadFile(path)
	if err != nil {
		return err
	}
	e.Model = f

	// Read Metadata
	// Block Count
	if val, ok := getKV(f, "llama.block_count", "qwen3moe.block_count"); ok {
		e.Config.Layers = int(toFloat64(val)) // GGUF numbers can be various types
	} else {
		e.Config.Layers = 1 // Default for test
	}

	// Embedding Dim
	if val, ok := getKV(f, "llama.embedding_length", "qwen3moe.embedding_length"); ok {
		e.Config.Dim = int(toFloat64(val))
	}

	// Heads
	if val, ok := getKV(f, "llama.attention.head_count", "qwen3moe.attention.head_count"); ok {
		e.Config.Heads = int(toFloat64(val))
	} else {
		return errors.New("missing attention.head_count")
	}

	// Hidden Dim (FFN context)
	if val, ok := getKV(f, "llama.feed_forward_length", "qwen3moe.feed_forward_length"); ok {
		e.Config.HiddenDim = int(toFloat64(val))
	} else {
		e.Config.HiddenDim = 4 * e.Config.Dim // fallback
	}

	// KV Heads (GQA)
	if val, ok := getKV(f, "llama.attention.head_count_kv", "qwen3moe.attention.head_count_kv"); ok {
		e.Config.KVHeads = int(toFloat64(val))
	} else {
		// Default to Heads (MHA)
		e.Config.KVHeads = e.Config.Heads
	}

	// Head Dim
	e.Config.HeadDim = e.Config.Dim / e.Config.Heads

	// Seq Len (Context)
	if val, ok := getKV(f, "llama.context_length", "qwen3moe.context_length"); ok {
		e.Config.SeqLen = int(toFloat64(val))
	} else {
		e.Config.SeqLen = 2048 // default
	}

	// RoPE Freq
	if val, ok := getKV(f, "llama.rope.freq_base", "qwen3moe.rope.freq_base"); ok {
		e.Config.RopeTheta = float32(toFloat64(val))
		log.Printf("[MODEL] RoPE Theta: %.0f", e.Config.RopeTheta)
	} else {
		// Default to 10k for Llama 2, but Mistral v0.3 uses 1M
		// We'll set it properly based on architecture below
		e.Config.RopeTheta = 10000.0
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
	e.Config.Eps = eps
	fmt.Printf("[DEBUG] NormEps: %e\n", eps)

	// Sliding Window Size (for Mistral)
	// Mistral uses 4096-token sliding window attention
	// If not specified in GGUF, default to 4096 for Mistral, 0 (disabled) for others
	if val, ok := getKV(f, "llama.attention.sliding_window", ""); ok {
		e.Config.WindowSize = int(toFloat64(val))
		log.Printf("[MODEL] Sliding Window Size: %d", e.Config.WindowSize)
	} else {
		// Check if this is Mistral (heuristic: has specific architecture name)
		if arch, ok := f.KV["general.architecture"].(string); ok && arch == "llama" {
			// For Mistral models, default to 4096
			// For other models (Llama, etc.), use 0 (full attention)
			e.Config.WindowSize = 4096
			log.Printf("[MODEL] Sliding Window Size (Default for Mistral): %d", e.Config.WindowSize)
		} else {
			e.Config.WindowSize = 0 // Full attention
		}
	}

	// Log Model Architecture
	if arch, ok := f.KV["general.architecture"].(string); ok {
		log.Printf("[MODEL] Architecture: %s", arch)

		// Heuristic: If it's llama or mistral and WindowSize not set,
		// check if we should assume Mistral SWA.
		if strings.Contains(strings.ToLower(arch), "llama") || strings.Contains(strings.ToLower(arch), "mistral") {
			if e.Config.WindowSize == 0 {
				// Most Mistral models in GGUF don't have the sliding_window key set correctly,
				// but they still require it for correctness if they are v0.3+.
				// However, Llama 2 also uses "llama" arch.
				// Heuristic: If RopeTheta is 1M, it's likely Mistral v0.3.
				if e.Config.RopeTheta >= 1000000.0 {
					e.Config.WindowSize = 4096
					log.Printf("[MODEL] Heuristic: Mistral v0.3 detected, using 4096 SWA")
				}
			}
		}
	}
	if val, ok := getKV(f, "llama.vocab_size", ""); ok {
		e.Config.VocabSize = int(toFloat64(val))
		log.Printf("[MODEL] Vocab Size: %d", e.Config.VocabSize)
	} else {
		// Fallback for Smollm2 / Llama3 if missing
		e.Config.VocabSize = 49152
		log.Printf("[MODEL] Vocab Size (Default): %d", e.Config.VocabSize)
	}

	// Set Precision Mode based on model dimensions (explicit configuration instead of heuristic)
	// This can be overridden by metadata if available
	if arch, ok := f.KV["general.architecture"].(string); ok {
		log.Printf("[MODEL] Architecture: %s", arch)
	}

	if val, ok := f.KV["llama.attention.precision"]; ok {
		if prec, ok := val.(string); ok {
			switch prec {
			case "f16":
				e.Config.PrecisionMode = PrecisionFP16
				log.Printf("[MODEL] Precision Mode: FP16 (from metadata)")
			case "f32":
				e.Config.PrecisionMode = PrecisionF32FFN
				log.Printf("[MODEL] Precision Mode: F32 FFN (from metadata)")
			case "mixed":
				e.Config.PrecisionMode = PrecisionMixed
				log.Printf("[MODEL] Precision Mode: Mixed (from metadata)")
			default:
				log.Printf("[MODEL] Unknown precision mode: %s, using auto", prec)
			}
		} else {
			e.Config.PrecisionMode = PrecisionAuto
		}
	} else {
		// Auto-detect based on dimension
		if e.Config.Dim < 1024 {
			e.Config.PrecisionMode = PrecisionF32FFN
			log.Printf("[MODEL] Precision Mode: Auto (F32 FFN for small model, dim=%d)", e.Config.Dim)
		} else if e.Config.Dim >= 4096 {
			e.Config.PrecisionMode = PrecisionMixed
			log.Printf("[MODEL] Precision Mode: Auto (Mixed precision for large model, dim=%d)", e.Config.Dim)
		} else {
			e.Config.PrecisionMode = PrecisionFP16
			log.Printf("[MODEL] Precision Mode: Auto (FP16 for medium model, dim=%d)", e.Config.Dim)
		}
	}

	if val, ok := f.KV["tokenizer.ggml.bos_token_id"]; ok {
		log.Printf("[MODEL] BOS Token ID: %d", int(toFloat64(val)))
	}
	if val, ok := f.KV["tokenizer.ggml.eos_token_id"]; ok {
		log.Printf("[MODEL] EOS Token ID: %d", int(toFloat64(val)))
	}

	// Initialize KV Cache (now that we have dimensions)
	if err := e.initKVCache(); err != nil {
		return err
	}

	log.Printf("[MODEL] Config: Layers=%d, Dim=%d, HiddenDim=%d, Heads=%d, KVHeads=%d, HeadDim=%d, Eps=%e, RopeFreq=%f\n",
		e.Config.Layers, e.Config.Dim, e.Config.HiddenDim, e.Config.Heads, e.Config.KVHeads, e.Config.HeadDim, e.Config.Eps, e.Config.RopeTheta)

	// Initialize Weights Slices
	layers := e.Config.Layers
	e.Weights.AttnQ = make([]*device.Tensor, layers)
	e.Weights.AttnK = make([]*device.Tensor, layers)
	e.Weights.AttnV = make([]*device.Tensor, layers)
	e.Weights.AttnO = make([]*device.Tensor, layers)
	e.Weights.AttnNorm = make([]*device.Tensor, layers)
	e.Weights.FfnGate = make([]*device.Tensor, layers)
	e.Weights.FfnDown = make([]*device.Tensor, layers)
	e.Weights.FfnUp = make([]*device.Tensor, layers)
	e.Weights.FfnNorm = make([]*device.Tensor, layers)

	// Map tensors
	for _, t := range f.Tensors {
		cols := int(t.Dimensions[0])
		rows := 1
		if len(t.Dimensions) > 1 {
			rows = int(t.Dimensions[1])
		}
		for i := 2; i < len(t.Dimensions); i++ {
			rows *= int(t.Dimensions[i])
		}

		fmt.Printf("[TENSOR] %-30s | Type: %-10v | Shape: [%d, %d]\n", t.Name, t.Type, rows, cols)

		var mt *device.Tensor
		numElements := rows * cols

		// Validate tensor dimensions before processing
		if err := ValidateTensorDimensions(t.Name, rows, cols, t.Type); err != nil {
			log.Printf("[VALIDATION] %s: %v", t.Name, err)
		}

		// Map tensor types
		// Validate F16 Conversion
		if device.Float32ToFloat16(0.0) != 0 {
			panic(fmt.Sprintf("Float32ToFloat16(0.0) = %x (Expected 0)", device.Float32ToFloat16(0.0)))
		}

		if false {
		} else {
			if t.Type == gguf.GGMLTypeF32 {
				// Standard F32 Tensor
				mt = e.Ctx.NewTensorFP32(rows, cols)
				// ... F32 load ...
				dataBytes := numElements * 4
				if uint64(len(t.Data)) < uint64(dataBytes) {
					return fmt.Errorf("tensor %s data truncated", t.Name)
				}
				rawBytes := t.Data[:dataBytes]

				f32Data := make([]float32, numElements)
				for i := 0; i < numElements; i++ {
					bits := binary.LittleEndian.Uint32(rawBytes[i*4 : (i+1)*4])
					f32Data[i] = math.Float32frombits(bits)
				}

				mt.LoadFrom(f32Data)

				// Heuristic for GlobalScale disabled
				e.GlobalScale = 1.0
				if t.Name == "blk.0.attn_norm.weight" || t.Name == "output_norm.weight" {
					var sum, sumSq float64
					for _, v := range f32Data {
						val := float64(v)
						sum += math.Abs(val)
						sumSq += val * val
					}
					mean := float32(sum / float64(numElements))
					rms := float32(math.Sqrt(sumSq / float64(numElements)))
					fmt.Printf("WEIGHT DEBUG: %s Mean=%e RMS=%e First4=%v HEX=%x\n", t.Name, mean, rms, f32Data[:4], rawBytes[:16])
				}
			} else if t.Type == gguf.GGMLTypeF16 {
				if isNormWeight(t.Name) {
					// Promote Norm weights to FP32 for precision and kernel compatibility
					f32Data := gguf.DequantizeF16(t.Data, numElements)
					mt = e.Ctx.NewTensorFP32(rows, cols)
					mt.LoadFrom(f32Data)
					fmt.Printf("[PROMOTED] %s to FP32\n", t.Name)
				} else {
					mt = e.Ctx.NewTensorWithType(rows, cols, device.DataTypeF16)
					// ... F16 load ...
					dataBytes := numElements * 2
					if uint64(len(t.Data)) < uint64(dataBytes) {
						return fmt.Errorf("tensor %s data truncated", t.Name)
					}
					mt.LoadFromRaw(t.Data[:dataBytes])
				}
			} else if t.Type == gguf.GGMLTypeQ4_K {
				// Type 12 (Q4_K).

				if e.Config.Dim < 1024 {
					f32Data := gguf.DequantizeQ4K(t.Data, numElements)
					mt = e.Ctx.NewTensorWithType(rows, cols, device.DataTypeF16)
					mt.LoadFrom(f32Data)
				} else {
					// Large Models: Use Q4K Tensor and Kernels
					var err error
					mt, err = e.Ctx.NewQ4KTensor(rows, cols)
					if err != nil {
						return fmt.Errorf("failed to create Q4K tensor %s: %w", t.Name, err)
					}
					dataBytes := (numElements / 256) * 144
					// Check size matches (handle truncated data)
					if uint64(len(t.Data)) < uint64(dataBytes) {
						return fmt.Errorf("tensor %s data truncated (Need %d, Has %d)", t.Name, dataBytes, len(t.Data))
					}

					mt.LoadFromRaw(t.Data[:dataBytes])

				}
			} else if t.Type == gguf.GGMLTypeQ4_0 {
				// Type 2 (Q4_0).
				// Always use native Q4_0 kernel
				// rows * cols elements.
				// Block size 32. 18 bytes per block.
				// Check alignment
				if numElements%32 != 0 {
					return fmt.Errorf("Q4_0 tensor %s size %d not divisible by 32", t.Name, numElements)
				}
				mt = e.Ctx.NewTensorWithType(rows, cols, device.DataTypeQ4_0)
				dataBytes := (numElements / 32) * 18
				if uint64(len(t.Data)) < uint64(dataBytes) {
					return fmt.Errorf("tensor %s data truncated (Need %d, Has %d)", t.Name, dataBytes, len(t.Data))
				}
				mt.LoadFromRaw(t.Data[:dataBytes])

			} else if t.Type == gguf.GGMLTypeQ6_K {
				// Type 14 (Q6_K).
				if t.Name == "output.weight" || e.Config.Dim >= 1024 {
					// Use Native Q6K
					mt = e.Ctx.NewTensorWithType(rows, cols, device.DataTypeQ6K)
					dataBytes := (numElements / 256) * 210
					if uint64(len(t.Data)) < uint64(dataBytes) {
						return fmt.Errorf("tensor %s data truncated", t.Name)
					}
					mt.LoadFromRaw(t.Data[:dataBytes])
				} else {
					f32Data := gguf.DequantizeQ6K(t.Data, numElements)
					mt = e.Ctx.NewTensorWithType(rows, cols, device.DataTypeF16)
					mt.LoadFrom(f32Data)
				}

			} else if t.Type == gguf.GGMLTypeQ4_K_S { // 99 Unused
				mt = e.Ctx.NewTensor(rows, cols) // fallback
			} else {
				fmt.Printf("Skipping unsupported tensor type: %d (%s)\n", t.Type, t.Name)
				continue
			}
		}

		if mt != nil {
			if t.Type == gguf.GGMLTypeF16 || t.Type == gguf.GGMLTypeF32 || t.Type == gguf.GGMLTypeQ6_K || t.Name == "token_embd.weight" {

			} else if t.Type == gguf.GGMLTypeQ4_K {
				mt.ScanQ4KScales(t.Name)
			}
		}

		// Mapping Logic
		name := t.Name

		// 1. Global weights
		switch name {
		case "token_embd.weight":
			e.Weights.TokenEmb = mt
			continue
		case "output_norm.weight":
			e.Weights.OutputNorm = mt
			continue
		case "output.weight":
			e.Weights.Output = mt
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
				e.Weights.AttnQ[layerIdx] = mt
			case "attn_k.weight":
				e.Weights.AttnK[layerIdx] = mt
			case "attn_v.weight":
				e.Weights.AttnV[layerIdx] = mt
			case "attn_output.weight":
				e.Weights.AttnO[layerIdx] = mt
			case "attn_norm.weight":
				e.Weights.AttnNorm[layerIdx] = mt
			case "ffn_gate.weight":
				e.Weights.FfnGate[layerIdx] = mt
				if layerIdx == 0 {
					e.Config.HiddenDim = rows
				}
				continue
			case "ffn_down.weight":
				e.Weights.FfnDown[layerIdx] = mt
				// Debug all FFN Down weights
				if layerIdx <= 5 {
					if mt.DataType() == device.DataTypeQ4K {
						log.Printf("blk.%d.ffn_down: Q4K quantized", layerIdx)
						mt.ScanQ4KScales(fmt.Sprintf("blk.%d.ffn_down", layerIdx))
					} else {
						log.Printf("blk.%d.ffn_down: F16 (rows=%d, cols=%d)", layerIdx, mt.Rows(), mt.Cols())
					}
				}
			case "ffn_up.weight":
				e.Weights.FfnUp[layerIdx] = mt
			case "ffn_norm.weight":
				e.Weights.FfnNorm[layerIdx] = mt
			}
		}
	}

	// Fallback: many models share token_embd.weight with output.weight
	// For Llama/Mistral architectures, always tie output to token_embd for correctness
	if e.Weights.TokenEmb != nil {
		e.Weights.Output = e.Weights.TokenEmb
		log.Printf("[MODEL] Tied output.weight to token_embd.weight for Llama architecture")
	}

	return nil
}

// Helper for loose typing in GGUF KV
func toFloat64(v interface{}) float64 {
	switch i := v.(type) {
	case float64:
		return i
	case float32:
		return float64(i)
	case uint64:
		return float64(i)
	case uint32:
		return float64(i)
	case int32:
		return float64(i)
	case int64:
		return float64(i)
	case int:
		return float64(i)
	default:
		return 0
	}
}

func (e *Engine) Infer(inputTokens []int, tokensToGenerate int, samplerConfig SamplerConfig) ([]int, error) {
	// Validation
	if len(inputTokens) == 0 {
		return nil, errors.New("empty input tokens")
	}
	fmt.Printf("DEBUG: Input Tokens: %v\n", inputTokens)

	// Enable activation logging if requested
	if samplerConfig.DebugActivations {
		promptText := fmt.Sprintf("tokens:%v", inputTokens) // Simple representation for now
		e.ActLogger.Enable(promptText, inputTokens)
	}

	fmt.Printf("DEBUG CONFIG: RopeTheta=%.2f HeadDim=%d Heads=%d KVHeads=%d Eps=%.6f\n",
		e.Config.RopeTheta, e.Config.HeadDim, e.Config.Heads, e.Config.KVHeads, e.Config.Eps)

	// Phase 1: Prefill (Process all tokens except last one, generating KV cache)
	if e.Model == nil {
		return nil, errors.New("no model loaded")
	}

	// Validate critical weights are loaded
	if e.Weights.TokenEmb == nil {
		return nil, errors.New("token embedding weights not loaded")
	}
	if e.Weights.OutputNorm == nil {
		return nil, errors.New("output norm weights not loaded")
	}
	if e.Weights.Output == nil {
		return nil, errors.New("output weights not loaded")
	}

	// Validate input tokens are within vocab range
	for i, token := range inputTokens {
		if token < 0 || token >= e.Weights.TokenEmb.Rows() {
			return nil, fmt.Errorf("input token %d at position %d is out of vocab range [0, %d)", token, i, e.Weights.TokenEmb.Rows())
		}
	}

	// tStart := time.Now()
	result := make([]int, 0, tokensToGenerate)

	// Reset KV cache position
	e.CachePos = 0

	// Lock OS thread for AutoreleasePool consistency
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	sampler := NewSampler(samplerConfig)

	// Main Generation Loop

	// Create scratch buffers for the layer fusion
	// New Scratch Buffer for Zero-Alloc	// Initialize scratch buffers (Heap backed, includes Logits)
	scratch := e.Ctx.NewLayerScratch(1, e.Config.Dim, e.Config.HiddenDim,
		e.Config.Heads, e.Config.KVHeads, e.Config.HeadDim, e.Config.SeqLen, e.Config.VocabSize)
	defer scratch.Free()

	// Logits are in scratch.Logits
	logits := scratch.Logits

	// Phase 1: Prefill all input tokens
	log.Printf("DEBUG: START INFER, CachePos=%d", e.CachePos)
	for i := 0; i < len(inputTokens); i++ {
		// Autorelease Pool for this iteration
		pool := e.Ctx.AutoreleasePoolPush()

		tToken := time.Now()
		lastToken := inputTokens[i]

		current := e.Weights.TokenEmb.EmbeddingLookup(lastToken, e.GlobalScale)
		if samplerConfig.DebugActivations || (i < 10) {
			current.ScanMax(fmt.Sprintf("[Pos %d] Token %d Embedding", e.CachePos, lastToken))
		}

		// DEBUG: Print first 4 elements of embedding
		e.Ctx.Synchronize()
		// embData := current.ToHost()
		// fmt.Printf("DEBUG_EMB: Token %d (pos %d) first 4: %v\n", lastToken, i, embData[:4])

		currentF32 := current.ToF32()
		current.ReturnToPool() // Release F16

		// Log embedding if first token
		if i == 0 && e.ActLogger.IsEnabled() {
			embData := currentF32.ToHost()
			e.ActLogger.LogEmbedding(embData)
		}

		// Debug Embedding
		if i < 2 {
			embData := currentF32.ToHost()
			fmt.Printf("DEBUG: Token %d Embedding (First 10): %v\n", lastToken, embData[:10])
		}

		// Track embedding stats for first token
		if i == 0 {
			stats := currentF32.GetStats(16)
			e.TraceTracker.RecordLayer("embedding", -1, stats)
		}

		// Layers (Attention + FFN)
		for l := 0; l < e.Config.Layers; l++ {
			log.Printf("DEBUG_PRECISION: Layer %d, Dim=%d, PrecisionMode=%d (0=Auto, 1=FP16, 2=F32FFN, 3=Mixed)", l, e.Config.Dim, e.Config.PrecisionMode)
			attnNorm := e.Weights.AttnNorm[l]
			q := e.Weights.AttnQ[l]
			k := e.Weights.AttnK[l]
			v := e.Weights.AttnV[l]
			o := e.Weights.AttnO[l]
			ffnNorm := e.Weights.FfnNorm[l]
			ffnGate := e.Weights.FfnGate[l]
			ffnUp := e.Weights.FfnUp[l]
			ffnDown := e.Weights.FfnDown[l]
			kCache := e.KVCacheK[l]
			vCache := e.KVCacheV[l]

			currentF32.Layer(l, attnNorm, q, k, v, o, ffnNorm, ffnGate, ffnUp, ffnDown, kCache, vCache,
				scratch, // Pass scratch
				e.TraceTracker,
				e.CachePos, e.Config.Heads, e.Config.KVHeads, e.Config.HeadDim, e.Config.RopeTheta, e.Config.Eps, e.Config.HiddenDim, e.Config.SeqLen, e.Config.WindowSize, e.GlobalScale, samplerConfig.DebugActivations, int(e.Config.PrecisionMode))

			// Log layer activation details if enabled
			if e.ActLogger.IsEnabled() {
				// Get samples and max values for logging
				qSample := GetSampleFromTensor(scratch.QPart.ToHost(), 10)
				kSample := GetSampleFromTensor(scratch.KPart.ToHost(), 10)
				vSample := GetSampleFromTensor(scratch.VPart.ToHost(), 10)

				qMax := GetMaxFromTensor(scratch.QPart.ToHost())
				kMax := GetMaxFromTensor(scratch.KPart.ToHost())
				vMax := GetMaxFromTensor(scratch.VPart.ToHost())
				attnMax := GetMaxFromTensor(scratch.AttOut.ToHost())
				ffnMax := GetMaxFromTensor(currentF32.ToHost())

				e.ActLogger.LogLayer(l, qMax, kMax, vMax, attnMax, ffnMax,
					qSample, kSample, vSample,
					scratch.QPart.ToHost(), scratch.KPart.ToHost(), scratch.VPart.ToHost(),
					scratch.AttOut.ToHost(), currentF32.ToHost())
			}

			// Log layer output if enabled OR if first token (for recovery analysis)
			if samplerConfig.DebugActivations || (i == 0) {
				currentF32.ScanMax(fmt.Sprintf("[Pos %d] Layer %d Output", e.CachePos, l))

				// Track layer stats for first token
				if i == 0 {
					stats := currentF32.GetStats(16)
					e.TraceTracker.RecordLayer(fmt.Sprintf("layer_%d", l), l, stats)
				}
			}
		}

		// If this is the LAST prompt token, sample the first next token
		if i == len(inputTokens)-1 {
			// Final Norm (F32 -> F16)
			// Debug Output Norm Weights
			if i == 0 {
				e.Weights.OutputNorm.ScanMax("Output Norm Weights")
				e.Weights.Output.ScanMax("Output Weights")
			}

			// Debug: check currentF32 before final norm
			currentF32.ScanMax(fmt.Sprintf("Layer %d Final Input (before norm)", i))

			// Check for and handle Inf/NaN values before final norm
			if infInfo := currentF32.ScanForNaN(fmt.Sprintf("Layer %d Pre-Final Norm", i), 5); infInfo.HasInf || infInfo.HasNaN() {
				log.Printf("[WARN] Found %d NaN and %d Inf values in pre-final norm, clamping extreme values", infInfo.Count, infInfo.InfCount)
				// Clamp extreme values to prevent RMSNorm issues
				hostData := currentF32.ToHost()
				for j := range hostData {
					if math.IsInf(float64(hostData[j]), 0) {
						// Replace Inf with a large but finite value
						if hostData[j] > 0 {
							hostData[j] = 1e6
						} else {
							hostData[j] = -1e6
						}
					} else if math.IsNaN(float64(hostData[j])) {
						hostData[j] = 0.0 // Replace NaN with zero
					}
					// Also clamp extremely large values
					if hostData[j] > 1e6 {
						hostData[j] = 1e6
					} else if hostData[j] < -1e6 {
						hostData[j] = -1e6
					}
				}
				// Load cleaned data back
				cleanTensor := e.Ctx.NewTensorFP32(currentF32.Rows(), currentF32.Cols())
				cleanTensor.LoadFromF32(hostData)
				// Use cleaned tensor for RMSNorm
				cleanTensor.RMSNormFP32_ToF16_Into(e.Weights.OutputNorm, e.Config.Eps, scratch.Normed)
				e.Ctx.Synchronize()
				cleanTensor.ReturnToPool()
			} else {
				currentF32.RMSNormFP32_ToF16_Into(e.Weights.OutputNorm, e.Config.Eps, scratch.Normed)
				e.Ctx.Synchronize()
			}

			// Debug: check normed output
			scratch.Normed.ScanMax(fmt.Sprintf("Layer %d Final Norm Output", i))

			// Check for NaN in normalized output
			if nanInfo := scratch.Normed.ScanForNaN("Final Norm", 5); nanInfo.HasNaN() {
				log.Printf("[ERROR] NaN detected in Final Norm: count=%d, positions=%v", nanInfo.Count, nanInfo.Positions)
			}

			// Output Head (F16 -> F32 Logits)
			// scratch.Normed contains result. Use it.
			scratch.Normed.LinearToFP32_Into(e.Weights.Output, logits)
			e.Ctx.Synchronize()

			// Debug: check final logits
			logits.ScanMax(fmt.Sprintf("Layer %d Final Logits", i))

			logitsData := logits.ToHost()

			// Check for NaN in logits BEFORE sampling
			nanInfo := device.DetectNaN(logitsData, 10)
			if nanInfo.HasNaN() {
				log.Printf("[ERROR] NaN detected in logits: count=%d, positions=%v, hasInf=%v", nanInfo.Count, nanInfo.Positions, nanInfo.HasInf)
				// Try to recover by using argmax of non-NaN values
				validCount := 0
				for _, v := range logitsData {
					if !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) {
						validCount++
					}
				}
				if validCount == 0 {
					log.Printf("[ERROR] All logits are NaN or Inf, cannot recover")
					// Fallback: use last valid logits if available
					if e.LastLogits != nil && device.IsValid(e.LastLogits) {
						log.Printf("[WARN] Using last valid logits for recovery")
						logitsData = e.LastLogits
					} else {
						return nil, fmt.Errorf("all logits are NaN/Inf and no recovery logits available")
					}
				} else {
					log.Printf("[WARN] Found %d valid logits out of %d, filtering NaNs", validCount, len(logitsData))
				}
			}

			// Update last valid logits
			if device.IsValid(logitsData) {
				if e.LastLogits == nil || len(e.LastLogits) < len(logitsData) {
					e.LastLogits = make([]float32, len(logitsData))
				}
				copy(e.LastLogits, logitsData)
			}

			params := make([]struct {
				idx int
				v   float32
			}, len(logitsData))
			for idx, v := range logitsData {
				params[idx].idx = idx
				params[idx].v = v
			}
			sort.Slice(params, func(i, j int) bool { return params[i].v > params[j].v })
			fmt.Printf("Top 10 Logits: ")
			for i := 0; i < 10 && i < len(params); i++ {
				fmt.Printf("[%d:%.2f] ", params[i].idx, params[i].v)
			}
			fmt.Printf("\n")
			nextToken := sampler.Sample(logitsData, inputTokens, e.Config.VocabSize)
			fmt.Printf("Sampler Picked: %d\n", nextToken)

			// Log logits if enabled
			if e.ActLogger.IsEnabled() {
				// We need to pass nil or a default list if we removed expectedTokens definition
				e.ActLogger.LogLogits(logitsData, nil)
			}

			result = append(result, nextToken)
		}

		currentF32.ReturnToPool()

		// Increment CachePos after processing the token
		e.CachePos++
		metrics.RecordInference(1, time.Since(tToken))
		e.Ctx.AutoreleasePoolPop(pool)
	}

	// Phase 2: Generation loop (remaining tokens)
	for i := 1; i < tokensToGenerate; i++ {
		// Autorelease Pool for this iteration
		pool := e.Ctx.AutoreleasePoolPush()

		tToken := time.Now()
		lastToken := result[len(result)-1]

		if lastToken < 0 || lastToken >= e.Weights.TokenEmb.Rows() {
			return nil, fmt.Errorf("token %d is out of vocab range", lastToken)
		}

		current := e.Weights.TokenEmb.EmbeddingLookup(lastToken, e.GlobalScale)

		currentF32 := current.ToF32()
		current.ReturnToPool() // Release F16 embedding

		// Layers (Attention + FFN)
		for l := 0; l < e.Config.Layers; l++ {
			log.Printf("DEBUG_PRECISION: Layer %d, Dim=%d, PrecisionMode=%d (0=Auto, 1=FP16, 2=F32FFN, 3=Mixed)", l, e.Config.Dim, e.Config.PrecisionMode)
			// e.updateCurrentLayer(l) // This function call is not defined in the provided context. Removed.
			attnNorm := e.Weights.AttnNorm[l]
			q := e.Weights.AttnQ[l]
			k := e.Weights.AttnK[l]
			v := e.Weights.AttnV[l]
			o := e.Weights.AttnO[l] // Corrected from AttnOutput
			ffnNorm := e.Weights.FfnNorm[l]
			ffnGate := e.Weights.FfnGate[l]
			ffnUp := e.Weights.FfnUp[l]
			ffnDown := e.Weights.FfnDown[l]

			kCache := e.KVCacheK[l] // Corrected from KCache
			vCache := e.KVCacheV[l] // Corrected from VCache

			if l == e.Config.Layers-1 && i == 0 {
				// ffnDown.ScanMax("Last Layer FFN Down Weight")
				// fmt.Printf("DEBUG: Eps: %e\n", e.Config.Eps)
			}

			// Layer now handles F32 currentF32, using mixed precision
			currentF32.Layer(l, attnNorm, q, k, v, o, ffnNorm, ffnGate, ffnUp, ffnDown, kCache, vCache,
				scratch, // Pass scratch
				e.TraceTracker,
				e.CachePos, e.Config.Heads, e.Config.KVHeads, e.Config.HeadDim, e.Config.RopeTheta, e.Config.Eps, e.Config.HiddenDim, e.Config.SeqLen, e.Config.WindowSize, e.GlobalScale, samplerConfig.DebugActivations, int(e.Config.PrecisionMode))

			// Log layer activation details if enabled (generation phase)
			if e.ActLogger.IsEnabled() && i >= len(inputTokens)-1 {
				qSample := GetSampleFromTensor(scratch.QPart.ToHost(), 10)
				kSample := GetSampleFromTensor(scratch.KPart.ToHost(), 10)
				vSample := GetSampleFromTensor(scratch.VPart.ToHost(), 10)

				qMax := GetMaxFromTensor(scratch.QPart.ToHost())
				kMax := GetMaxFromTensor(scratch.KPart.ToHost())
				vMax := GetMaxFromTensor(scratch.VPart.ToHost())
				attnMax := GetMaxFromTensor(scratch.AttOut.ToHost())
				ffnMax := GetMaxFromTensor(currentF32.ToHost())

				e.ActLogger.LogLayer(l, qMax, kMax, vMax, attnMax, ffnMax,
					qSample, kSample, vSample,
					scratch.QPart.ToHost(), scratch.KPart.ToHost(), scratch.VPart.ToHost(),
					scratch.AttOut.ToHost(), currentF32.ToHost())
			}

			if samplerConfig.DebugActivations {
				currentF32.ScanMax(fmt.Sprintf("[Pos %d] Layer %d Output", e.CachePos, l))
			}
		}

		// Final Norm (F32 -> F16)

		// Use Into to avoid alloc
		currentF32.RMSNormFP32_ToF16_Into(e.Weights.OutputNorm, e.Config.Eps, scratch.Normed)

		// Output Head (F16 -> F32 Logits)
		// Reuse pre-allocated logits buffer
		// Output into scratch.Logits (which is 'logits')
		normedF32 := scratch.Normed.ToF32()
		normedF32.LinearF32_Into(e.Weights.Output, logits, e.GlobalScale)
		normedF32.ReturnToPool()

		// Logic update: RMSNormFP32_ToF16_Into does NOT allocate.
		// It writes to scratch.Normed.
		// No need to ReturnToPool for scratch.Normed (it's persistent scratch).

		// Logic update: RMSNormFP32_ToF16_Into does NOT allocate.
		// It writes to scratch.Normed.
		// No need to ReturnToPool for scratch.Normed (it's persistent scratch).

		e.Ctx.Synchronize()
		logitsData := logits.ToHost()

		// Save logits for debugging
		if e.LastLogits == nil || len(e.LastLogits) < len(logitsData) {
			e.LastLogits = make([]float32, len(logitsData))
		}
		copy(e.LastLogits, logitsData)

		currentF32.ReturnToPool()

		// Construct full history for Repetition Penalty
		// We need to include input tokens + generated tokens
		fullHistory := make([]int, 0, len(inputTokens)+len(result))
		fullHistory = append(fullHistory, inputTokens...)
		fullHistory = append(fullHistory, result...)

		// DEBUG: Print top 3 logits for each generation
		var m1, m2, m3 float32 = -1e9, -1e9, -1e9
		var i1, i2, i3 int = -1, -1, -1
		for idx, v := range logitsData {
			if v > m1 {
				m3, i3 = m2, i2
				m2, i2 = m1, i1
				m1, i1 = v, idx
			} else if v > m2 {
				m3, i3 = m2, i2
				m2, i2 = v, idx
			} else if v > m3 {
				m3, i3 = v, idx
			}
		}
		fmt.Printf("Gen %d Top 3: [%d:%.2f] [%d:%.2f] [%d:%.2f]\n", i, i1, m1, i2, m2, i3, m3)

		maxIdx := sampler.Sample(logitsData, fullHistory, e.Config.VocabSize)

		if maxIdx == 128001 { // <|end_of_text|> in Llama 3
			e.Ctx.AutoreleasePoolPop(pool)
			break
		}

		result = append(result, maxIdx)
		e.CachePos++
		metrics.RecordInference(1, time.Since(tToken))
		e.Ctx.AutoreleasePoolPop(pool)
	}

	// Save activation log if enabled
	if e.ActLogger.IsEnabled() {
		err := e.ActLogger.SaveToFile("activation_debug.json")
		if err != nil {
			log.Printf("Warning: Failed to save activation log: %v", err)
		}
	}

	return result, nil
}

func (e *Engine) initKVCache() error {
	layers := e.Config.Layers
	windowSize := e.Config.WindowSize
	if windowSize == 0 {
		// Default to full sequence length if not specified
		windowSize = e.Config.SeqLen
	}
	kvHeads := e.Config.KVHeads
	headDim := e.Config.HeadDim

	// Check verify dims
	if kvHeads == 0 || headDim == 0 {
		return errors.New("invalid kv dims")
	}

	e.KVCacheK = make([]*device.Tensor, layers)
	e.KVCacheV = make([]*device.Tensor, layers)

	// SLIDING WINDOW ATTENTION:
	// For Mistral, we use a rolling buffer KV cache of size WindowSize (4096)
	// instead of full SeqLen. This bounds memory usage and implements
	// the sliding window attention mechanism.
	// Cache is indexed as: cacheIdx = pos % windowSize

	rows := windowSize // Changed from seqLen to windowSize
	cols := kvHeads * headDim

	// Align 4096
	align := func(n int) int {
		return (n + 4095) & ^4095
	}
	szPerTensor := align(rows * cols * 2) // F16
	totalSz := layers * 2 * szPerTensor

	fmt.Printf("Alloc KVCache Heap: %d bytes\n", totalSz)
	heap := e.Ctx.NewHeap(totalSz)
	if heap == nil {
		return errors.New("KVCache Heap alloc failed")
	}

	// We don't store heap ref in Engine struct for now (leaks at exit),
	// assuming Engine lives for lifetime of process or we add Free() later.

	for i := 0; i < layers; i++ {
		e.KVCacheK[i] = e.Ctx.NewBufferFromHeap(heap, szPerTensor, rows, cols, device.DataTypeF16)
		e.KVCacheV[i] = e.Ctx.NewBufferFromHeap(heap, szPerTensor, rows, cols, device.DataTypeF16)

		if e.KVCacheK[i] == nil || e.KVCacheV[i] == nil {
			return errors.New("KVCache Buffer alloc failed")
		}
	}

	return nil
}

func (e *Engine) Close() {
	if e.Ctx != nil {
		e.Ctx.Free()
	}
}

// LoadWeightFromGGUF decodes weights to F32 for CPU reference
func LoadWeightFromGGUF(e *Engine, name string) []float32 {
	var t *gguf.TensorInfo
	for _, tensor := range e.Model.Tensors {
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

	if t.Type == gguf.GGMLTypeQ4_K {
		return gguf.DequantizeQ4K(t.Data, numElements)
	} else if t.Type == gguf.GGMLTypeQ6_K {
		return gguf.DequantizeQ6K(t.Data, numElements)
	} else if t.Type == gguf.GGMLTypeF32 {
		out := make([]float32, numElements)
		for i := 0; i < numElements; i++ {
			bits := binary.LittleEndian.Uint32(t.Data[i*4 : (i+1)*4])
			out[i] = math.Float32frombits(bits)
		}
		return out
	} else if t.Type == gguf.GGMLTypeF16 {
		out := make([]float32, numElements)
		for i := 0; i < numElements; i++ {
			bits := binary.LittleEndian.Uint16(t.Data[i*2 : (i+1)*2])
			out[i] = device.Float16ToFloat32(bits)
		}
		return out
	}
	panic(fmt.Sprintf("Unsupported type %d for %s", t.Type, name))
}

func isNormWeight(name string) bool {
	return strings.HasSuffix(name, "attn_norm.weight") || strings.HasSuffix(name, "ffn_norm.weight") || name == "output_norm.weight"
}

// ValidateTensorDimensions validates tensor dimensions based on quantization type
func ValidateTensorDimensions(name string, rows, cols int, ggufType gguf.GGMLType) error {
	switch ggufType {
	case gguf.GGMLTypeF32, gguf.GGMLTypeF16:
		if rows <= 0 || cols <= 0 {
			return fmt.Errorf("invalid dimensions: rows=%d, cols=%d", rows, cols)
		}
	case gguf.GGMLTypeQ4_0:
		if cols%32 != 0 {
			return fmt.Errorf("Q4_0 requires cols divisible by 32, got cols=%d", cols)
		}
		if rows <= 0 || cols <= 0 {
			return fmt.Errorf("invalid Q4_0 dimensions: rows=%d, cols=%d", rows, cols)
		}
	case gguf.GGMLTypeQ4_K:
		if cols%256 != 0 {
			return fmt.Errorf("Q4_K requires cols divisible by 256, got cols=%d", cols)
		}
		if rows <= 0 || cols <= 0 {
			return fmt.Errorf("invalid Q4_K dimensions: rows=%d, cols=%d", rows, cols)
		}
	case gguf.GGMLTypeQ6_K:
		if cols%256 != 0 {
			return fmt.Errorf("Q6_K requires cols divisible by 256, got cols=%d", cols)
		}
		if rows <= 0 || cols <= 0 {
			return fmt.Errorf("invalid Q6_K dimensions: rows=%d, cols=%d", rows, cols)
		}
	}
	return nil
}

// GetLastLogits returns the logits from the most recent inference step
func (e *Engine) GetLastLogits() []float32 {
	return e.LastLogits
}
