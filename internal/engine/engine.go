package engine

import (
	"encoding/binary"
	"errors"
	"fmt"
	"log"
	"math"

	"strings"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
	"github.com/23skdu/longbow-quarrel/internal/metrics"
)

func NewEngine(modelPath string) (*Engine, error) {
	// Initialize Metal Context
	ctx := device.NewContext()
	
	e := &Engine{
		Ctx: ctx,
		Weights: &LlamaWeights{},
	}
	
	// Load Model
	if err := e.loadModel(modelPath); err != nil {
		ctx.Free()
		return nil, err
	}
	
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
	if val, ok := f.KV["llama.block_count"]; ok {
		e.Config.Layers = int(toFloat64(val)) // GGUF numbers can be various types
	} else {
		e.Config.Layers = 1 // Default for test
	}
	
	// Embedding Dim
	if val, ok := f.KV["llama.embedding_length"]; ok {
		e.Config.Dim = int(toFloat64(val))
	}
	
	// Heads
	if val, ok := f.KV["llama.attention.head_count"]; ok {
		e.Config.Heads = int(toFloat64(val))
	} else {
		return errors.New("missing llama.attention.head_count")
	}

	// Hidden Dim (FFN context)
	if val, ok := f.KV["llama.feed_forward_length"]; ok {
		e.Config.HiddenDim = int(toFloat64(val))
	} else {
		e.Config.HiddenDim = 4 * e.Config.Dim // fallback
	}
	
	// KV Heads (GQA)
	if val, ok := f.KV["llama.attention.head_count_kv"]; ok {
		e.Config.KVHeads = int(toFloat64(val))
	} else {
		// Default to Heads (MHA)
		e.Config.KVHeads = e.Config.Heads
	}
	
	// Head Dim
	e.Config.HeadDim = e.Config.Dim / e.Config.Heads
	
	// Seq Len (Context)
	if val, ok := f.KV["llama.context_length"]; ok {
		e.Config.SeqLen = int(toFloat64(val))
	} else {
		e.Config.SeqLen = 2048 // default
	}
	
	// RoPE Freq
	if val, ok := f.KV["llama.rope.freq_base"]; ok {
		e.Config.RopeTheta = float32(toFloat64(val))
	} else {
		e.Config.RopeTheta = 10000.0
	}
	
	// RMS Norm Eps
	if val, ok := f.KV["llama.attention.layer_norm_rms_epsilon"]; ok {
		e.Config.Eps = float32(toFloat64(val))
	} else {
		e.Config.Eps = 1e-5
	}
	
	// Log Model Architecture
	if arch, ok := f.KV["general.architecture"].(string); ok {
		log.Printf("[MODEL] Architecture: %s", arch)
	}
	if vSize, ok := f.KV["llama.vocab_size"]; ok {
		log.Printf("[MODEL] Vocab Size: %v", vSize)
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
		
		// fmt.Printf("[TENSOR] %-30s | Type: %-10v | Shape: [%d, %d]\n", t.Name, t.Type, rows, cols)

		var mt *device.Tensor
		numElements := rows * cols
		
		// Map tensor types
		// Validate F16 Conversion
		if device.Float32ToFloat16(0.0) != 0 {
			panic(fmt.Sprintf("Float32ToFloat16(0.0) = %x (Expected 0)", device.Float32ToFloat16(0.0)))
		}
		
		if false { } else {
			// Standard F16 Tensor (allocated as F16, loaded as F32 or F16)
			mt = e.Ctx.NewTensor(rows, cols)
			
			if t.Type == gguf.GGMLTypeF32 {
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
				
			} else if t.Type == gguf.GGMLTypeF16 {
				// ... F16 load ...
				dataBytes := numElements * 2
				if uint64(len(t.Data)) < uint64(dataBytes) { 
					return fmt.Errorf("tensor %s data truncated", t.Name)
				}
				mt.LoadFromRaw(t.Data[:dataBytes])
				


			} else if t.Type == gguf.GGMLTypeQ4_K {
				// Type 12 (Q4_K). 
				// token_embd must be CPU Dequantized to F16.
				if t.Name == "token_embd.weight" {
					fmt.Printf("DEBUG: token_embd.weight: t.Offset=%d\n", t.Offset)
                    // DEBUG: Inspect first few bytes (Row 0 and Row 1)
                    if len(t.Data) > 2320 {
                        fmt.Printf("DEBUG: token_embd.weight Row 0 [0:16] = %x\n", t.Data[:16])
                        fmt.Printf("DEBUG: token_embd.weight Row 1 [2304:2320] = %x\n", t.Data[2304:2320])
                    }
					mt = e.Ctx.NewTensor(rows, cols)
					f32Data := gguf.DequantizeQ4K(t.Data, numElements)
					
					mt.LoadFrom(f32Data)
				} else if t.Name == "blk.0.attn_q.weight" {
					fmt.Printf("DEBUG_OFFSET: blk.0.attn_q.weight Offset=%d\n", t.Offset)
					if len(t.Data) >= 32 {
						fmt.Printf("DEBUG_BYTES: blk.0.attn_q.weight [0:32] = %x\n", t.Data[:32])
					}
					// FIX: Actually Load the Tensor!
					mt = e.Ctx.NewQ4KTensor(rows, cols)
					dataBytes := (numElements / 256) * 144
					if uint64(len(t.Data)) < uint64(dataBytes) {
						return fmt.Errorf("tensor %s data truncated", t.Name)
					}
					mt.LoadFromRaw(t.Data[:dataBytes])
				} else {
					// Check blk.1.ffn_down size
					if strings.Contains(t.Name, "blk.1.ffn_down") {
                        expected := (rows * cols / 256) * 144
                        if len(t.Data) < expected {
                            fmt.Printf("CRITICAL: %s Truncated! %d < %d\n", t.Name, len(t.Data), expected)
                        } else {
                            fmt.Printf("DEBUG: %s Size OK: %d\n", t.Name, len(t.Data))
                        }
                    }
					
					mt = e.Ctx.NewQ4KTensor(rows, cols)
					dataBytes := (numElements / 256) * 144
					if uint64(len(t.Data)) < uint64(dataBytes) {
						return fmt.Errorf("tensor %s data truncated", t.Name)
					}
					mt.LoadFromRaw(t.Data[:dataBytes])
				}

			} else if t.Type == gguf.GGMLTypeQ6_K {
				// Type 14 (Q6_K).
				// CPU Dequantize (No GPU Kernel yet)
				f32Data := gguf.DequantizeQ6K(t.Data, numElements)
				if t.Name == "output.weight" {
					fmt.Printf("DEBUG: output.weight probe [0:10]: %v\n", f32Data[:10])
				}
				mt = e.Ctx.NewTensor(rows, cols)
				mt.LoadFrom(f32Data)

			} else if t.Type == gguf.GGMLTypeQ4_K_S { // 99 Unused
				mt = e.Ctx.NewTensor(rows, cols) // fallback
			} else {
				fmt.Printf("Skipping unsupported tensor type: %d (%s)\n", t.Type, t.Name)
				continue
			}
		}
		
		if mt != nil {
			if t.Type == gguf.GGMLTypeF16 || t.Type == gguf.GGMLTypeF32 || t.Type == gguf.GGMLTypeQ6_K || t.Name == "token_embd.weight" {
				mt.ScanNaNs(t.Name)
				mt.ScanMax(t.Name)
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
			if len(parts) < 3 { continue }
			
			// Parse N
			layerIdx := 0
			if n, err := fmt.Sscanf(parts[1], "%d", &layerIdx); n != 1 || err != nil {
				continue
			}
			if layerIdx >= layers { continue }
			
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
	if e.Weights.Output == nil && e.Weights.TokenEmb != nil {
		e.Weights.Output = e.Weights.TokenEmb
	}
	
	return nil 
}

// Helper for loose typing in GGUF KV
func toFloat64(v interface{}) float64 {
	switch i := v.(type) {
	case float64: return i
	case float32: return float64(i)
	case uint64: return float64(i)
	case uint32: return float64(i)
	case int32: return float64(i)
	case int64: return float64(i)
	case int: return float64(i)
	default: return 0
	}
}

func (e *Engine) Infer(inputTokens []int, tokensToGenerate int) ([]int, error) {
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
	
	// Reset KV cache position for each inference call
	e.CachePos = 0
	
	// Main Generation Loop
	
	// Create scratch buffers for the layer fusion
	s1 := e.Ctx.NewTensorPooled(1, e.Config.Dim*4) // Oversized for various intermediates
	s2 := e.Ctx.NewTensorPooled(1, e.Config.Dim*4)
	s3 := e.Ctx.NewTensorPooled(1, e.Config.Dim*4)
	s4 := e.Ctx.NewTensorPooled(1, e.Config.Heads*e.Config.SeqLen*2) // *2 for float32 scores storage

	// Phase 1: Prefill all input tokens
	for i := 0; i < len(inputTokens); i++ {
		tToken := time.Now()
		lastToken := inputTokens[i]
		

		
		current := e.Weights.TokenEmb.EmbeddingLookup(lastToken)
		
		// DEBUG: Verify Input on GPU
		// if i == 0 {
		// 	current.ScanMean("Embed Output")
		// }

		// Layers (Attention + FFN)
		for l := 0; l < e.Config.Layers; l++ {
			current.Layer(e.Weights.AttnNorm[l], e.Weights.AttnQ[l], e.Weights.AttnK[l], e.Weights.AttnV[l], e.Weights.AttnO[l],
				e.Weights.FfnNorm[l], e.Weights.FfnGate[l], e.Weights.FfnUp[l], e.Weights.FfnDown[l],
				e.KVCacheK[l], e.KVCacheV[l], s1, s2, s3, s4,
				e.CachePos, e.Config.Heads, e.Config.KVHeads, e.Config.HeadDim,
				e.Config.RopeTheta, e.Config.Eps, e.Config.HiddenDim, e.Config.SeqLen)
		}

		// If this is the LAST prompt token, sample the first next token
		if i == len(inputTokens)-1 {
			normed := current.RMSNorm(e.Weights.OutputNorm, e.Config.Eps)
			logits := normed.Linear(e.Weights.Output)
			logitsData := logits.ToHost()

			maxIdx := 0
			maxVal := logitsData[0]
			for idx, val := range logitsData {
				if val > maxVal {
					maxVal = val
					maxIdx = idx
				}
			}
			result = append(result, maxIdx)
		}

		e.CachePos++
		metrics.RecordInference(1, time.Since(tToken))
	}

	// Phase 2: Generation loop (remaining tokens)
	for i := 1; i < tokensToGenerate; i++ {
		tToken := time.Now()
		lastToken := result[len(result)-1]
		
		if lastToken < 0 || lastToken >= e.Weights.TokenEmb.Rows() {
			return nil, fmt.Errorf("token %d is out of vocab range", lastToken)
		}
		
		current := e.Weights.TokenEmb.EmbeddingLookup(lastToken)

		// Layers (Attention + FFN)
		for l := 0; l < e.Config.Layers; l++ {
			current.Layer(e.Weights.AttnNorm[l], e.Weights.AttnQ[l], e.Weights.AttnK[l], e.Weights.AttnV[l], e.Weights.AttnO[l],
				e.Weights.FfnNorm[l], e.Weights.FfnGate[l], e.Weights.FfnUp[l], e.Weights.FfnDown[l],
				e.KVCacheK[l], e.KVCacheV[l], s1, s2, s3, s4,
				e.CachePos, e.Config.Heads, e.Config.KVHeads, e.Config.HeadDim,
				e.Config.RopeTheta, e.Config.Eps, e.Config.HiddenDim, e.Config.SeqLen)
		}

		normed := current.RMSNorm(e.Weights.OutputNorm, e.Config.Eps)
		logits := normed.Linear(e.Weights.Output)
		logitsData := logits.ToHost()
		
		// Logit analysis for generation

		maxIdx := 0
		maxVal := logitsData[0]
		for idx, val := range logitsData {
			if val > maxVal {
				maxVal = val
				maxIdx = idx
			}
		}
		
		if maxIdx == 128001 { // <|end_of_text|> in Llama 3
			break
		}
		
		result = append(result, maxIdx)
		e.CachePos++
		metrics.RecordInference(1, time.Since(tToken))
	}

	return result, nil
}



func (e *Engine) initKVCache() error {
	layers := e.Config.Layers
	seqLen := e.Config.SeqLen
	kvHeads := e.Config.KVHeads
	headDim := e.Config.HeadDim
	
	// Check verify dims
	if kvHeads == 0 || headDim == 0 {
		return errors.New("invalid kv dims")
	}
	
	e.KVCacheK = make([]*device.Tensor, layers)
	e.KVCacheV = make([]*device.Tensor, layers)
	
	// Total size per layer:
	// Rows = seqLen
	// Cols = kvHeads * headDim (dim of K/V per token)
	// We store as [SeqLen, KVDim] usually?
	// Metal MatMul prefers [M, K] x [K, N] -> [M, N]
	// Attention Q * K^T.
	// K cache usually needs to be [KVDim, SeqLen] or transposable.
	// Let's stick to row-major [SeqLen, kvHeads * headDim] for storage?
	// Actually, for flash attention / efficient access, we often want [Heads, SeqLen, HeadDim].
	// But our basic kernels are 2D.
	// Let's allocate [SeqLen, kvHeads * headDim].
	
	rows := seqLen
	cols := kvHeads * headDim
	
	for i := 0; i < layers; i++ {
		e.KVCacheK[i] = e.Ctx.NewTensor(rows, cols)
		e.KVCacheV[i] = e.Ctx.NewTensor(rows, cols)
		
		// Zero init to prevent garbage data
		e.KVCacheK[i].ZeroInit()
		e.KVCacheV[i].ZeroInit()
	}
	
	e.CachePos = 0
	return nil
}

func (e *Engine) Close() {
	if e.Ctx != nil {
		e.Ctx.Free()
	}
}
