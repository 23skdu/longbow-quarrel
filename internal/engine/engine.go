package engine

import (
	"encoding/binary"
	"errors"
	"fmt"
	"log"
	"math"
	"runtime"

	"time"

	"strings"
	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
	"github.com/23skdu/longbow-quarrel/internal/metrics"
)

func NewEngine(modelPath string, debugDequant bool) (*Engine, error) {
	// Initialize Metal Context
	ctx := device.NewContext()
	
	e := &Engine{
		Ctx: ctx,
		Weights: &LlamaWeights{},
		ActLogger: NewActivationLogger(),
	}
	e.Config.DebugDequant = debugDequant
	
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
	if _, ok := f.KV["llama.rope.freq_base"]; ok {
		// FORCE 10k for Debug
		e.Config.RopeTheta = 10000.0
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
		e.Config.VocabSize = int(toFloat64(vSize))
		log.Printf("[MODEL] Vocab Size: %d", e.Config.VocabSize)
	} else {
		// Fallback for Smollm2 / Llama3 if missing
		e.Config.VocabSize = 49152 
		log.Printf("[MODEL] Vocab Size (Default): %d", e.Config.VocabSize)
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
		
		// Map tensor types
		// Validate F16 Conversion
		if device.Float32ToFloat16(0.0) != 0 {
			panic(fmt.Sprintf("Float32ToFloat16(0.0) = %x (Expected 0)", device.Float32ToFloat16(0.0)))
		}
		
		if false { } else {
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
				if t.Name == "blk.0.attn_norm.weight" {
					var sum float64
					for _, v := range f32Data {
						sum += float64(math.Abs(float64(v)))
					}
					mean := float32(sum / float64(numElements))
					fmt.Printf("GlobalScale (Disabled): Mean=%e Scale=1.0 FIRST_4=%x\n", mean, rawBytes[:16])
				}
			} else if t.Type == gguf.GGMLTypeF16 {
				mt = e.Ctx.NewTensor(rows, cols)
				// ... F16 load ...
				dataBytes := numElements * 2
				if uint64(len(t.Data)) < uint64(dataBytes) { 
					return fmt.Errorf("tensor %s data truncated", t.Name)
				}
				mt.LoadFromRaw(t.Data[:dataBytes])
				


			} else if t.Type == gguf.GGMLTypeQ4_K {
				// Type 12 (Q4_K). 
				
				if e.Config.Dim < 1024 || strings.Contains(t.Name, "token_embd.weight") {
					f32Data := gguf.DequantizeQ4K(t.Data, numElements)
					mt = e.Ctx.NewTensor(rows, cols)
					mt.LoadFrom(f32Data)
				} else {
					// Large Models: Use Q4K Tensor and Kernels
					mt = e.Ctx.NewQ4KTensor(rows, cols)
					dataBytes := (numElements / 256) * 144
					// Check size matches (handle truncated data)
					if uint64(len(t.Data)) < uint64(dataBytes) {
						return fmt.Errorf("tensor %s data truncated (Need %d, Has %d)", t.Name, dataBytes, len(t.Data))
					}
					mt.LoadFromRaw(t.Data[:dataBytes])
					
				}

			} else if t.Type == gguf.GGMLTypeQ6_K {
				// Type 14 (Q6_K).
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

func (e *Engine) Infer(inputTokens []int, tokensToGenerate int, samplerConfig SamplerConfig) ([]int, error) {
	// Validation
	if len(inputTokens) == 0 {
		return nil, errors.New("empty input tokens")
	}
	fmt.Printf("DEBUG: Input Tokens: %v\n", inputTokens)

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
		
		// Layers (Attention + FFN)
		for l := 0; l < e.Config.Layers; l++ {
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
				e.CachePos, e.Config.Heads, e.Config.KVHeads, e.Config.HeadDim, e.Config.RopeTheta, e.Config.Eps, e.Config.HiddenDim, e.Config.SeqLen, e.GlobalScale)
			
			// Log layer output if enabled and first token
			if i == 0 {
				currentF32.ScanMax(fmt.Sprintf("Layer %d Output", l))
			}
		}

		// If this is the LAST prompt token, sample the first next token
		if i == len(inputTokens)-1 {
			// Final Norm (F32 -> F16)
			// Debug Output Norm Weights
			if i == 0 {
				// e.Weights.OutputNorm.ScanMax("Output Norm Weights")
			}
			currentF32.RMSNormFP32_ToF16_Into(e.Weights.OutputNorm, e.Config.Eps, scratch.Normed)
			
			// Output Head (F16 -> F32 Logits)
			// scratch.Normed contains result. Use it.
			scratch.Normed.LinearToFP32_Into(e.Weights.Output, logits)
			logitsData := logits.ToHost()
			
			// Use Sampler
			// DEBUG: Print top 3 manually
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
			fmt.Printf("Top 3: [%d:%.2f] [%d:%.2f] [%d:%.2f]\n", i1, m1, i2, m2, i3, m3)
			nextObj := sampler.Sample(logitsData, inputTokens, e.Config.VocabSize)
			
			// Log logits if enabled
			if e.ActLogger.IsEnabled() {
				// We need to pass nil or a default list if we removed expectedTokens definition
				e.ActLogger.LogLogits(logitsData, nil)
			}

			result = append(result, nextObj)
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

			if l == e.Config.Layers - 1 && i == 0 {
				// ffnDown.ScanMax("Last Layer FFN Down Weight")
				// fmt.Printf("DEBUG: Eps: %e\n", e.Config.Eps)
			}

			// Layer now handles F32 currentF32, using mixed precision
			currentF32.Layer(l, attnNorm, q, k, v, o, ffnNorm, ffnGate, ffnUp, ffnDown, kCache, vCache,
				scratch, // Pass scratch
				e.CachePos, e.Config.Heads, e.Config.KVHeads, e.Config.HeadDim, e.Config.RopeTheta, e.Config.Eps, e.Config.HiddenDim, e.Config.SeqLen, e.GlobalScale) // Corrected variable names
		}
		
		// Final Norm (F32 -> F16)
		
		// Use Into to avoid alloc
		currentF32.RMSNormFP32_ToF16_Into(e.Weights.OutputNorm, e.Config.Eps, scratch.Normed)
		// scratch.Normed.ScanMax("DEBUG_FINAL: Normed")
		
		// Output Head (F16 -> F32 Logits)
		// Reuse pre-allocated logits buffer
		// scratch.Normed contains result. Use it.
		// Output into scratch.Logits (which is 'logits')
		scratch.Normed.LinearToFP32_Into(e.Weights.Output, logits)
		// logits.ScanMax("DEBUG_FINAL: Logits")
		
		// Logic update: RMSNormFP32_ToF16_Into does NOT allocate.
		// It writes to scratch.Normed.
		// No need to ReturnToPool for scratch.Normed (it's persistent scratch).
		
		logitsData := logits.ToHost()
		
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
