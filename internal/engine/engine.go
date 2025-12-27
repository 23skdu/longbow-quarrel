package engine

import (
	"encoding/binary"
	"errors"
	"fmt"
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
	
	// Initialize KV Cache (now that we have dimensions)
	if err := e.initKVCache(); err != nil {
		return err
	}
	
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
		// Parse tensor name for mapping
		// e.g. "blk.0.attn_q.weight" or "token_embd.weight"
		
		var mt *device.Tensor
		
		// Only supporting F32 for now (real models are F16/Qx, but we kept F32 limit for simplicity in previous steps)
		// For Llama 3 support we MUST likely support F16 input because smollm2 is F16.
		// Let's relax check or add F16 support.
		// If input is F16, we can copy directly (since our Metal tensors are F16).
		
		cols := int(t.Dimensions[0])
		rows := 1
		if len(t.Dimensions) > 1 {
			rows = int(t.Dimensions[1])
		}
		for i := 2; i < len(t.Dimensions); i++ {
			rows *= int(t.Dimensions[i])
		}
		
		mt = e.Ctx.NewTensor(rows, cols)
		numElements := rows * cols /* rows * cols */
		
		if t.Type == gguf.GGMLTypeF32 {
			dataBytes := numElements * 4
			if uint64(len(t.Data)) < uint64(dataBytes) { return errors.New("tensor data truncated") }
			
			rawBytes := t.Data[:dataBytes]
			f32Data := make([]float32, numElements)
			for i := 0; i < numElements; i++ {
				bits := binary.LittleEndian.Uint32(rawBytes[i*4 : (i+1)*4])
				f32Data[i] = math.Float32frombits(bits)
			}
			mt.LoadFrom(f32Data)
			
		} else if t.Type == gguf.GGMLTypeF16 {
			// Direct copy!
			dataBytes := numElements * 2
			if uint64(len(t.Data)) < uint64(dataBytes) { return errors.New("tensor data truncated") }
			
			// LoadFromRaw expects bytes to be FP16
			mt.LoadFromRaw(t.Data[:dataBytes])
		} else {
			// Skip quantization for now
			continue // verify if smollm2 is Q or F16. Usually F16.
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
			// naive parse
			fmt.Sscanf(parts[1], "%d", &layerIdx)
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
			case "ffn_down.weight":
				e.Weights.FfnDown[layerIdx] = mt
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
	
	tStart := time.Now()
	result := make([]int, 0, tokensToGenerate)
	
	// Create dummy hidden state (Batch=1, Dim=1) 
	// Since our "token_embd.weight" from test is 1x1, let's match that.
	// Dim = 1.
	// dim := e.Config.Dim // use real config
	
	// Allocate hidden state on GPU
	// hidden := e.Ctx.NewTensor(1, dim)
	// hidden.LoadFrom([]float32{1.0})
	
	// Main Generation Loop
	for i := 0; i < tokensToGenerate; i++ {
		tToken := time.Now()
		
		// 1. Get Input Embedding
		// Input is a single token ID (last one)
		lastToken := 0
		if i == 0 {
			if len(inputTokens) > 0 {
				lastToken = inputTokens[len(inputTokens)-1]
			}
		} else {
			lastToken = result[len(result)-1]
		}
		
		// Lookup embedding
		// TokenEmb: [Vocab, Dim]
		// Rows=Vocab, Cols=Dim.
		
		var current *device.Tensor
		
		if e.Weights.TokenEmb != nil && lastToken < e.Weights.TokenEmb.Rows() {
			current = e.Weights.TokenEmb.EmbeddingLookup(lastToken)
		} else {
			// Fallback (or first token if prompt empty, though lastToken handles that)
			current = e.Ctx.NewTensor(1, e.Config.Dim)
			// Zero init?
		}
		
		// Layers
		for l := 0; l < e.Config.Layers; l++ {
			residual := current // Keep reference

			// 1-2. Fused RMSNorm + QKV Linear (eliminates intermediate buffer)
			q := current.RMSNormLinear(e.Weights.AttnNorm[l], e.Weights.AttnQ[l], e.Config.Eps)
			k := current.RMSNormLinear(e.Weights.AttnNorm[l], e.Weights.AttnK[l], e.Config.Eps)
			v := current.RMSNormLinear(e.Weights.AttnNorm[l], e.Weights.AttnV[l], e.Config.Eps)

			// 3. RoPE
			// Q: Heads * HeadDim
			// K: KVHeads * HeadDim
			// SeqLen = 1 (current token)
			q.RoPE(e.CachePos, e.Config.HeadDim, e.Config.Heads, 1)
			k.RoPE(e.CachePos, e.Config.HeadDim, e.Config.KVHeads, 1)

			// 4. KV Cache Update
			// Store current k,v into cache at e.CachePos
			k.StoreKV(v, e.KVCacheK[l], e.KVCacheV[l], e.CachePos, e.Config.Heads, e.Config.HeadDim)
			
			// 5. Attention (GQA)
			// Using fused kernel: q is [1, num_heads * head_dim]
			weighted := q.Attention(e.KVCacheK[l], e.KVCacheV[l], e.CachePos, e.Config.Heads, e.Config.KVHeads, e.Config.HeadDim)

			// Output Proj
			// AttnO is [Dim, Dim]. Linear works.
			out := weighted.Linear(e.Weights.AttnO[l])

			// Residual Add
			current = residual.Add(out)

			// FFN: Fused RMSNorm + Gate/Up projections
			g := current.RMSNormLinear(e.Weights.FfnNorm[l], e.Weights.FfnGate[l], e.Config.Eps)
			up := current.RMSNormLinear(e.Weights.FfnNorm[l], e.Weights.FfnUp[l], e.Config.Eps)

			// SwiGLU: silu(gate) * up
			// SwiGLU: silu(gate) * up
			// Kernel: val * silu(gate). So val=up, gate=g.
			activated := up.SwiGLU(g)
			down := activated.Linear(e.Weights.FfnDown[l])

			current = residual.Add(down)
		}

		// Final Norm
		normed := current.RMSNorm(e.Weights.OutputNorm, e.Config.Eps)

		// Output Head
		logits := normed.Linear(e.Weights.Output)

		// ArgMax
		logitsData := logits.ToHost()
		maxIdx := 0
		maxVal := logitsData[0]
		for idx, val := range logitsData {
			if val > maxVal {
				maxVal = val
				maxIdx = idx
			}
		}
		nextToken := maxIdx
		result = append(result, nextToken)

		e.CachePos++
		metrics.RecordInference(1, time.Since(tToken)) 
	}
	
	_ = time.Since(tStart)
	
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
		
		// TODO: Zero init? NewTensor might be garbage.
		// We should probably zero it or track valid length.
		// We track CachePos.
	}
	
	e.CachePos = 0
	return nil
}

func (e *Engine) Close() {
	if e.Ctx != nil {
		e.Ctx.Free()
	}
}
