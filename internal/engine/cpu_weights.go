package engine

import (
	"encoding/binary"
	"fmt"
	"math"
	"sync"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
	"github.com/23skdu/longbow-quarrel/internal/simd"
)

// CPUWeights holds weights loaded in host memory for CPU execution or hybrid GPU/CPU offloading.
type CPUWeights struct {
	TokenEmb   [][]float32
	Output     []float32
	OutputNorm []float32
	AttnQ      [][]float32
	AttnK      [][]float32
	AttnV      [][]float32
	AttnO      [][]float32
	AttnNorm   [][]float32
	FfnGate    [][]float32
	FfnDown    [][]float32
	FfnUp      [][]float32
	FfnNorm    [][]float32

	// Qwen3.5 & hybrid layer support
	AttnQNorm [][]float32
	AttnKNorm [][]float32
	AttnQKV   [][]float32
	AttnGate  [][]float32
	SSMConv1d     [][]float32
	SSMConv1dBias [][]float32
	SSMA          [][]float32
	SSMD          [][]float32
	SSMAlpha      [][]float32
	SSMBeta       [][]float32
	SSMDtWeight   [][]float32
	SSMDtBias     [][]float32
	SSMNorm       [][]float32
	SSMOut        [][]float32

	// Memory-efficient raw tensor handles (e.g. Q8_0 directly from mmap)
	RawTokenEmb *gguf.TensorInfo
	RawOutput   *gguf.TensorInfo
	RawAttnQ    []*gguf.TensorInfo
	RawAttnK    []*gguf.TensorInfo
	RawAttnV    []*gguf.TensorInfo
	RawAttnO    []*gguf.TensorInfo
	RawFfnGate  []*gguf.TensorInfo
	RawFfnDown  []*gguf.TensorInfo
	RawFfnUp    []*gguf.TensorInfo
	RawAttnQKV  []*gguf.TensorInfo
	RawAttnGate []*gguf.TensorInfo
	RawSSMOut   []*gguf.TensorInfo
}

func loadCPUWeights(f *gguf.GGUFFile, cfg config.Config) (*CPUWeights, error) {
	w := &CPUWeights{}

	numLayers := cfg.Layers
	if numLayers <= 0 {
		numLayers = 1
	}

	w.TokenEmb = make([][]float32, 0)
	w.Output = make([]float32, 0)
	w.OutputNorm = make([]float32, 0)
	w.AttnQ = make([][]float32, numLayers)
	w.AttnK = make([][]float32, numLayers)
	w.AttnV = make([][]float32, numLayers)
	w.AttnO = make([][]float32, numLayers)
	w.AttnNorm = make([][]float32, numLayers)
	w.FfnGate = make([][]float32, numLayers)
	w.FfnDown = make([][]float32, numLayers)
	w.FfnUp = make([][]float32, numLayers)
	w.FfnNorm = make([][]float32, numLayers)

	w.AttnQNorm = make([][]float32, numLayers)
	w.AttnKNorm = make([][]float32, numLayers)
	w.AttnQKV = make([][]float32, numLayers)
	w.AttnGate = make([][]float32, numLayers)
	w.SSMConv1d = make([][]float32, numLayers)
	w.SSMConv1dBias = make([][]float32, numLayers)
	w.SSMA = make([][]float32, numLayers)
	w.SSMD = make([][]float32, numLayers)
	w.SSMAlpha = make([][]float32, numLayers)
	w.SSMBeta = make([][]float32, numLayers)
	w.SSMDtWeight = make([][]float32, numLayers)
	w.SSMDtBias = make([][]float32, numLayers)
	w.SSMNorm = make([][]float32, numLayers)
	w.SSMOut = make([][]float32, numLayers)

	w.RawAttnQ = make([]*gguf.TensorInfo, numLayers)
	w.RawAttnK = make([]*gguf.TensorInfo, numLayers)
	w.RawAttnV = make([]*gguf.TensorInfo, numLayers)
	w.RawAttnO = make([]*gguf.TensorInfo, numLayers)
	w.RawFfnGate = make([]*gguf.TensorInfo, numLayers)
	w.RawFfnDown = make([]*gguf.TensorInfo, numLayers)
	w.RawFfnUp = make([]*gguf.TensorInfo, numLayers)
	w.RawAttnQKV = make([]*gguf.TensorInfo, numLayers)
	w.RawAttnGate = make([]*gguf.TensorInfo, numLayers)
	w.RawSSMOut = make([]*gguf.TensorInfo, numLayers)

	for _, t := range f.Tensors {
		isQ8Matrix := t.Type == gguf.GGMLTypeQ8_0 && len(t.Dimensions) >= 2

		switch t.Name {
		case "token_embd.weight":
			w.RawTokenEmb = t
			if w.RawOutput == nil {
				w.RawOutput = t
			}
			if !isQ8Matrix {
				data, err := decodeTensorData(t)
				if err == nil {
					w.TokenEmb = append(w.TokenEmb, data)
					if len(w.Output) == 0 {
						w.Output = data
					}
				}
			}
		case "output.weight", "lm_head.weight":
			w.RawOutput = t
			if !isQ8Matrix {
				data, err := decodeTensorData(t)
				if err == nil {
					w.Output = data
				}
			}
		case "output_norm.weight":
			data, err := decodeTensorData(t)
			if err == nil {
				w.OutputNorm = data
			}
		default:
			var layer int
			var _, _ = fmt.Sscanf(t.Name, "blk.%d.", &layer)
			if layer < numLayers {
				switch {
				case containsStr(t.Name, "attn_q_norm.weight"):
					if data, err := decodeTensorData(t); err == nil {
						w.AttnQNorm[layer] = data
					}
				case containsStr(t.Name, "attn_k_norm.weight"):
					if data, err := decodeTensorData(t); err == nil {
						w.AttnKNorm[layer] = data
					}
				case containsStr(t.Name, "attn_norm.weight"):
					if data, err := decodeTensorData(t); err == nil {
						w.AttnNorm[layer] = data
					}
				case containsStr(t.Name, "ffn_norm.weight"), containsStr(t.Name, "post_attention_norm.weight"):
					if data, err := decodeTensorData(t); err == nil {
						w.FfnNorm[layer] = data
					}
				case containsStr(t.Name, "ssm_conv1d.weight"):
					if data, err := decodeTensorData(t); err == nil {
						w.SSMConv1d[layer] = data
					}
				case containsStr(t.Name, "ssm_conv1d.bias"):
					if data, err := decodeTensorData(t); err == nil {
						w.SSMConv1dBias[layer] = data
					}
				case containsStr(t.Name, "ssm_alpha.weight"):
					if data, err := decodeTensorData(t); err == nil {
						w.SSMAlpha[layer] = data
					}
				case containsStr(t.Name, "ssm_beta.weight"):
					if data, err := decodeTensorData(t); err == nil {
						w.SSMBeta[layer] = data
					}
				case containsStr(t.Name, "ssm_dt.weight"):
					if data, err := decodeTensorData(t); err == nil {
						w.SSMDtWeight[layer] = data
					}
				case containsStr(t.Name, "ssm_dt.bias"):
					if data, err := decodeTensorData(t); err == nil {
						w.SSMDtBias[layer] = data
					}
				case containsStr(t.Name, "ssm_norm.weight"):
					if data, err := decodeTensorData(t); err == nil {
						w.SSMNorm[layer] = data
					}
				case containsStr(t.Name, "ssm_a"):
					if data, err := decodeTensorData(t); err == nil {
						w.SSMA[layer] = data
					}
				case containsStr(t.Name, "ssm_d"):
					if data, err := decodeTensorData(t); err == nil {
						w.SSMD[layer] = data
					}
				case containsStr(t.Name, "attn_q.weight"):
					w.RawAttnQ[layer] = t
					if !isQ8Matrix {
						if data, err := decodeTensorData(t); err == nil {
							w.AttnQ[layer] = data
						}
					}
				case containsStr(t.Name, "attn_k.weight"):
					w.RawAttnK[layer] = t
					if !isQ8Matrix {
						if data, err := decodeTensorData(t); err == nil {
							w.AttnK[layer] = data
						}
					}
				case containsStr(t.Name, "attn_v.weight"):
					w.RawAttnV[layer] = t
					if !isQ8Matrix {
						if data, err := decodeTensorData(t); err == nil {
							w.AttnV[layer] = data
						}
					}
				case containsStr(t.Name, "attn_output.weight"):
					w.RawAttnO[layer] = t
					if !isQ8Matrix {
						if data, err := decodeTensorData(t); err == nil {
							w.AttnO[layer] = data
						}
					}
				case containsStr(t.Name, "attn_qkv.weight"):
					w.RawAttnQKV[layer] = t
					if !isQ8Matrix {
						if data, err := decodeTensorData(t); err == nil {
							w.AttnQKV[layer] = data
						}
					}
				case containsStr(t.Name, "attn_gate.weight"):
					w.RawAttnGate[layer] = t
					if !isQ8Matrix {
						if data, err := decodeTensorData(t); err == nil {
							w.AttnGate[layer] = data
						}
					}
				case containsStr(t.Name, "ssm_out.weight"):
					w.RawSSMOut[layer] = t
					if !isQ8Matrix {
						if data, err := decodeTensorData(t); err == nil {
							w.SSMOut[layer] = data
						}
					}
				case containsStr(t.Name, "ffn_gate.weight"):
					w.RawFfnGate[layer] = t
					if !isQ8Matrix {
						if data, err := decodeTensorData(t); err == nil {
							w.FfnGate[layer] = data
						}
					}
				case containsStr(t.Name, "ffn_down.weight"):
					w.RawFfnDown[layer] = t
					if !isQ8Matrix {
						if data, err := decodeTensorData(t); err == nil {
							w.FfnDown[layer] = data
						}
					}
				case containsStr(t.Name, "ffn_up.weight"):
					w.RawFfnUp[layer] = t
					if !isQ8Matrix {
						if data, err := decodeTensorData(t); err == nil {
							w.FfnUp[layer] = data
						}
					}
				}
			}
		}
	}

	return w, nil
}

func decodeTensorData(t *gguf.TensorInfo) ([]float32, error) {
	numElements := uint32(1)
	for _, d := range t.Dimensions {
		numElements *= uint32(d) // #nosec G115
	}

	switch t.Type {
	case gguf.GGMLTypeQ4_K:
		return gguf.DequantizeQ4K_SIMD(t.Data, int(numElements)), nil
	case gguf.GGMLTypeQ6_K:
		return gguf.DequantizeQ6K_SIMD(t.Data, int(numElements)), nil
	case gguf.GGMLTypeQ5_0:
		return gguf.DequantizeQ5_0(t.Data, int(numElements)), nil
	case gguf.GGMLTypeQ8_0:
		return gguf.DequantizeQ8_0(t.Data, int(numElements)), nil
	case gguf.GGMLTypeQ2_K:
		return gguf.DequantizeQ2K(t.Data, int(numElements)), nil
	case gguf.GGMLTypeQ3_K:
		return gguf.DequantizeQ3K(t.Data, int(numElements)), nil
	case gguf.GGMLTypeQ5_K:
		return gguf.DequantizeQ5K(t.Data, int(numElements)), nil
	case gguf.GGMLTypeQ4_0:
		return gguf.DequantizeQ4_0(t.Data, int(numElements)), nil
	case gguf.GGMLTypeF32:
		data := make([]float32, numElements)
		for i := uint32(0); i < numElements; i++ {
			offset := uint64(i) * 4
			bits := uint32(t.Data[offset]) | uint32(t.Data[offset+1])<<8 | uint32(t.Data[offset+2])<<16 | uint32(t.Data[offset+3])<<24
			data[i] = math.Float32frombits(bits)
		}
		return data, nil
	case gguf.GGMLTypeF16:
		data := make([]float32, numElements)
		for i := uint32(0); i < numElements; i++ {
			offset := uint64(i) * 2
			bits := uint16(t.Data[offset]) | uint16(t.Data[offset+1])<<8
			data[i] = gguf.Float16ToFloat32(bits)
		}
		return data, nil
	case gguf.GGMLTypeBF16:
		return gguf.DequantizeBF16(t.Data, int(numElements)), nil
	default:
		return make([]float32, numElements), nil
	}
}

func containsStr(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > len(substr) && (s[:len(substr)] == substr || containsStr(s[1:], substr)))
}

// CPUKVCache stores key and value representations across layers for a sequence.
type CPUKVCache struct {
	mu     sync.Mutex
	Keys   [][]float32 // [layerIdx] -> flattened [numTokens * kvHeads * headDim]
	Values [][]float32 // [layerIdx] -> flattened [numTokens * kvHeads * headDim]
	MaxLen int         // 0 = unbounded; >0 = sliding-window eviction with attention-sink

	// Mamba/SSM state for hybrid models (Qwen 3.5, Nemotron, etc.)
	ConvState [][]float32 // [layerIdx] -> flattened [dConv * dInner] ring buffer
	SSMState  [][]float32 // [layerIdx] -> flattened [dInner * dState] hidden state
}

// NewCPUKVCache creates a per-sequence KV cache for CPU execution.
func NewCPUKVCache(numLayers int) *CPUKVCache {
	if numLayers <= 0 {
		numLayers = 1
	}
	return &CPUKVCache{
		Keys:   make([][]float32, numLayers),
		Values: make([][]float32, numLayers),
	}
}

// NewCPUKVCacheWithWindow creates a sliding-window KV cache.
// windowSize=0 disables eviction (same as NewCPUKVCache).
func NewCPUKVCacheWithWindow(numLayers, windowSize int) *CPUKVCache {
	c := NewCPUKVCache(numLayers)
	c.MaxLen = windowSize
	return c
}

// Reset clears all cached keys, values, and SSM state.
func (c *CPUKVCache) Reset() {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	for i := range c.Keys {
		c.Keys[i] = c.Keys[i][:0]
		c.Values[i] = c.Values[i][:0]
	}
	for i := range c.ConvState {
		for j := range c.ConvState[i] {
			c.ConvState[i][j] = 0
		}
	}
	for i := range c.SSMState {
		for j := range c.SSMState[i] {
			c.SSMState[i][j] = 0
		}
	}
}

// ApplyLayerCPU executes a single transformer layer on CPU for CPUEngine or layer offloading.
func ApplyLayerCPU(w *CPUWeights, input []float32, layerIdx int, cfg config.Config) []float32 {
	return ApplyLayerCPUKV(w, input, layerIdx, 0, nil, cfg)
}

// ApplyLayerCPUKV executes a single transformer layer on CPU with rotary embeddings and KV cache.
func ApplyLayerCPUKV(w *CPUWeights, input []float32, layerIdx int, pos int, kv *CPUKVCache, cfg config.Config) []float32 {
	if layerIdx < 0 || w == nil || len(input) == 0 {
		return input
	}
	normed := make([]float32, len(input))
	if layerIdx < len(w.AttnNorm) && len(w.AttnNorm[layerIdx]) > 0 {
		simd.RMSNorm(input, w.AttnNorm[layerIdx], normed, 1, len(input), cfg.Eps)
	} else {
		copy(normed, input)
	}

	var out []float32

	if w.HasFullAttn(layerIdx) {
		var qWeight, kWeight, vWeight, oWeight []float32
		var rawQ, rawK, rawV, rawO *gguf.TensorInfo
		if layerIdx < len(w.AttnQ) {
			qWeight = w.AttnQ[layerIdx]
		}
		if layerIdx < len(w.AttnK) {
			kWeight = w.AttnK[layerIdx]
		}
		if layerIdx < len(w.AttnV) {
			vWeight = w.AttnV[layerIdx]
		}
		if layerIdx < len(w.AttnO) {
			oWeight = w.AttnO[layerIdx]
		}
		if layerIdx < len(w.RawAttnQ) {
			rawQ = w.RawAttnQ[layerIdx]
		}
		if layerIdx < len(w.RawAttnK) {
			rawK = w.RawAttnK[layerIdx]
		}
		if layerIdx < len(w.RawAttnV) {
			rawV = w.RawAttnV[layerIdx]
		}
		if layerIdx < len(w.RawAttnO) {
			rawO = w.RawAttnO[layerIdx]
		}

		qFull := w.MatVec(qWeight, rawQ, normed)
		k := w.MatVec(kWeight, rawK, normed)
		v := w.MatVec(vWeight, rawV, normed)

		// Derive correct head_dim from K weight dimensions
		// K: [hidden, kv_heads * head_dim] → head_dim = len(k) / kv_heads
		attnHeadDim := cfg.HeadDim
		if cfg.KVHeads > 0 && len(k) > 0 {
			derivedHeadDim := len(k) / cfg.KVHeads
			if derivedHeadDim > 0 {
				attnHeadDim = derivedHeadDim
			}
		}
		attnHeads := cfg.Heads
		if attnHeadDim > 0 {
			attnHeads = len(qFull) / (2 * attnHeadDim) // Q output = 2 * heads * head_dim (query + gate)
		}
		querySize := attnHeads * attnHeadDim

		// Split Q output: first half = query, second half = gate (Qwen 3.5 query-gate)
		query := qFull[:querySize]
		gate := qFull[querySize:]
		if len(gate) > querySize {
			gate = gate[:querySize]
		}

		// Per-head QK-norm (norm weight [head_dim] broadcast to all heads)
		if layerIdx < len(w.AttnQNorm) && len(w.AttnQNorm[layerIdx]) > 0 && attnHeadDim > 0 {
			normWeight := w.AttnQNorm[layerIdx]
			if len(normWeight) == attnHeadDim {
				qNormed := make([]float32, len(query))
				for h := 0; h < attnHeads; h++ {
					off := h * attnHeadDim
					simd.RMSNorm(query[off:off+attnHeadDim], normWeight, qNormed[off:off+attnHeadDim], 1, attnHeadDim, cfg.Eps)
				}
				query = qNormed
			}
		}
		if layerIdx < len(w.AttnKNorm) && len(w.AttnKNorm[layerIdx]) > 0 && attnHeadDim > 0 {
			normWeight := w.AttnKNorm[layerIdx]
			kvHeads := cfg.KVHeads
			if kvHeads <= 0 {
				kvHeads = cfg.Heads
			}
			if len(normWeight) == attnHeadDim && len(k) == kvHeads*attnHeadDim {
				kNormed := make([]float32, len(k))
				for h := 0; h < kvHeads; h++ {
					off := h * attnHeadDim
					simd.RMSNorm(k[off:off+attnHeadDim], normWeight, kNormed[off:off+attnHeadDim], 1, attnHeadDim, cfg.Eps)
				}
				k = kNormed
			}
		}

		// Apply partial RoPE (only first rotaryDim of each head's headDim dims)
		ropeTheta := cfg.RopeTheta
		if ropeTheta <= 0 {
			ropeTheta = 10000.0
		}
		rotaryDim := attnHeadDim / 4 // partial_rotary_factor=0.25
		if rotaryDim > 0 && attnHeads > 0 && len(query) >= attnHeads*attnHeadDim {
			partialRoPECPU(query, []int{pos}, attnHeads, attnHeadDim, rotaryDim, ropeTheta)
		}
		kvHeads := cfg.KVHeads
		if kvHeads <= 0 {
			kvHeads = cfg.Heads
		}
		if rotaryDim > 0 && kvHeads > 0 && len(k) >= kvHeads*attnHeadDim {
			partialRoPECPU(k, []int{pos}, kvHeads, attnHeadDim, rotaryDim, ropeTheta)
		}

		var attn []float32
		if kv != nil {
			attn = attentionCPUKV(query, k, v, layerIdx, pos, kv, attnHeads, kvHeads, attnHeadDim)
		} else {
			attn = attentionCPU(query, k, v, attnHeads, kvHeads, attnHeadDim)
		}
		// Apply output gate: attn_out * sigmoid(gate)
		if len(attn) == len(gate) {
			for i := range attn {
				attn[i] *= sigmoidCPU(gate[i])
			}
		}
		out = w.MatVec(oWeight, rawO, attn)
	} else if w.HasSSM(layerIdx) {
		var qkvWeight []float32
		var rawQKV *gguf.TensorInfo
		if layerIdx < len(w.AttnQKV) {
			qkvWeight = w.AttnQKV[layerIdx]
		}
		if layerIdx < len(w.RawAttnQKV) {
			rawQKV = w.RawAttnQKV[layerIdx]
		}

		qkv := w.MatVec(qkvWeight, rawQKV, normed)
		if len(qkv) == 0 {
			out = normed
		} else {
			out = w.mambaForward(qkv, normed, layerIdx, kv, cfg)
		}
	}

	residual := make([]float32, len(input))
	if len(out) == len(input) {
		for i := range residual {
			residual[i] = input[i] + out[i]
		}
	} else {
		copy(residual, input)
	}

	// FFN Sublayer
	if w.HasFFN(layerIdx) {
		normedFFN := make([]float32, len(residual))
		simd.RMSNorm(residual, w.FfnNorm[layerIdx], normedFFN, 1, len(residual), cfg.Eps)

		var gateWeight, upWeight, downWeight []float32
		var rawGate, rawUp, rawDown *gguf.TensorInfo
		if layerIdx < len(w.FfnGate) {
			gateWeight = w.FfnGate[layerIdx]
		}
		if layerIdx < len(w.FfnUp) {
			upWeight = w.FfnUp[layerIdx]
		}
		if layerIdx < len(w.FfnDown) {
			downWeight = w.FfnDown[layerIdx]
		}
		if layerIdx < len(w.RawFfnGate) {
			rawGate = w.RawFfnGate[layerIdx]
		}
		if layerIdx < len(w.RawFfnUp) {
			rawUp = w.RawFfnUp[layerIdx]
		}
		if layerIdx < len(w.RawFfnDown) {
			rawDown = w.RawFfnDown[layerIdx]
		}

		gate := w.MatVec(gateWeight, rawGate, normedFFN)
		up := w.MatVec(upWeight, rawUp, normedFFN)

		swiGLU := make([]float32, len(gate))
		simd.SwiGLU(gate, up, swiGLU)

		down := w.MatVec(downWeight, rawDown, swiGLU)

		result := make([]float32, len(residual))
		for i := range result {
			result[i] = residual[i] + down[i]
		}
		return result
	}

	return residual
}

func vecDot(a, b []float32) float32 {
	return simd.VecDotF32(a, b)
}

func vecFMA(dst, src []float32, weight float32) {
	simd.VecFMAF32(dst, src, weight)
}

func attentionCPUKV(q, k, v []float32, layerIdx, pos int, kv *CPUKVCache, numHeads, kvHeads, headDim int) []float32 {
	if numHeads <= 0 || headDim <= 0 || len(q) == 0 {
		return make([]float32, len(q))
	}
	if kvHeads <= 0 {
		kvHeads = numHeads
	}

	kvDim := kvHeads * headDim
	if len(k) < kvDim || len(v) < kvDim {
		return make([]float32, len(q))
	}

	kv.mu.Lock()
	if layerIdx >= len(kv.Keys) {
		newKeys := make([][]float32, layerIdx+1)
		newVals := make([][]float32, layerIdx+1)
		copy(newKeys, kv.Keys)
		copy(newVals, kv.Values)
		kv.Keys = newKeys
		kv.Values = newVals
	}
	kv.Keys[layerIdx] = append(kv.Keys[layerIdx], k[:kvDim]...)
	kv.Values[layerIdx] = append(kv.Values[layerIdx], v[:kvDim]...)

	// Sliding-window eviction with attention sink:
	// If MaxLen > 0 and we exceed the window, evict the oldest token
	// while always preserving position 0 (the attention sink).
	if kv.MaxLen > 0 {
		numSlots := len(kv.Keys[layerIdx]) / kvDim
		for numSlots > kv.MaxLen {
			// Preserve slot 0 (attention sink); evict slot 1
			if numSlots <= 1 {
				break
			}
			// Remove element at index 1 (keep [0] and [2:])
			keys := kv.Keys[layerIdx]
			vals := kv.Values[layerIdx]
			copy(keys[kvDim:], keys[kvDim*2:])
			kv.Keys[layerIdx] = keys[:len(keys)-kvDim]
			copy(vals[kvDim:], vals[kvDim*2:])
			kv.Values[layerIdx] = vals[:len(vals)-kvDim]
			numSlots--
		}
	}

	cachedK := kv.Keys[layerIdx]
	cachedV := kv.Values[layerIdx]
	kv.mu.Unlock()

	numCached := len(cachedK) / kvDim
	if numCached <= 0 {
		return make([]float32, len(q))
	}

	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	result := make([]float32, numHeads*headDim)
	scores := make([]float32, numCached)

	kvPerHead := numHeads / kvHeads
	if kvPerHead < 1 {
		kvPerHead = 1
	}

	for h := 0; h < numHeads; h++ {
		qHead := q[h*headDim : (h+1)*headDim]
		kh := h / kvPerHead
		if kh >= kvHeads {
			kh = kvHeads - 1
		}

		var maxScore float32 = -math.MaxFloat32
		for j := 0; j < numCached; j++ {
			kOffset := j*kvDim + kh*headDim
			kVec := cachedK[kOffset : kOffset+headDim]
			s := simd.VecDotF32(qHead, kVec) * scale
			scores[j] = s
			if s > maxScore {
				maxScore = s
			}
		}

		var sumExp float32
		for j := 0; j < numCached; j++ {
			w := float32(math.Exp(float64(scores[j] - maxScore)))
			scores[j] = w
			sumExp += w
		}

		invSum := float32(1.0)
		if sumExp > 0 {
			invSum = 1.0 / sumExp
		}

		outHead := result[h*headDim : (h+1)*headDim]
		for j := 0; j < numCached; j++ {
			w := scores[j] * invSum
			vOffset := j*kvDim + kh*headDim
			vVec := cachedV[vOffset : vOffset+headDim]
			simd.VecFMAF32(outHead, vVec, w)
		}
	}

	return result
}

func attentionCPU(q, k, v []float32, numHeads, kvHeads, headDim int) []float32 {
	if numHeads <= 0 || headDim <= 0 || len(q) == 0 {
		return make([]float32, len(q))
	}
	seqLen := len(q) / (numHeads * headDim)
	if seqLen <= 0 {
		return make([]float32, len(q))
	}
	if kvHeads <= 0 {
		kvHeads = numHeads
	}

	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	result := make([]float32, len(q))
	kvStride := headDim * seqLen
	totalKHeads := len(k) / kvStride
	if totalKHeads <= 0 {
		totalKHeads = 1
	}

	scores := make([]float32, seqLen)

	for h := 0; h < numHeads; h++ {
		qHead := q[h*kvStride : (h+1)*kvStride]

		var kh int
		if totalKHeads == numHeads {
			kh = h
		} else {
			kvPerHead := numHeads / kvHeads
			if kvPerHead < 1 {
				kvPerHead = 1
			}
			kh = h / kvPerHead
			if kh >= totalKHeads {
				kh = totalKHeads - 1
			}
		}

		kHead := k[kh*kvStride : (kh+1)*kvStride]
		vHead := v[kh*kvStride : (kh+1)*kvStride]

		for i := 0; i < seqLen; i++ {
			qVec := qHead[i*headDim : (i+1)*headDim]

			var maxScore float32 = -math.MaxFloat32
			for j := 0; j <= i; j++ {
				kVec := kHead[j*headDim : (j+1)*headDim]
				score := vecDot(qVec, kVec) * scale
				scores[j] = score
				if score > maxScore {
					maxScore = score
				}
			}

			var sumExp float32
			for j := 0; j <= i; j++ {
				w := float32(math.Exp(float64(scores[j] - maxScore)))
				scores[j] = w
				sumExp += w
			}

			invSum := float32(1.0)
			if sumExp > 0 {
				invSum = 1.0 / sumExp
			}

			outVec := result[h*kvStride+i*headDim : h*kvStride+(i+1)*headDim]
			for j := 0; j <= i; j++ {
				w := scores[j] * invSum
				vVec := vHead[j*headDim : (j+1)*headDim]
				vecFMA(outVec, vVec, w)
			}
		}
	}

	return result
}

func (w *CPUWeights) Free() {
	w.TokenEmb = nil
	w.Output = nil
	w.OutputNorm = nil
	w.AttnQ = nil
	w.AttnK = nil
	w.AttnV = nil
	w.AttnO = nil
	w.AttnNorm = nil
	w.FfnGate = nil
	w.FfnDown = nil
	w.FfnUp = nil
	w.FfnNorm = nil
	w.SSMConv1d = nil
	w.SSMConv1dBias = nil
	w.SSMA = nil
	w.SSMD = nil
	w.SSMAlpha = nil
	w.SSMBeta = nil
	w.SSMDtWeight = nil
	w.SSMDtBias = nil
	w.SSMNorm = nil
	w.SSMOut = nil
	w.RawTokenEmb = nil
	w.RawOutput = nil
	w.RawAttnQ = nil
	w.RawAttnK = nil
	w.RawAttnV = nil
	w.RawAttnO = nil
	w.RawFfnGate = nil
	w.RawFfnDown = nil
	w.RawFfnUp = nil
	w.RawAttnQKV = nil
	w.RawAttnGate = nil
	w.RawSSMOut = nil
}

func (w *CPUWeights) MatVec(f32Weight []float32, raw *gguf.TensorInfo, x []float32) []float32 {
	if len(f32Weight) > 0 {
		cols := len(x)
		if cols == 0 {
			return nil
		}
		return simd.MatVecMul(f32Weight, x, len(f32Weight)/cols, cols)
	}
	if raw != nil {
		cols := len(x)
		if cols == 0 {
			return nil
		}
		rows := int(raw.Dimensions[0]) // #nosec G115 -- safe: tensor dimensions fit in int
		if len(raw.Dimensions) > 1 {
			rows = int(raw.Dimensions[1]) // #nosec G115 -- safe: tensor dimensions fit in int
		}
		switch raw.Type {
		case gguf.GGMLTypeQ8_0:
			return gguf.MatVecMulQ8_0(raw.Data, x, rows, cols)
		case gguf.GGMLTypeQ4_K:
			return gguf.MatVecMulQ4_K(raw.Data, x, rows, cols)
		case gguf.GGMLTypeQ6_K:
			return gguf.MatVecMulQ6_K(raw.Data, x, rows, cols)
		case gguf.GGMLTypeBF16:
			return gguf.MatVecMulBF16(raw.Data, x, rows, cols)
		case gguf.GGMLTypeQ4_0:
			return gguf.MatVecMulQ4_0(raw.Data, x, rows, cols)
		case gguf.GGMLTypeQ5_0:
			return gguf.MatVecMulQ5_0(raw.Data, x, rows, cols)
		case gguf.GGMLTypeQ2_K:
			return gguf.MatVecMulQ2_K(raw.Data, x, rows, cols)
		case gguf.GGMLTypeQ3_K:
			return gguf.MatVecMulQ3_K(raw.Data, x, rows, cols)
		case gguf.GGMLTypeQ5_K:
			return gguf.MatVecMulQ5_K(raw.Data, x, rows, cols)
		default:
			// Fallback: dequantize then matvecmul for F16, BF16, F32, and any unrecognized type
			data, err := decodeTensorData(raw)
			if err == nil && len(data) > 0 {
				return simd.MatVecMul(data, x, len(data)/cols, cols)
			}
		}

	}
	return nil
}

func (w *CPUWeights) GetTokenEmbedding(tokenId int, hiddenSize int) []float32 {
	if len(w.TokenEmb) > 0 && len(w.TokenEmb[0]) > 0 {
		vocabSize := len(w.TokenEmb[0]) / hiddenSize
		if tokenId >= vocabSize {
			tokenId = 0
		}
		startIdx := tokenId * hiddenSize
		if startIdx+hiddenSize <= len(w.TokenEmb[0]) {
			out := make([]float32, hiddenSize)
			copy(out, w.TokenEmb[0][startIdx:startIdx+hiddenSize])
			return out
		}
	}
	if w.RawTokenEmb != nil {
		out := make([]float32, hiddenSize)
		switch w.RawTokenEmb.Type {
		case gguf.GGMLTypeQ8_0:
			const blockSize = 32
			const blockSizeBytes = 34
			blocksPerRow := hiddenSize / blockSize
			rowBytes := blocksPerRow * blockSizeBytes
			offset := tokenId * rowBytes
			if offset+rowBytes <= len(w.RawTokenEmb.Data) {
				return gguf.DequantizeQ8_0(w.RawTokenEmb.Data[offset:offset+rowBytes], hiddenSize)
			}
		case gguf.GGMLTypeF32:
			offset := tokenId * hiddenSize * 4
			for i := 0; i < hiddenSize && offset+(i+1)*4 <= len(w.RawTokenEmb.Data); i++ {
				bits := binary.LittleEndian.Uint32(w.RawTokenEmb.Data[offset+i*4:])
				out[i] = math.Float32frombits(bits)
			}
			return out
		}
	}
	return make([]float32, hiddenSize)
}

func (w *CPUWeights) HasFullAttn(layerIdx int) bool {
	if layerIdx < 0 {
		return false
	}
	hasQ := (layerIdx < len(w.AttnQ) && len(w.AttnQ[layerIdx]) > 0) || (layerIdx < len(w.RawAttnQ) && w.RawAttnQ[layerIdx] != nil)
	hasK := (layerIdx < len(w.AttnK) && len(w.AttnK[layerIdx]) > 0) || (layerIdx < len(w.RawAttnK) && w.RawAttnK[layerIdx] != nil)
	hasV := (layerIdx < len(w.AttnV) && len(w.AttnV[layerIdx]) > 0) || (layerIdx < len(w.RawAttnV) && w.RawAttnV[layerIdx] != nil)
	hasO := (layerIdx < len(w.AttnO) && len(w.AttnO[layerIdx]) > 0) || (layerIdx < len(w.RawAttnO) && w.RawAttnO[layerIdx] != nil)
	return hasQ && hasK && hasV && hasO
}

func (w *CPUWeights) HasSSM(layerIdx int) bool {
	if layerIdx < 0 {
		return false
	}
	hasQKV := (layerIdx < len(w.AttnQKV) && len(w.AttnQKV[layerIdx]) > 0) || (layerIdx < len(w.RawAttnQKV) && w.RawAttnQKV[layerIdx] != nil)
	hasOut := (layerIdx < len(w.SSMOut) && len(w.SSMOut[layerIdx]) > 0) || (layerIdx < len(w.RawSSMOut) && w.RawSSMOut[layerIdx] != nil)
	return hasQKV && hasOut
}

func (w *CPUWeights) HasFFN(layerIdx int) bool {
	if layerIdx < 0 {
		return false
	}
	hasNorm := layerIdx < len(w.FfnNorm) && len(w.FfnNorm[layerIdx]) > 0
	hasGate := (layerIdx < len(w.FfnGate) && len(w.FfnGate[layerIdx]) > 0) || (layerIdx < len(w.RawFfnGate) && w.RawFfnGate[layerIdx] != nil)
	hasUp := (layerIdx < len(w.FfnUp) && len(w.FfnUp[layerIdx]) > 0) || (layerIdx < len(w.RawFfnUp) && w.RawFfnUp[layerIdx] != nil)
	hasDown := (layerIdx < len(w.FfnDown) && len(w.FfnDown[layerIdx]) > 0) || (layerIdx < len(w.RawFfnDown) && w.RawFfnDown[layerIdx] != nil)
	return hasNorm && hasGate && hasUp && hasDown
}

// mambaForward dispatches to the appropriate SSM forward pass based on architecture.
func (w *CPUWeights) mambaForward(qkv []float32, normed []float32, layerIdx int, kv *CPUKVCache, cfg config.Config) []float32 {
	hasAlpha := layerIdx < len(w.SSMAlpha) && len(w.SSMAlpha[layerIdx]) > 0
	hasBeta := layerIdx < len(w.SSMBeta) && len(w.SSMBeta[layerIdx]) > 0
	if hasAlpha && hasBeta {
		return w.gatedDeltaNetForward(qkv, normed, layerIdx, kv, cfg)
	}
	return w.mamba2Forward(qkv, layerIdx, kv, cfg)
}

// mamba2Forward executes a standard Mamba-2 SSM layer.
func (w *CPUWeights) mamba2Forward(qkv []float32, layerIdx int, kv *CPUKVCache, cfg config.Config) []float32 {
	dInner := 0
	if layerIdx < len(w.SSMA) {
		dInner = len(w.SSMA[layerIdx])
	}
	if dInner == 0 {
		return qkv
	}
	totalProj := len(qkv)
	dState := (totalProj - 2*dInner) / 2
	if dState <= 0 {
		dState = 1
	}
	xzSize := 2 * dInner
	xz := qkv[:xzSize]
	var B, C []float32
	if xzSize+dState <= totalProj {
		B = qkv[xzSize : xzSize+dState]
	}
	if xzSize+2*dState <= totalProj {
		C = qkv[xzSize+dState : xzSize+2*dState]
	}
	dConv := 4
	if layerIdx < len(w.SSMConv1d) {
		convWeight := w.SSMConv1d[layerIdx]
		if xzSize > 0 {
			dConv = len(convWeight) / xzSize
		}
	}
	if kv != nil {
		kv.mu.Lock()
		if kv.ConvState == nil {
			kv.ConvState = make([][]float32, layerIdx+1)
		}
		if layerIdx >= len(kv.ConvState) {
			newState := make([][]float32, layerIdx+1)
			copy(newState, kv.ConvState)
			kv.ConvState = newState
		}
		if kv.ConvState[layerIdx] == nil || len(kv.ConvState[layerIdx]) != dConv*xzSize {
			kv.ConvState[layerIdx] = make([]float32, dConv*xzSize)
		}
		kv.mu.Unlock()
	}
	xzConv := make([]float32, xzSize)
	if kv != nil {
		convState := kv.ConvState[layerIdx]
		if dConv > 1 {
			copy(convState[0:], convState[xzSize:])
		}
		copy(convState[(dConv-1)*xzSize:], xz)
		if layerIdx < len(w.SSMConv1d) {
			convWeight := w.SSMConv1d[layerIdx]
			var convBias []float32
			if layerIdx < len(w.SSMConv1dBias) {
				convBias = w.SSMConv1dBias[layerIdx]
			}
			for i := 0; i < xzSize; i++ {
				var sum float32
				if convBias != nil && i < len(convBias) {
					sum = convBias[i]
				}
				for k := 0; k < dConv; k++ {
					wIdx := k*xzSize + i
					sIdx := k*xzSize + i
					if wIdx < len(convWeight) && sIdx < len(convState) {
						sum += convState[sIdx] * convWeight[wIdx]
					}
				}
				xzConv[i] = sum
			}
		} else {
			copy(xzConv, xz)
		}
	} else {
		copy(xzConv, xz)
	}
	x := xzConv[:dInner]
	z := xzConv[dInner:]
	zAct := make([]float32, len(z))
	for i, v := range z {
		zAct[i] = siluCPU(v)
	}
	dt := make([]float32, dInner)
	if layerIdx < len(w.SSMDtWeight) && len(w.SSMDtWeight[layerIdx]) > 0 {
		dtWeight := w.SSMDtWeight[layerIdx]
		cols := len(qkv) / (len(dtWeight) / dInner)
		if cols > 0 {
			dtResult := simd.MatVecMul(dtWeight, qkv, dInner, cols)
			for i := range dt {
				if i < len(dtResult) {
					dt[i] = dtResult[i]
				}
			}
		}
	}
	if layerIdx < len(w.SSMDtBias) && len(w.SSMDtBias[layerIdx]) > 0 {
		for i := range dt {
			if i < len(w.SSMDtBias[layerIdx]) {
				dt[i] += w.SSMDtBias[layerIdx][i]
			}
		}
	}
	for i := range dt {
		dt[i] = softplusCPU(dt[i])
	}
	if kv != nil {
		kv.mu.Lock()
		if kv.SSMState == nil {
			kv.SSMState = make([][]float32, layerIdx+1)
		}
		if layerIdx >= len(kv.SSMState) {
			newState := make([][]float32, layerIdx+1)
			copy(newState, kv.SSMState)
			kv.SSMState = newState
		}
		if kv.SSMState[layerIdx] == nil || len(kv.SSMState[layerIdx]) != dInner*dState {
			kv.SSMState[layerIdx] = make([]float32, dInner*dState)
		}
		kv.mu.Unlock()
	}
	out := make([]float32, dInner)
	if kv != nil {
		ssmState := kv.SSMState[layerIdx]
		A := w.SSMA[layerIdx]
		var D []float32
		if layerIdx < len(w.SSMD) {
			D = w.SSMD[layerIdx]
		}
		for i := 0; i < dInner; i++ {
			aVal := float32(0)
			if i < len(A) {
				aVal = A[i]
			}
			expADt := float32(math.Exp(float64(aVal * dt[i])))
			xDt := x[i] * dt[i]
			for j := 0; j < dState; j++ {
				idx := i*dState + j
				bVal := float32(0)
				if j < len(B) {
					bVal = B[j]
				}
				ssmState[idx] = expADt*ssmState[idx] + bVal*xDt
			}
			var y float32
			for j := 0; j < dState; j++ {
				cVal := float32(0)
				if j < len(C) {
					cVal = C[j]
				}
				y += cVal * ssmState[i*dState+j]
			}
			dVal := float32(0)
			if D != nil && i < len(D) {
				dVal = D[i]
			}
			out[i] = y + dVal*x[i]
		}
	} else {
		copy(out, x)
	}
	for i := range out {
		if i < len(zAct) {
			out[i] *= zAct[i]
		}
	}
	var ssmOutWeight []float32
	var rawSSMOut *gguf.TensorInfo
	if layerIdx < len(w.SSMOut) {
		ssmOutWeight = w.SSMOut[layerIdx]
	}
	if layerIdx < len(w.RawSSMOut) {
		rawSSMOut = w.RawSSMOut[layerIdx]
	}
	if ssmOutWeight != nil || rawSSMOut != nil {
		return w.MatVec(ssmOutWeight, rawSSMOut, out)
	}
	return out
}

// gatedDeltaNetForward implements the GatedDeltaNet (Qwen 3.5) SSM layer.
//
// Architecture:
//   - qkv projection [8192] → conv1d → SiLU → split into Q[2048], K[2048], V[4096]
//   - Gate projection [4096] → z
//   - Alpha projection [32] → decay gate g = -exp(A) * softplus(alpha + dt_bias)
//   - Beta projection [32] → write strength beta = sigmoid(beta_raw)
//   - Delta rule scan: S = g*S + beta*(v - S@k)⊗k^T; y = S@q
//   - Output: RMSNorm(y) * SiLU(z) → ssm_out
func (w *CPUWeights) gatedDeltaNetForward(qkv []float32, normed []float32, layerIdx int, kv *CPUKVCache, cfg config.Config) []float32 {
	totalProj := len(qkv)
	if totalProj == 0 {
		return qkv
	}

	// Infer dimensions from tensor shapes
	dInner := totalProj / 2
	if dInner == 0 {
		dInner = 4096
	}

	numVHeads := 0
	if layerIdx < len(w.SSMA) {
		numVHeads = len(w.SSMA[layerIdx])
	}
	if numVHeads == 0 {
		numVHeads = 32
	}
	headDim := dInner / numVHeads
	if headDim == 0 {
		headDim = 128
	}

	// numKHeads: Q and K are interleaved from numKHeads to numVHeads.
	// The qk portion of conv output is 2 * numKHeads * headKDim.
	// headKDim = headDim for this model (both 128).
	// qkSize = 2 * numKHeads * headDim, vSize = numVHeads * headDim
	// totalProj = qkSize + vSize = 2 * numKHeads * headDim + numVHeads * headDim
	numKHeads := (totalProj - dInner) / (2 * headDim)
	if numKHeads <= 0 {
		numKHeads = numVHeads / 2
		if numKHeads == 0 {
			numKHeads = 16
		}
	}
	headKDim := headDim
	qSize := numKHeads * headKDim
	kSize := numKHeads * headKDim
	vSize := numVHeads * headDim
	convDim := qSize + kSize + vSize

	dConv := 4
	if layerIdx < len(w.SSMConv1d) && len(w.SSMConv1d[layerIdx]) > 0 {
		convWeight := w.SSMConv1d[layerIdx]
		if convDim > 0 {
			dConv = len(convWeight) / convDim
		}
	}
	if dConv <= 0 {
		dConv = 4
	}

	// Lazy-init conv state
	if kv != nil {
		kv.mu.Lock()
		if kv.ConvState == nil {
			kv.ConvState = make([][]float32, layerIdx+1)
		}
		if layerIdx >= len(kv.ConvState) {
			newState := make([][]float32, layerIdx+1)
			copy(newState, kv.ConvState)
			kv.ConvState = newState
		}
		if kv.ConvState[layerIdx] == nil || len(kv.ConvState[layerIdx]) != dConv*convDim {
			kv.ConvState[layerIdx] = make([]float32, dConv*convDim)
		}
		kv.mu.Unlock()
	}

	// 1. Causal conv1d on qkv (Metal convention: state[0] = newest)
	convOut := make([]float32, convDim)
	if kv != nil {
		convState := kv.ConvState[layerIdx]

		// Shift state right: state[k] ← state[k-1] for k=dConv-1..1 (matching Metal kernel)
		if dConv > 1 {
			for k := dConv - 1; k > 0; k-- {
				copy(convState[k*convDim:(k+1)*convDim], convState[(k-1)*convDim:k*convDim])
			}
		}
		// Insert newest input at position 0
		copy(convState[:convDim], qkv)

		if layerIdx < len(w.SSMConv1d) {
			convWeight := w.SSMConv1d[layerIdx]
			var convBias []float32
			if layerIdx < len(w.SSMConv1dBias) {
				convBias = w.SSMConv1dBias[layerIdx]
			}
			for i := 0; i < convDim; i++ {
				var sum float32
				if convBias != nil && i < len(convBias) {
					sum = convBias[i]
				}
				for k := 0; k < dConv; k++ {
					wIdx := i*dConv + k
					sIdx := k*convDim + i
					if wIdx < len(convWeight) && sIdx < len(convState) {
						sum += convState[sIdx] * convWeight[wIdx]
					}
				}
				convOut[i] = sum
			}
		} else {
			copy(convOut, qkv)
		}
	} else {
		copy(convOut, qkv)
	}

	// 2. SiLU activation on conv output
	for i := range convOut {
		convOut[i] = siluCPU(convOut[i])
	}

	// 3. Split into Q, K, V
	qRaw := convOut[:qSize]
	kRaw := convOut[qSize : qSize+kSize]
	vRaw := convOut[qSize+kSize:]

	// 4. Compute alpha (decay) and beta (write strength)
	alphaRaw := make([]float32, numVHeads)
	betaRaw := make([]float32, numVHeads)
	if layerIdx < len(w.SSMAlpha) && len(w.SSMAlpha[layerIdx]) > 0 {
		alphaResult := w.MatVec(w.SSMAlpha[layerIdx], nil, normed)
		for i := 0; i < numVHeads && i < len(alphaResult); i++ {
			alphaRaw[i] = alphaResult[i]
		}
	}
	if layerIdx < len(w.SSMBeta) && len(w.SSMBeta[layerIdx]) > 0 {
		betaResult := w.MatVec(w.SSMBeta[layerIdx], nil, normed)
		for i := 0; i < numVHeads && i < len(betaResult); i++ {
			betaRaw[i] = betaResult[i]
		}
	}

	A := w.SSMA[layerIdx]
	dtBias := make([]float32, numVHeads)
	if layerIdx < len(w.SSMDtBias) && len(w.SSMDtBias[layerIdx]) > 0 {
		for i := 0; i < numVHeads && i < len(w.SSMDtBias[layerIdx]); i++ {
			dtBias[i] = w.SSMDtBias[layerIdx][i]
		}
	}

	gates := make([]float32, numVHeads)
	betas := make([]float32, numVHeads)
	for h := 0; h < numVHeads; h++ {
		aLog := float32(0)
		if h < len(A) {
			aLog = A[h]
		}
		gates[h] = -float32(math.Exp(float64(aLog))) * softplusCPU(alphaRaw[h]+dtBias[h])
		betas[h] = sigmoidCPU(betaRaw[h])
	}

	// 5. Repeat-interleave Q, K from numKHeads to numVHeads
	repeatFactor := numVHeads / numKHeads
	if repeatFactor < 1 {
		repeatFactor = 1
	}
	q := make([]float32, dInner)
	k := make([]float32, dInner)
	for h := 0; h < numVHeads; h++ {
		srcHead := h / repeatFactor
		srcOff := srcHead * headKDim
		dstOff := h * headDim
		for d := 0; d < headDim && d < headKDim; d++ {
			if srcOff+d < len(qRaw) {
				q[dstOff+d] = qRaw[srcOff+d]
			}
			if srcOff+d < len(kRaw) {
				k[dstOff+d] = kRaw[srcOff+d]
			}
		}
	}

	// 6. L2-normalize Q and K per head
	for h := 0; h < numVHeads; h++ {
		off := h * headDim
		qNorm := float32(0)
		kNorm := float32(0)
		for d := 0; d < headDim; d++ {
			qNorm += q[off+d] * q[off+d]
			kNorm += k[off+d] * k[off+d]
		}
		qNorm = float32(math.Sqrt(float64(qNorm) + 1e-6))
		kNorm = float32(math.Sqrt(float64(kNorm) + 1e-6))
		for d := 0; d < headDim; d++ {
			q[off+d] /= qNorm
			k[off+d] /= kNorm
		}
	}

	// 7. Lazy-init SSM state: [numVHeads * headDim * headKDim]
	stateSize := numVHeads * headDim * headKDim
	if kv != nil {
		kv.mu.Lock()
		if kv.SSMState == nil {
			kv.SSMState = make([][]float32, layerIdx+1)
		}
		if layerIdx >= len(kv.SSMState) {
			newState := make([][]float32, layerIdx+1)
			copy(newState, kv.SSMState)
			kv.SSMState = newState
		}
		if kv.SSMState[layerIdx] == nil || len(kv.SSMState[layerIdx]) != stateSize {
			kv.SSMState[layerIdx] = make([]float32, stateSize)
		}
		kv.mu.Unlock()
	}

	// 8. Gated delta rule scan (single-step inference)
	// State layout: [numVHeads, headDim, headKDim] — S[i,j] for i in [headDim), j in [headKDim)
	// Update: S = g*S + beta*(v - S@k)⊗k^T
	// Readout: y = S @ q
	y := make([]float32, dInner)
	if kv != nil {
		ssmState := kv.SSMState[layerIdx]
		for h := 0; h < numVHeads; h++ {
			g := gates[h]
			beta := betas[h]
			qOff := h * headDim
			kOff := h * headKDim
			vOff := h * headDim
			stateOff := h * headDim * headKDim

			// S_h @ k_h: [headDim, headKDim] @ [headKDim] = [headDim]
			sk := make([]float32, headDim)
			for i := 0; i < headDim; i++ {
				var sum float32
				for j := 0; j < headKDim; j++ {
					sum += ssmState[stateOff+i*headKDim+j] * k[kOff+j]
				}
				sk[i] = sum
			}

			// err = v_h - S_h @ k_h
			err := make([]float32, headDim)
			for i := 0; i < headDim; i++ {
				vi := float32(0)
				if vOff+i < len(vRaw) {
					vi = vRaw[vOff+i]
				}
				err[i] = vi - sk[i]
			}

			// State update: S = g*S + beta * err ⊗ k^T
			for i := 0; i < headDim; i++ {
				for j := 0; j < headKDim; j++ {
					stateIdx := stateOff + i*headKDim + j
					ssmState[stateIdx] = g*ssmState[stateIdx] + beta*err[i]*k[kOff+j]
				}
			}

			// Readout: y_h = S_h @ q_h
			for i := 0; i < headDim; i++ {
				var sum float32
				for j := 0; j < headKDim; j++ {
					sum += ssmState[stateOff+i*headKDim+j] * q[qOff+j]
				}
				y[qOff+i] = sum
			}
		}
	} else {
		copy(y, vRaw)
	}

	// 9. Compute z gate from attn_gate projection and apply RMSNorm * SiLU gating
	var z []float32
	if layerIdx < len(w.AttnGate) || layerIdx < len(w.RawAttnGate) {
		var gateWeight []float32
		var rawGate *gguf.TensorInfo
		if layerIdx < len(w.AttnGate) {
			gateWeight = w.AttnGate[layerIdx]
		}
		if layerIdx < len(w.RawAttnGate) {
			rawGate = w.RawAttnGate[layerIdx]
		}
		z = w.MatVec(gateWeight, rawGate, normed)
	}
	if z == nil {
		z = make([]float32, dInner)
	}

	// SiLU(z)
	zAct := make([]float32, len(z))
	for i, v := range z {
		zAct[i] = siluCPU(v)
	}

	// RMSNorm per-head then gate with SiLU(z)
	ssmNormWeight := make([]float32, headDim)
	if layerIdx < len(w.SSMNorm) && len(w.SSMNorm[layerIdx]) == headDim {
		copy(ssmNormWeight, w.SSMNorm[layerIdx])
	} else if layerIdx < len(w.SSMNorm) && len(w.SSMNorm[layerIdx]) > 0 {
		ssmNormWeight = w.SSMNorm[layerIdx]
	}

	for h := 0; h < numVHeads; h++ {
		off := h * headDim
		normSize := headDim
		if off+normSize > len(y) {
			normSize = len(y) - off
		}
		if normSize <= 0 {
			continue
		}
		rms := float32(0)
		for d := 0; d < normSize; d++ {
			rms += y[off+d] * y[off+d]
		}
		rms = float32(math.Sqrt(float64(rms)/float64(normSize) + 1e-6))
		for d := 0; d < normSize; d++ {
			wVal := float32(1)
			if d < len(ssmNormWeight) {
				wVal = ssmNormWeight[d]
			}
			yNorm := (y[off+d] / rms) * wVal
			if off+d < len(zAct) {
				y[off+d] = yNorm * zAct[off+d]
			} else {
				y[off+d] = yNorm
			}
		}
	}

	// 10. Output projection
	var ssmOutWeight []float32
	var rawSSMOut *gguf.TensorInfo
	if layerIdx < len(w.SSMOut) {
		ssmOutWeight = w.SSMOut[layerIdx]
	}
	if layerIdx < len(w.RawSSMOut) {
		rawSSMOut = w.RawSSMOut[layerIdx]
	}
	if ssmOutWeight != nil || rawSSMOut != nil {
		return w.MatVec(ssmOutWeight, rawSSMOut, y)
	}
	return y
}

func siluCPU(x float32) float32 {
	if x < -20 {
		return 0
	}
	if x > 20 {
		return x
	}
	return x / (1.0 + float32(math.Exp(-float64(x))))
}

func softplusCPU(x float32) float32 {
	if x > 20 {
		return x
	}
	if x < -20 {
		return 0
	}
	return float32(math.Log(float64(1.0+float32(math.Exp(float64(x))))))
}

func sigmoidCPU(x float32) float32 {
	if x > 20 {
		return 1.0
	}
	if x < -20 {
		return 0.0
	}
	return 1.0 / (1.0 + float32(math.Exp(float64(-x))))
}

func partialRoPECPU(tensor []float32, positions []int, heads, headDim, rotaryDim int, theta float32) {
	half := rotaryDim / 2
	if half <= 0 {
		return
	}
	for h := 0; h < heads; h++ {
		pos := 0
		if len(positions) > 0 {
			pos = positions[0]
		}
		for d := 0; d < half; d++ {
			offset := h*headDim + d
			freq := float32(pos) / float32(math.Pow(float64(theta), float64(2*d)/float64(headDim)))
			cos := float32(math.Cos(float64(freq)))
			sin := float32(math.Sin(float64(freq)))
			ei := offset
			oi := offset + half
			ev := tensor[ei]
			od := tensor[oi]
			tensor[ei] = ev*cos - od*sin
			tensor[oi] = ev*sin + od*cos
		}
	}
}
