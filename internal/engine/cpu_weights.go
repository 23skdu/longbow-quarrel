package engine

import (
	"encoding/binary"
	"fmt"
	"math"
	"strings"
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

	// Gemma4 support
	PostAttentionNorm [][]float32
	PostFfnNorm       [][]float32
	InpGate           [][]float32
	Proj              [][]float32
	PostNorm          [][]float32
	LayerOutputScale  [][]float32

	PerLayerModelProj *gguf.TensorInfo
	PerLayerProjNorm  []float32
	PerLayerTokenEmbd *gguf.TensorInfo
	RawInpGate        []*gguf.TensorInfo
	RawProj           []*gguf.TensorInfo

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

	w.PostAttentionNorm = make([][]float32, numLayers)
	w.PostFfnNorm = make([][]float32, numLayers)
	w.InpGate = make([][]float32, numLayers)
	w.Proj = make([][]float32, numLayers)
	w.PostNorm = make([][]float32, numLayers)
	w.LayerOutputScale = make([][]float32, numLayers)
	w.RawInpGate = make([]*gguf.TensorInfo, numLayers)
	w.RawProj = make([]*gguf.TensorInfo, numLayers)

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
		case "per_layer_model_proj.weight":
			w.PerLayerModelProj = t
		case "per_layer_proj_norm.weight":
			if data, err := decodeTensorData(t); err == nil {
				w.PerLayerProjNorm = data
			}
		case "per_layer_token_embd.weight":
			w.PerLayerTokenEmbd = t
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
				case strings.HasSuffix(t.Name, ".post_attention_norm.weight"):
					if data, err := decodeTensorData(t); err == nil {
						w.PostAttentionNorm[layer] = data
					}
					if len(w.FfnNorm[layer]) == 0 {
						w.FfnNorm[layer] = w.PostAttentionNorm[layer]
					}
				case containsStr(t.Name, "ffn_norm.weight"):
					if data, err := decodeTensorData(t); err == nil {
						w.FfnNorm[layer] = data
					}
				case strings.HasSuffix(t.Name, ".post_ffw_norm.weight"), strings.HasSuffix(t.Name, ".post_ffn_norm.weight"):
					if data, err := decodeTensorData(t); err == nil {
						w.PostFfnNorm[layer] = data
					}
				case strings.HasSuffix(t.Name, ".post_norm.weight"):
					if data, err := decodeTensorData(t); err == nil {
						w.PostNorm[layer] = data
					}
				case strings.HasSuffix(t.Name, ".inp_gate.weight"):
					w.RawInpGate[layer] = t
					if !isQ8Matrix {
						if data, err := decodeTensorData(t); err == nil {
							w.InpGate[layer] = data
						}
					}
				case strings.HasSuffix(t.Name, ".proj.weight"):
					w.RawProj[layer] = t
					if !isQ8Matrix {
						if data, err := decodeTensorData(t); err == nil {
							w.Proj[layer] = data
						}
					}
				case strings.HasSuffix(t.Name, ".layer_output_scale.weight"):
					if data, err := decodeTensorData(t); err == nil {
						w.LayerOutputScale[layer] = data
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

	if len(w.Output) == 0 && w.RawOutput == nil {
		if len(w.TokenEmb) > 0 {
			w.Output = w.TokenEmb[0]
		}
		if w.RawTokenEmb != nil {
			w.RawOutput = w.RawTokenEmb
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
	case gguf.GGMLTypeIQ4_NL:
		return gguf.DequantizeIQ4NL(t.Data, int(numElements)), nil
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
		var query, gate []float32
		if attnHeadDim > 0 && len(qFull) == 2*attnHeads*attnHeadDim {
			// Qwen 3.5 query-gate: interleaved [query_h, gate_h] for each head
			query = make([]float32, attnHeads*attnHeadDim)
			gate = make([]float32, attnHeads*attnHeadDim)
			for h := 0; h < attnHeads; h++ {
				srcQ := h * 2 * attnHeadDim
				srcGate := srcQ + attnHeadDim
				copy(query[h*attnHeadDim:(h+1)*attnHeadDim], qFull[srcQ:srcQ+attnHeadDim])
				copy(gate[h*attnHeadDim:(h+1)*attnHeadDim], qFull[srcGate:srcGate+attnHeadDim])
			}
		} else if attnHeadDim > 0 && len(qFull) > attnHeads*attnHeadDim && len(qFull)%(2*attnHeadDim) == 0 {
			attnHeads = len(qFull) / (2 * attnHeadDim)
			query = make([]float32, attnHeads*attnHeadDim)
			gate = make([]float32, attnHeads*attnHeadDim)
			for h := 0; h < attnHeads; h++ {
				srcQ := h * 2 * attnHeadDim
				srcGate := srcQ + attnHeadDim
				copy(query[h*attnHeadDim:(h+1)*attnHeadDim], qFull[srcQ:srcQ+attnHeadDim])
				copy(gate[h*attnHeadDim:(h+1)*attnHeadDim], qFull[srcGate:srcGate+attnHeadDim])
			}
		} else {
			query = qFull
			gate = nil
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
		rotaryDim := attnHeadDim
		if cfg.Architecture == "qwen35" || (attnHeadDim > 64 && attnHeadDim == 256) {
			rotaryDim = 64
		}
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
		if len(gate) > 0 && len(attn) == len(gate) {
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

		for i := range gate {
			gate[i] = siluCPU(gate[i]) * up[i]
		}
		down := w.MatVec(downWeight, rawDown, gate)

		for i := range residual {
			residual[i] += down[i]
		}
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
	_ = pos
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
		case gguf.GGMLTypeIQ4_NL:
			return gguf.MatVecMulIQ4_NL(raw.Data, x, rows, cols)
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

func dequantizeRow(t *gguf.TensorInfo, rowIdx int, rowElements int) []float32 {
	if t == nil || rowElements <= 0 {
		return make([]float32, rowElements)
	}
	var rowBytes int
	switch t.Type {
	case gguf.GGMLTypeF32:
		rowBytes = rowElements * 4
	case gguf.GGMLTypeF16, gguf.GGMLTypeBF16:
		rowBytes = rowElements * 2
	case gguf.GGMLTypeQ4_0, gguf.GGMLTypeIQ4_NL:
		rowBytes = (rowElements / 32) * 18
	case gguf.GGMLTypeQ5_0:
		rowBytes = (rowElements / 32) * 22
	case gguf.GGMLTypeQ8_0:
		rowBytes = (rowElements / 32) * 34
	case gguf.GGMLTypeQ4_K:
		rowBytes = (rowElements / 256) * 144
	case gguf.GGMLTypeQ5_K:
		rowBytes = (rowElements / 256) * 176
	case gguf.GGMLTypeQ6_K:
		rowBytes = (rowElements / 256) * 210
	case gguf.GGMLTypeQ3_K:
		rowBytes = (rowElements / 256) * 110
	case gguf.GGMLTypeQ2_K:
		rowBytes = (rowElements / 256) * 84
	default:
		return make([]float32, rowElements)
	}

	offset := rowIdx * rowBytes
	if offset < 0 || offset+rowBytes > len(t.Data) {
		return make([]float32, rowElements)
	}
	slice := t.Data[offset : offset+rowBytes]

	switch t.Type {
	case gguf.GGMLTypeQ4_K:
		return gguf.DequantizeQ4K_SIMD(slice, rowElements)
	case gguf.GGMLTypeQ6_K:
		return gguf.DequantizeQ6K_SIMD(slice, rowElements)
	case gguf.GGMLTypeQ5_0:
		return gguf.DequantizeQ5_0(slice, rowElements)
	case gguf.GGMLTypeQ8_0:
		return gguf.DequantizeQ8_0(slice, rowElements)
	case gguf.GGMLTypeQ2_K:
		return gguf.DequantizeQ2K(slice, rowElements)
	case gguf.GGMLTypeQ3_K:
		return gguf.DequantizeQ3K(slice, rowElements)
	case gguf.GGMLTypeQ5_K:
		return gguf.DequantizeQ5K(slice, rowElements)
	case gguf.GGMLTypeQ4_0:
		return gguf.DequantizeQ4_0(slice, rowElements)
	case gguf.GGMLTypeF32:
		out := make([]float32, rowElements)
		for i := 0; i < rowElements; i++ {
			bits := binary.LittleEndian.Uint32(slice[i*4:])
			out[i] = math.Float32frombits(bits)
		}
		return out
	case gguf.GGMLTypeF16:
		out := make([]float32, rowElements)
		for i := 0; i < rowElements; i++ {
			bits := binary.LittleEndian.Uint16(slice[i*2:])
			out[i] = gguf.Float16ToFloat32(bits)
		}
		return out
	case gguf.GGMLTypeBF16:
		return gguf.DequantizeBF16(slice, rowElements)
	case gguf.GGMLTypeIQ4_NL:
		return gguf.DequantizeIQ4NL(slice, rowElements)
	default:
		return make([]float32, rowElements)
	}
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
		return dequantizeRow(w.RawTokenEmb, tokenId, hiddenSize)
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
	_ = cfg
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
	_ = cfg
	totalProj := len(qkv)
	if totalProj == 0 {
		return qkv
	}

	// In GatedDeltaNet (e.g. Qwen 3.5):
	// headDim is determined by SSMNorm weight size (128).
	headDim := 128
	if layerIdx < len(w.SSMNorm) && len(w.SSMNorm[layerIdx]) > 0 {
		headDim = len(w.SSMNorm[layerIdx])
	}
	numVHeads := 0
	if layerIdx < len(w.SSMA) {
		numVHeads = len(w.SSMA[layerIdx])
	}
	if numVHeads == 0 {
		numVHeads = 16
	}
	vSize := numVHeads * headDim
	dInner := vSize

	// totalProj = qSize + kSize + vSize
	// qkSize = totalProj - vSize = 2 * numKHeads * headKDim
	// headKDim == headDim (128)
	headKDim := headDim
	qkSize := totalProj - vSize
	if qkSize < 0 {
		qkSize = 0
	}
	numKHeads := qkSize / (2 * headKDim)
	if numKHeads <= 0 {
		numKHeads = 16
	}
	qSize := numKHeads * headKDim
	kSize := numKHeads * headKDim
	convDim := totalProj

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

	// 1. Causal conv1d on qkv
	convOut := make([]float32, convDim)
	if kv != nil {
		convState := kv.ConvState[layerIdx]

		// Shift state left: state[0..dConv-2] ← state[1..dConv-1]
		if dConv > 1 {
			copy(convState[0:], convState[convDim:])
		}
		// Insert newest input at the last slot
		copy(convState[(dConv-1)*convDim:], qkv)

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
	qRaw := make([]float32, qSize)
	copy(qRaw, convOut[:qSize])
	kRaw := make([]float32, kSize)
	copy(kRaw, convOut[qSize:qSize+kSize])
	vRaw := convOut[qSize+kSize:]

	// L2-normalize Q and K per-head (matching llama.cpp ggml_l2_norm)
	for h := 0; h < numKHeads; h++ {
		qOff := h * headKDim
		var sumQ float32
		for d := 0; d < headKDim; d++ {
			sumQ += qRaw[qOff+d] * qRaw[qOff+d]
		}
		normQ := float32(math.Sqrt(float64(sumQ)))
		scaleQ := float32(1.0) / float32(math.Max(1e-6, float64(normQ)))
		for d := 0; d < headKDim; d++ {
			qRaw[qOff+d] *= scaleQ
		}

		kOff := h * headKDim
		var sumK float32
		for d := 0; d < headKDim; d++ {
			sumK += kRaw[kOff+d] * kRaw[kOff+d]
		}
		normK := float32(math.Sqrt(float64(sumK)))
		scaleK := float32(1.0) / float32(math.Max(1e-6, float64(normK)))
		for d := 0; d < headKDim; d++ {
			kRaw[kOff+d] *= scaleK
		}
	}

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
		decayGate := aLog * softplusCPU(alphaRaw[h]+dtBias[h])
		gates[h] = float32(math.Exp(float64(decayGate)))
		betas[h] = sigmoidCPU(betaRaw[h])
	}

	// 5. Repeat Q, K from numKHeads to numVHeads (matching ggml_repeat: tile modulo)
	q := make([]float32, dInner)
	k := make([]float32, dInner)
	for h := 0; h < numVHeads; h++ {
		srcHead := h % numKHeads
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

	// 6. Lazy-init SSM state: [numVHeads * headDim * headKDim]
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

	// 7. Gated delta rule scan (single-step inference)
	// Matching ggml_compute_forward_gated_delta_net:
	// 1. S_h = g * S_h (decay prior state)
	// 2. sk = S_h @ k_h (retrieval)
	// 3. err = beta * (v_h - sk) (prediction error)
	// 4. S_h = S_h + err ⊗ k_h^T (associative write)
	// 5. y_h = (1 / sqrt(headKDim)) * (S_h @ q_h) (scaled readout)
	y := make([]float32, dInner)
	if kv != nil {
		ssmState := kv.SSMState[layerIdx]
		scale := float32(1.0 / math.Sqrt(float64(headKDim)))
		for h := 0; h < numVHeads; h++ {
			g := gates[h]
			beta := betas[h]
			qOff := h * headDim
			kOff := h * headKDim
			vOff := h * headDim
			stateOff := h * headDim * headKDim

			// 1. Decay state S = g * S
			for i := 0; i < headDim; i++ {
				for j := 0; j < headKDim; j++ {
					stateIdx := stateOff + i*headKDim + j
					ssmState[stateIdx] *= g
				}
			}

			// 2. sk = S @ k
			sk := make([]float32, headDim)
			for i := 0; i < headDim; i++ {
				var sum float32
				for j := 0; j < headKDim; j++ {
					sum += ssmState[stateOff+i*headKDim+j] * k[kOff+j]
				}
				sk[i] = sum
			}

			// 3. err = beta * (v - sk)
			err := make([]float32, headDim)
			for i := 0; i < headDim; i++ {
				vi := float32(0)
				if vOff+i < len(vRaw) {
					vi = vRaw[vOff+i]
				}
				err[i] = beta * (vi - sk[i])
			}

			// 4. Update: S = S + err ⊗ k^T
			for i := 0; i < headDim; i++ {
				for j := 0; j < headKDim; j++ {
					stateIdx := stateOff + i*headKDim + j
					ssmState[stateIdx] += err[i] * k[kOff+j]
				}
			}

			// 5. Readout: y_h = scale * (S_h @ q_h)
			for i := 0; i < headDim; i++ {
				var sum float32
				for j := 0; j < headKDim; j++ {
					sum += ssmState[stateOff+i*headKDim+j] * q[qOff+j]
				}
				y[qOff+i] = sum * scale
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
			freq := float32(pos) / float32(math.Pow(float64(theta), float64(2*d)/float64(rotaryDim)))
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

func geluCPU(x float32) float32 {
	const sqrt2OverPi = float32(0.7978845608)
	const coeff = float32(0.044715)
	return 0.5 * x * (1.0 + float32(math.Tanh(float64(sqrt2OverPi*(x+coeff*x*x*x)))))
}

// ComputeGemma4PLE computes the per-layer embedding slices for each layer of Gemma 4.
// It combines per_layer_token_embd and per_layer_model_proj.
func (w *CPUWeights) ComputeGemma4PLE(tok int, hidden []float32, numLayers int) [][]float32 {
	const pleDim = 256
	ple := make([][]float32, numLayers)
	for i := 0; i < numLayers; i++ {
		ple[i] = make([]float32, pleDim)
	}

	totalPLEElements := numLayers * pleDim // 35 * 256 = 8960

	// 1. Token PLE lookup from w.PerLayerTokenEmbd
	var tokenPLE []float32
	if w.PerLayerTokenEmbd != nil {
		tokenPLE = dequantizeRow(w.PerLayerTokenEmbd, tok, totalPLEElements)
	}
	if len(tokenPLE) < totalPLEElements {
		tokenPLE = make([]float32, totalPLEElements)
	}
	const scaleToken = 16.0
	for j := range tokenPLE {
		tokenPLE[j] *= scaleToken
	}

	// 2. Model projection of hidden state through per_layer_model_proj
	var projPLE []float32
	if w.PerLayerModelProj != nil {
		projPLE = w.MatVec(nil, w.PerLayerModelProj, hidden)
	}
	if len(projPLE) < totalPLEElements {
		projPLE = make([]float32, totalPLEElements)
	}
	scaleProj := float32(1.0 / math.Sqrt(float64(len(hidden))))
	for j := range projPLE {
		projPLE[j] *= scaleProj
	}

	// 3. For each layer i: combine normalized projSlice and tokenSlice, then scale by 1/sqrt(2)
	invSqrt2 := float32(1.0 / math.Sqrt(2.0))
	for i := 0; i < numLayers; i++ {
		sliceStart := i * pleDim
		sliceEnd := sliceStart + pleDim
		projSlice := projPLE[sliceStart:sliceEnd]
		tokenSlice := tokenPLE[sliceStart:sliceEnd]

		normedProj := make([]float32, pleDim)
		if len(w.PerLayerProjNorm) == pleDim {
			simd.RMSNorm(projSlice, w.PerLayerProjNorm, normedProj, 1, pleDim, 1e-6)
		} else {
			copy(normedProj, projSlice)
		}

		for k := 0; k < pleDim; k++ {
			ple[i][k] = (normedProj[k] + tokenSlice[k]) * invSqrt2
		}
	}

	return ple
}

// ApplyGemma4LayerCPU executes a single transformer layer of Gemma 4 on CPU.
func ApplyGemma4LayerCPU(
	w *CPUWeights,
	x []float32,
	layerIdx int,
	pos int,
	kv *CPUKVCache,
	pleSlice []float32,
	cfg config.Config,
) []float32 {
	if layerIdx < 0 || w == nil || len(x) == 0 {
		return x
	}

	// 1. Pre-attention RMSNorm
	normedAttn := make([]float32, len(x))
	if layerIdx < len(w.AttnNorm) && len(w.AttnNorm[layerIdx]) > 0 {
		simd.RMSNorm(x, w.AttnNorm[layerIdx], normedAttn, 1, len(x), 1e-6)
	} else {
		copy(normedAttn, x)
	}

	// Layer characteristics
	isFull := (layerIdx % 5) == 4
	headDim := 256
	heads := 8
	kvHeads := 1
	ropeTheta := float32(10000.0)
	if isFull {
		headDim = 512
		ropeTheta = float32(1000000.0)
	}

	// 2. K and V: compute for layer < 15, reuse for layer >= 15
	kvLayerIdx := layerIdx
	if layerIdx >= 15 {
		if isFull {
			kvLayerIdx = 14
		} else {
			kvLayerIdx = 13
		}
	} else {
		var kWeight, vWeight []float32
		var rawK, rawV *gguf.TensorInfo
		if layerIdx < len(w.AttnK) {
			kWeight = w.AttnK[layerIdx]
		}
		if layerIdx < len(w.RawAttnK) {
			rawK = w.RawAttnK[layerIdx]
		}
		if layerIdx < len(w.AttnV) {
			vWeight = w.AttnV[layerIdx]
		}
		if layerIdx < len(w.RawAttnV) {
			rawV = w.RawAttnV[layerIdx]
		}

		k := w.MatVec(kWeight, rawK, normedAttn) // shape [headDim]
		v := w.MatVec(vWeight, rawV, normedAttn) // shape [headDim]

		// K-Norm
		if layerIdx < len(w.AttnKNorm) && len(w.AttnKNorm[layerIdx]) == headDim {
			kNormed := make([]float32, headDim)
			simd.RMSNorm(k, w.AttnKNorm[layerIdx], kNormed, 1, headDim, 1e-6)
			k = kNormed
		}

		// K RoPE
		partialRoPECPU(k, []int{pos}, kvHeads, headDim, headDim, ropeTheta)

		// V Unit RMSNorm (weight=1.0, eps=1e-6)
		vNormed := make([]float32, headDim)
		var sumSq float32
		for _, val := range v {
			sumSq += val * val
		}
		rms := float32(math.Sqrt(float64(sumSq/float32(headDim) + 1e-6)))
		for d := 0; d < headDim; d++ {
			vNormed[d] = v[d] / rms
		}
		v = vNormed

		// Store in KV cache
		if kv != nil {
			kv.mu.Lock()
			if layerIdx >= len(kv.Keys) {
				newKeys := make([][]float32, layerIdx+1)
				newVals := make([][]float32, layerIdx+1)
				copy(newKeys, kv.Keys)
				copy(newVals, kv.Values)
				kv.Keys = newKeys
				kv.Values = newVals
			}
			kv.Keys[layerIdx] = append(kv.Keys[layerIdx], k...)
			kv.Values[layerIdx] = append(kv.Values[layerIdx], v...)
			kv.mu.Unlock()
		}
	}

	// 3. Query projection
	var qWeight []float32
	var rawQ *gguf.TensorInfo
	if layerIdx < len(w.AttnQ) {
		qWeight = w.AttnQ[layerIdx]
	}
	if layerIdx < len(w.RawAttnQ) {
		rawQ = w.RawAttnQ[layerIdx]
	}
	q := w.MatVec(qWeight, rawQ, normedAttn) // shape [heads * headDim]

	// Q-Norm (per-head with AttnQNorm)
	if layerIdx < len(w.AttnQNorm) && len(w.AttnQNorm[layerIdx]) == headDim {
		qNormed := make([]float32, len(q))
		for h := 0; h < heads; h++ {
			off := h * headDim
			simd.RMSNorm(q[off:off+headDim], w.AttnQNorm[layerIdx], qNormed[off:off+headDim], 1, headDim, 1e-6)
		}
		q = qNormed
	}

	// Q RoPE
	partialRoPECPU(q, []int{pos}, heads, headDim, headDim, ropeTheta)

	// 4. Multi-head Attention
	attnOut := make([]float32, heads*headDim)
	if kv != nil && kvLayerIdx < len(kv.Keys) {
		kv.mu.Lock()
		cachedK := kv.Keys[kvLayerIdx]
		cachedV := kv.Values[kvLayerIdx]
		kv.mu.Unlock()

		numCachedTokens := len(cachedK) / headDim
		if numCachedTokens > 0 {
			// In Gemma 4, Q and K are normalized, and attention scale is 1.0
			scale := float32(1.0)

			windowStart := 0
			if !isFull && pos >= 512 {
				windowStart = pos - 512 + 1
			}
			if windowStart > numCachedTokens {
				windowStart = numCachedTokens
			}

			scores := make([]float32, numCachedTokens)
			for h := 0; h < heads; h++ {
				qHead := q[h*headDim : (h+1)*headDim]

				var maxScore float32 = -math.MaxFloat32
				for p := windowStart; p < numCachedTokens; p++ {
					kOffset := p * headDim
					kVec := cachedK[kOffset : kOffset+headDim]
					s := simd.VecDotF32(qHead, kVec) * scale
					scores[p] = s
					if s > maxScore {
						maxScore = s
					}
				}

				var sumExp float32
				for p := windowStart; p < numCachedTokens; p++ {
					exp := float32(math.Exp(float64(scores[p] - maxScore)))
					scores[p] = exp
					sumExp += exp
				}

				invSumExp := float32(0.0)
				if sumExp > 0 {
					invSumExp = 1.0 / sumExp
				}

				outHead := attnOut[h*headDim : (h+1)*headDim]
				for p := windowStart; p < numCachedTokens; p++ {
					weight := scores[p] * invSumExp
					vOffset := p * headDim
					vVec := cachedV[vOffset : vOffset+headDim]
					simd.VecFMAF32(outHead, vVec, weight)
				}
			}
		}
	}

	// 5. Attention output projection & post-attention norm
	var oWeight []float32
	var rawO *gguf.TensorInfo
	if layerIdx < len(w.AttnO) {
		oWeight = w.AttnO[layerIdx]
	}
	if layerIdx < len(w.RawAttnO) {
		rawO = w.RawAttnO[layerIdx]
	}
	attnProj := w.MatVec(oWeight, rawO, attnOut)
	if len(attnProj) == len(x) {
		normedAttnProj := make([]float32, len(attnProj))
		if layerIdx < len(w.PostAttentionNorm) && len(w.PostAttentionNorm[layerIdx]) > 0 {
			simd.RMSNorm(attnProj, w.PostAttentionNorm[layerIdx], normedAttnProj, 1, len(attnProj), 1e-6)
		} else {
			copy(normedAttnProj, attnProj)
		}
		for j := range x {
			x[j] += normedAttnProj[j]
		}
	}

	// 6. FFN branch
	normedFFN := make([]float32, len(x))
	if layerIdx < len(w.FfnNorm) && len(w.FfnNorm[layerIdx]) > 0 {
		simd.RMSNorm(x, w.FfnNorm[layerIdx], normedFFN, 1, len(x), 1e-6)
	} else {
		copy(normedFFN, x)
	}

	var gateWeight, upWeight, downWeight []float32
	var rawGate, rawUp, rawDown *gguf.TensorInfo
	if layerIdx < len(w.FfnGate) {
		gateWeight = w.FfnGate[layerIdx]
	}
	if layerIdx < len(w.RawFfnGate) {
		rawGate = w.RawFfnGate[layerIdx]
	}
	if layerIdx < len(w.FfnUp) {
		upWeight = w.FfnUp[layerIdx]
	}
	if layerIdx < len(w.RawFfnUp) {
		rawUp = w.RawFfnUp[layerIdx]
	}
	if layerIdx < len(w.FfnDown) {
		downWeight = w.FfnDown[layerIdx]
	}
	if layerIdx < len(w.RawFfnDown) {
		rawDown = w.RawFfnDown[layerIdx]
	}

	gate := w.MatVec(gateWeight, rawGate, normedFFN)
	up := w.MatVec(upWeight, rawUp, normedFFN)
	act := make([]float32, len(gate))
	for j := range act {
		act[j] = geluCPU(gate[j]) * up[j]
	}
	down := w.MatVec(downWeight, rawDown, act)

	if len(down) == len(x) {
		normedDown := make([]float32, len(down))
		if layerIdx < len(w.PostFfnNorm) && len(w.PostFfnNorm[layerIdx]) > 0 {
			simd.RMSNorm(down, w.PostFfnNorm[layerIdx], normedDown, 1, len(down), 1e-6)
		} else {
			copy(normedDown, down)
		}
		for j := range x {
			x[j] += normedDown[j]
		}
	}

	// 7. PLE residual branch
	if len(pleSlice) == 256 {
		var inpGateWeight, projWeight []float32
		var rawInpGate, rawProj *gguf.TensorInfo
		if layerIdx < len(w.InpGate) {
			inpGateWeight = w.InpGate[layerIdx]
		}
		if layerIdx < len(w.RawInpGate) {
			rawInpGate = w.RawInpGate[layerIdx]
		}
		if layerIdx < len(w.Proj) {
			projWeight = w.Proj[layerIdx]
		}
		if layerIdx < len(w.RawProj) {
			rawProj = w.RawProj[layerIdx]
		}

		inpGate := w.MatVec(inpGateWeight, rawInpGate, x)
		if len(inpGate) == 256 {
			gated := make([]float32, 256)
			for j := 0; j < 256; j++ {
				gated[j] = geluCPU(inpGate[j]) * pleSlice[j]
			}
			proj := w.MatVec(projWeight, rawProj, gated)
			if len(proj) == len(x) {
				normedProj := make([]float32, len(proj))
				if layerIdx < len(w.PostNorm) && len(w.PostNorm[layerIdx]) > 0 {
					simd.RMSNorm(proj, w.PostNorm[layerIdx], normedProj, 1, len(proj), 1e-6)
				} else {
					copy(normedProj, proj)
				}
				for j := range x {
					x[j] += normedProj[j]
				}
			}
		}
	}

	// 8. Layer output scale
	if layerIdx < len(w.LayerOutputScale) && len(w.LayerOutputScale[layerIdx]) > 0 {
		scale := w.LayerOutputScale[layerIdx][0]
		for j := range x {
			x[j] *= scale
		}
	}

	return x
}
