//go:build linux && !cuda

package engine

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
	"github.com/23skdu/longbow-quarrel/internal/logger"
	"github.com/23skdu/longbow-quarrel/internal/simd"
	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
)

type CPUEngine struct {
	model   *gguf.GGUFFile
	config  config.Config
	weights *CPUWeights
	tok     *tokenizer.Tokenizer
}

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
}

func init() {
	RegisterEngine("cpu", NewCPUEngine)
}

func NewCPUEngine(modelPath string, cfg config.Config) (Engine, error) {
	f, err := gguf.LoadFile(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load GGUF: %w", err)
	}

	weights, err := loadCPUWeights(f)
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("failed to load weights: %w", err)
	}

	tok, err := tokenizer.New(modelPath)
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("failed to load tokenizer: %w", err)
	}

	logger.Log.Info("CPU engine initialized", "model", modelPath)

	return &CPUEngine{
		model:   f,
		config:  cfg,
		weights: weights,
		tok:     tok,
	}, nil
}

func loadCPUWeights(f *gguf.GGUFFile) (*CPUWeights, error) {
	w := &CPUWeights{}

	numLayers := 1
	if v, ok := f.KV["llama.block_count"].(uint32); ok {
		numLayers = int(v)
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

	for _, t := range f.Tensors {
		data, err := decodeTensorData(t)
		if err != nil {
			continue
		}

		switch t.Name {
		case "token_embd.weight":
			w.TokenEmb = append(w.TokenEmb, data)
		case "output.weight":
			w.Output = data
		case "lm_head.weight":
			if len(w.Output) == 0 {
				w.Output = data
			}
		case "output_norm.weight":
			w.OutputNorm = data
		default:
			var layer int
			var _, _ = fmt.Sscanf(t.Name, "blk.%d.", &layer)
			if layer < numLayers {
				switch {
				case contains(t.Name, "attn_q.weight"):
					w.AttnQ[layer] = data
				case contains(t.Name, "attn_k.weight"):
					w.AttnK[layer] = data
				case contains(t.Name, "attn_v.weight"):
					w.AttnV[layer] = data
				case contains(t.Name, "attn_output.weight"):
					w.AttnO[layer] = data
				case contains(t.Name, "attn_norm.weight"):
					w.AttnNorm[layer] = data
				case contains(t.Name, "ffn_gate.weight"):
					w.FfnGate[layer] = data
				case contains(t.Name, "ffn_down.weight"):
					w.FfnDown[layer] = data
				case contains(t.Name, "ffn_up.weight"):
					w.FfnUp[layer] = data
				case contains(t.Name, "ffn_norm.weight"):
					w.FfnNorm[layer] = data
				}
			}
		}
	}

	return w, nil
}

func decodeTensorData(t *gguf.TensorInfo) ([]float32, error) {
	numElements := uint32(1)
	for _, d := range t.Dimensions {
		numElements *= uint32(d)
	}

	data := make([]float32, numElements)
	bytesPerElement := t.SizeBytes() / uint64(numElements)

	for i := uint32(0); i < numElements; i++ {
		offset := uint64(i) * bytesPerElement
		switch bytesPerElement {
		case 4:
			bits := uint32(t.Data[offset]) | uint32(t.Data[offset+1])<<8 | uint32(t.Data[offset+2])<<16 | uint32(t.Data[offset+3])<<24
			data[i] = math.Float32frombits(bits)
		case 2:
			bits := uint16(t.Data[offset]) | uint16(t.Data[offset+1])<<8
			data[i] = float32(bits) / 32767.0
		default:
			data[i] = float32(t.Data[offset]) / 127.0
		}
	}

	return data, nil
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > len(substr) && (s[:len(substr)] == substr || contains(s[1:], substr)))
}

func (e *CPUEngine) Infer(tokens []int, count int, cfg SamplerConfig) ([]int, error) {
	result := make([]int, 0, count)

	for len(result) < count {
		nextToken, err := e.sample(tokens, cfg)
		if err != nil {
			return result, err
		}
		result = append(result, nextToken)
		tokens = append(tokens, nextToken)
	}

	return result, nil
}

func (e *CPUEngine) InferWithCallback(tokens []int, count int, cfg SamplerConfig, callback func(token int)) ([]int, error) {
	result := make([]int, 0, count)

	for len(result) < count {
		nextToken, err := e.sample(tokens, cfg)
		if err != nil {
			return result, err
		}
		result = append(result, nextToken)
		tokens = append(tokens, nextToken)
		callback(nextToken)
	}

	return result, nil
}

func (e *CPUEngine) sample(tokens []int, cfg SamplerConfig) (int, error) {
	logits := e.forward(tokens)

	if cfg.Temperature > 0 {
		logits = applyTemperature(logits, cfg.Temperature)
	}

	logits = applyTopK(logits, cfg.TopK)
	logits = applyTopP(logits, cfg.TopP)

	probs := softmax(logits)

	seed := cfg.Seed + int64(len(tokens))
	r := rand.New(rand.NewSource(seed))

	return sampleFromDist(probs, r), nil
}

func (e *CPUEngine) forward(tokens []int) []float32 {
	hiddenSize := 4096
	if len(e.weights.TokenEmb) > 0 {
		hiddenSize = len(e.weights.TokenEmb[0])
	}

	hidden := make([]float32, hiddenSize)

	lastToken := tokens[len(tokens)-1]
	if lastToken < len(e.weights.TokenEmb) && len(e.weights.TokenEmb) > 0 {
		copy(hidden, e.weights.TokenEmb[lastToken])
	}

	return hidden
}

func (e *CPUEngine) Close() {
	if e.weights != nil {
		e.weights.Free()
	}
	if e.model != nil {
		e.model.Close()
	}
	logger.Log.Info("CPU engine closed")
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
}

func applyTemperature(logits []float32, temp float64) []float32 {
	result := make([]float32, len(logits))
	for i, l := range logits {
		result[i] = float32(float64(l) / temp)
	}
	return result
}

func applyTopK(logits []float32, k int) []float32 {
	if k >= len(logits) || k <= 0 {
		return logits
	}

	indices := make([]int, len(logits))
	for i := range indices {
		indices[i] = i
	}

	for i := 0; i < k; i++ {
		maxIdx := i
		maxVal := logits[indices[i]]
		for j := i + 1; j < len(logits); j++ {
			if logits[indices[j]] > maxVal {
				maxIdx = j
				maxVal = logits[indices[j]]
			}
		}
		indices[i], indices[maxIdx] = indices[maxIdx], indices[i]
	}

	topKVal := logits[indices[k-1]]
	for i := k; i < len(logits); i++ {
		if logits[indices[i]] > topKVal {
			logits[indices[i]] = float32(-math.Inf(1))
		}
	}

	return logits
}

func applyTopP(logits []float32, p float64) []float32 {
	probs := softmax(logits)
	cumSum := 0.0
	cutoff := p

	for i, prob := range probs {
		if prob > 0 {
			cumSum += float64(prob)
			if cumSum <= cutoff {
				logits[i] = float32(float64(logits[i]) * p / cutoff)
			} else {
				logits[i] = float32(-math.Inf(1))
			}
		}
	}

	return logits
}

func softmax(logits []float32) []float32 {
	probs := make([]float32, len(logits))
	copy(probs, logits)
	simd.SoftmaxAVX2(probs)
	return probs
}

func sampleFromDist(probs []float32, r *rand.Rand) int {
	cumSum := float32(0)
	threshold := float32(r.Float32())
	for i, p := range probs {
		cumSum += p
		if cumSum >= threshold {
			return i
		}
	}
	return len(probs) - 1
}
