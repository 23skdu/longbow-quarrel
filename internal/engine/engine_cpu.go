//go:build !cuda && !tpu && !metal

package engine

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
	"github.com/23skdu/longbow-quarrel/internal/logger"
	"github.com/23skdu/longbow-quarrel/internal/simd"
	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
	"sync"
	"time"
)

type CPUEngine struct {
	model   *gguf.GGUFFile
	config  config.Config
	weights *CPUWeights
	tok     *tokenizer.Tokenizer
	ctx     *device.Context
	cache   *PagedKVCache

	BatchManager *ContinuousBatchManager
	stopChan     chan struct{}
	doneChan     chan struct{}
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

	// Qwen3.5 & hybrid layer support
	AttnQNorm [][]float32
	AttnKNorm [][]float32
	AttnQKV   [][]float32
	AttnGate  [][]float32
	SSMConv1d [][]float32
	SSMA      [][]float32
	SSMAlpha  [][]float32
	SSMBeta   [][]float32
	SSMDtBias [][]float32
	SSMNorm   [][]float32
	SSMOut    [][]float32

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

func init() {
	RegisterEngine("cpu", NewCPUEngine)
}

func NewCPUEngine(modelPath string, cfg config.Config) (Engine, error) {
	f, err := gguf.LoadFile(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load GGUF: %w", err)
	}

	// Apply memory limit before any allocation
	if cfg.MaxMemoryMB > 0 {
		device.SetMaxMemoryMB(cfg.MaxMemoryMB)
		if cfg.DebugMemory {
			logger.Log.Info("memory limit set", "max_mb", cfg.MaxMemoryMB)
		}
	}

	// Load Dimensions from GGUF into config using dynamic architecture extraction
	modelCfg := ExtractModelConfig(f)
	if cfg.Layers <= 0 {
		cfg.Layers = modelCfg.Layers
	}
	if cfg.Dim <= 0 {
		cfg.Dim = modelCfg.Dim
	}
	if cfg.Heads <= 0 {
		cfg.Heads = modelCfg.Heads
	}
	if cfg.KVHeads <= 0 {
		cfg.KVHeads = modelCfg.KVHeads
	}
	if cfg.HeadDim <= 0 {
		cfg.HeadDim = modelCfg.HeadDim
	}
	if cfg.HiddenDim <= 0 {
		cfg.HiddenDim = modelCfg.HiddenDim
	}
	if cfg.VocabSize <= 0 {
		cfg.VocabSize = modelCfg.VocabSize
	}
	if cfg.SeqLen <= 0 {
		cfg.SeqLen = modelCfg.SeqLen
	}
	if cfg.Eps <= 0 {
		cfg.Eps = modelCfg.Eps
	}
	if cfg.RopeTheta <= 0 {
		cfg.RopeTheta = modelCfg.RopeTheta
	}
	if cfg.Architecture == "" {
		cfg.Architecture = modelCfg.Architecture
	}

	// Cap KV cache based on memory budget
	if cfg.MaxMemoryMB > 0 && cfg.KVCacheSize == 0 {
		// Estimate: each layer's KV cache = seqLen * kvHeads * headDim * 4 bytes * 2 (K+V)
		// Aim to use at most 60% of budget for KV cache
		estBytesPerToken := uint64(cfg.Layers) * uint64(cfg.KVHeads) * uint64(cfg.HeadDim) * 4 * 2 // #nosec G115
		if estBytesPerToken > 0 {
			maxTokensForKV := uint64(cfg.MaxMemoryMB) * 1024 * 1024 * 60 / 100 / estBytesPerToken
			if maxTokensForKV < uint64(cfg.SeqLen) { // #nosec G115 -- safe: SeqLen is bounded by memory
				cfg.SeqLen = int(maxTokensForKV) // #nosec G115
				if cfg.DebugMemory {
					logger.Log.Info("KV cache capped by memory limit", "seq_len", cfg.SeqLen)
				}
			}
		}
	}

	weights, err := loadCPUWeights(f, cfg)
	if err != nil {
		_ = f.Close()
		return nil, fmt.Errorf("failed to load weights: %w", err)
	}

	tok, err := tokenizer.New(modelPath)
	if err != nil {
		_ = f.Close()
		return nil, fmt.Errorf("failed to load tokenizer: %w", err)
	}

	ctx := device.NewContext()
	cache := &PagedKVCache{}
	if err := cache.Init(ctx, cfg); err != nil {
		ctx.Free()
		_ = f.Close()
		return nil, fmt.Errorf("failed to initialize paged cache: %w", err)
	}

	logger.Log.Info("CPU engine initialized with PagedKVCache", "model", modelPath, "heads", cfg.Heads, "kv_heads", cfg.KVHeads, "dim", cfg.Dim)

	e := &CPUEngine{
		model:        f,
		config:       cfg,
		weights:      weights,
		tok:          tok,
		ctx:          ctx,
		cache:        cache,
		BatchManager: NewContinuousBatchManager(),
		stopChan:     make(chan struct{}),
		doneChan:     make(chan struct{}),
	}

	go e.runBatchLoop()

	return e, nil
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
	w.SSMA = make([][]float32, numLayers)
	w.SSMAlpha = make([][]float32, numLayers)
	w.SSMBeta = make([][]float32, numLayers)
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
				case contains(t.Name, "attn_q_norm.weight"):
					if data, err := decodeTensorData(t); err == nil {
						w.AttnQNorm[layer] = data
					}
				case contains(t.Name, "attn_k_norm.weight"):
					if data, err := decodeTensorData(t); err == nil {
						w.AttnKNorm[layer] = data
					}
				case contains(t.Name, "attn_norm.weight"):
					if data, err := decodeTensorData(t); err == nil {
						w.AttnNorm[layer] = data
					}
				case contains(t.Name, "ffn_norm.weight"), contains(t.Name, "post_attention_norm.weight"):
					if data, err := decodeTensorData(t); err == nil {
						w.FfnNorm[layer] = data
					}
				case contains(t.Name, "ssm_conv1d.weight"):
					if data, err := decodeTensorData(t); err == nil {
						w.SSMConv1d[layer] = data
					}
				case contains(t.Name, "ssm_a"):
					if data, err := decodeTensorData(t); err == nil {
						w.SSMA[layer] = data
					}
				case contains(t.Name, "ssm_alpha.weight"):
					if data, err := decodeTensorData(t); err == nil {
						w.SSMAlpha[layer] = data
					}
				case contains(t.Name, "ssm_beta.weight"):
					if data, err := decodeTensorData(t); err == nil {
						w.SSMBeta[layer] = data
					}
				case contains(t.Name, "ssm_dt.bias"), contains(t.Name, "ssm_dt.weight"):
					if data, err := decodeTensorData(t); err == nil {
						w.SSMDtBias[layer] = data
					}
				case contains(t.Name, "ssm_norm.weight"):
					if data, err := decodeTensorData(t); err == nil {
						w.SSMNorm[layer] = data
					}
				case contains(t.Name, "attn_q.weight"):
					w.RawAttnQ[layer] = t
					if !isQ8Matrix {
						if data, err := decodeTensorData(t); err == nil {
							w.AttnQ[layer] = data
						}
					}
				case contains(t.Name, "attn_k.weight"):
					w.RawAttnK[layer] = t
					if !isQ8Matrix {
						if data, err := decodeTensorData(t); err == nil {
							w.AttnK[layer] = data
						}
					}
				case contains(t.Name, "attn_v.weight"):
					w.RawAttnV[layer] = t
					if !isQ8Matrix {
						if data, err := decodeTensorData(t); err == nil {
							w.AttnV[layer] = data
						}
					}
				case contains(t.Name, "attn_output.weight"):
					w.RawAttnO[layer] = t
					if !isQ8Matrix {
						if data, err := decodeTensorData(t); err == nil {
							w.AttnO[layer] = data
						}
					}
				case contains(t.Name, "attn_qkv.weight"):
					w.RawAttnQKV[layer] = t
					if !isQ8Matrix {
						if data, err := decodeTensorData(t); err == nil {
							w.AttnQKV[layer] = data
						}
					}
				case contains(t.Name, "attn_gate.weight"):
					w.RawAttnGate[layer] = t
					if !isQ8Matrix {
						if data, err := decodeTensorData(t); err == nil {
							w.AttnGate[layer] = data
						}
					}
				case contains(t.Name, "ssm_out.weight"):
					w.RawSSMOut[layer] = t
					if !isQ8Matrix {
						if data, err := decodeTensorData(t); err == nil {
							w.SSMOut[layer] = data
						}
					}
				case contains(t.Name, "ffn_gate.weight"):
					w.RawFfnGate[layer] = t
					if !isQ8Matrix {
						if data, err := decodeTensorData(t); err == nil {
							w.FfnGate[layer] = data
						}
					}
				case contains(t.Name, "ffn_down.weight"):
					w.RawFfnDown[layer] = t
					if !isQ8Matrix {
						if data, err := decodeTensorData(t); err == nil {
							w.FfnDown[layer] = data
						}
					}
				case contains(t.Name, "ffn_up.weight"):
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

	// Handle quantized types properly using gguf dequantization
	switch t.Type {
	case gguf.GGMLTypeQ4_K:
		return gguf.DequantizeQ4K(t.Data, int(numElements)), nil
	case gguf.GGMLTypeQ6_K:
		return gguf.DequantizeQ6K(t.Data, int(numElements)), nil
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
	case gguf.GGMLTypeF32:
		// Handle F32 directly
		data := make([]float32, numElements)
		for i := uint32(0); i < numElements; i++ {
			offset := uint64(i) * 4
			bits := uint32(t.Data[offset]) | uint32(t.Data[offset+1])<<8 | uint32(t.Data[offset+2])<<16 | uint32(t.Data[offset+3])<<24
			data[i] = math.Float32frombits(bits)
		}
		return data, nil
	case gguf.GGMLTypeF16:
		// Handle F16 - convert to F32
		data := make([]float32, numElements)
		for i := uint32(0); i < numElements; i++ {
			offset := uint64(i) * 2
			bits := uint16(t.Data[offset]) | uint16(t.Data[offset+1])<<8
			data[i] = float32(bits) / 32767.0
		}
		return data, nil
	default:
		// For unsupported types, fallback to zero-filled array
		return make([]float32, numElements), nil
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > len(substr) && (s[:len(substr)] == substr || contains(s[1:], substr)))
}

func (e *CPUEngine) Config() config.Config {
	return e.config
}

func (e *CPUEngine) GetSeqCachePos(seqID string) int {
	// CPU engine doesn't have a sophisticated KV cache yet
	return 0
}

func (e *CPUEngine) Infer(tokens []int, count int, cfg SamplerConfig) ([]int, error) {
	return e.inferInternal(tokens, count, cfg, nil, nil)
}

func (e *CPUEngine) InferWithLogits(tokens []int, count int, cfg SamplerConfig) ([]int, []float32, error) {
	var lastLogits []float32
	tokens, err := e.inferInternal(tokens, count, cfg, nil, func(logits []float32) {
		lastLogits = make([]float32, len(logits))
		copy(lastLogits, logits)
	})
	return tokens, lastLogits, err
}

func (e *CPUEngine) InferWithCallback(tokens []int, count int, cfg SamplerConfig, callback func(token int)) ([]int, error) {
	return e.inferInternal(tokens, count, cfg, callback, nil)
}

func (e *CPUEngine) InferWithCallbackLogits(tokens []int, count int, cfg SamplerConfig, tokenCallback func(int), logitsCallback func([]float32)) ([]int, error) {
	return e.inferInternal(tokens, count, cfg, tokenCallback, logitsCallback)
}

func (e *CPUEngine) runBatchLoop() {
	defer close(e.doneChan)
	for {
		select {
		case <-e.stopChan:
			return
		default:
		}

		// 1. Pull active sequences
		desc, _ := e.BatchManager.Step(16, e.cache, nil) // No PromptCache for CPU yet
		if desc == nil || len(desc.Sequences) == 0 {
			time.Sleep(10 * time.Millisecond)
			continue
		}

		// 2. Forward Pass
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

		// 3. Sampling & Update
		for i := range desc.Sequences {
			seq := desc.Sequences[i]
			logits := results[i].ToHostF32()
			results[i].Free()

			if seq.LogitsCallback != nil {
				seq.LogitsCallback(logits)
			}

			sampler := NewSampler(seq.Config)
			token := sampler.Sample(logits, seq.Tokens)

			// Update Sequence State
			chunkLen := 1
			if i < len(desc.Offsets)-1 {
				chunkLen = desc.Offsets[i+1] - desc.Offsets[i]
			} else {
				chunkLen = len(desc.Tokens) - desc.Offsets[i]
			}

			seq.Tokens = append(seq.Tokens, token)
			seq.Pos += chunkLen

			if seq.TokenCallback != nil {
				seq.TokenCallback(token)
			}

			// Termination
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

func (e *CPUEngine) ForwardBatch(desc *BatchDescriptor) ([]*device.Tensor, error) {
	batchSize := len(desc.Sequences)
	results := make([]*device.Tensor, batchSize)

	// Use a WaitGroup to parallelize sequence processing
	var wg sync.WaitGroup
	wg.Add(batchSize)

	for i := range desc.Sequences {
		go func(idx int) {
			defer wg.Done()

			start := desc.Offsets[idx]
			var end int
			if idx < batchSize-1 {
				end = desc.Offsets[idx+1]
			} else {
				end = len(desc.Tokens)
			}

			seqTokens := desc.Tokens[start:end]

			// For CPUEngine, each sequence is processed by a separate worker thread
			// In a production engine, this would call SIMD-optimized layer kernels.
			hidden := e.forward(seqTokens)

			res := e.ctx.NewTensorFP32(1, len(hidden))
			_ = res.LoadFrom(hidden) // Ignoring here for now as forward already produced the data
			results[idx] = res
		}(i)
	}

	wg.Wait()
	return results, nil
}

func (e *CPUEngine) inferInternal(inputTokens []int, tokensToGenerate int, samplerConfig SamplerConfig, tokenCallback func(int), logitsCallback func([]float32)) ([]int, error) {
	resChan := make(chan []int, 1)
	errChan := make(chan error, 1)

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

	select {
	case tokens := <-resChan:
		if len(tokens) > len(inputTokens) {
			return tokens[len(inputTokens):], nil
		}
		return []int{}, nil
	case err := <-errChan:
		return nil, err
	}
}

func (e *CPUEngine) SwapModel(modelPath string, cfg config.Config) error {
	e.Close()
	newEngine, err := NewCPUEngine(modelPath, cfg)
	if err != nil {
		return err
	}
	*e = *(newEngine.(*CPUEngine))
	return nil
}

func (e *CPUEngine) forward(tokens []int) []float32 {
	hiddenSize := e.config.Dim
	if hiddenSize <= 0 {
		hiddenSize = 576
	}

	hidden := make([]float32, hiddenSize)

	if len(tokens) > 0 {
		lastToken := tokens[len(tokens)-1]
		tokVec := e.weights.GetTokenEmbedding(lastToken, hiddenSize)
		copy(hidden, tokVec)
	}

	for layerIdx := 0; layerIdx < e.config.Layers; layerIdx++ {
		hidden = e.applyLayerCPU(hidden, layerIdx)
	}

	if e.weights.OutputNorm != nil {
		normed := make([]float32, len(hidden))
		simd.RMSNorm(hidden, e.weights.OutputNorm, normed, 1, len(hidden), e.config.Eps)
		hidden = normed
	}

	if len(e.weights.Output) > 0 || e.weights.RawOutput != nil {
		logits := e.weights.MatVec(e.weights.Output, e.weights.RawOutput, hidden)
		return logits
	}

	return hidden
}

func (e *CPUEngine) applyLayerCPU(input []float32, layerIdx int) []float32 {
	normed := make([]float32, len(input))
	if layerIdx < len(e.weights.AttnNorm) && len(e.weights.AttnNorm[layerIdx]) > 0 {
		simd.RMSNorm(input, e.weights.AttnNorm[layerIdx], normed, 1, len(input), e.config.Eps)
	} else {
		copy(normed, input)
	}

	var out []float32

	if e.weights.HasFullAttn(layerIdx) {
		var qWeight, kWeight, vWeight, oWeight []float32
		var rawQ, rawK, rawV, rawO *gguf.TensorInfo
		if layerIdx < len(e.weights.AttnQ) { qWeight = e.weights.AttnQ[layerIdx] }
		if layerIdx < len(e.weights.AttnK) { kWeight = e.weights.AttnK[layerIdx] }
		if layerIdx < len(e.weights.AttnV) { vWeight = e.weights.AttnV[layerIdx] }
		if layerIdx < len(e.weights.AttnO) { oWeight = e.weights.AttnO[layerIdx] }
		if layerIdx < len(e.weights.RawAttnQ) { rawQ = e.weights.RawAttnQ[layerIdx] }
		if layerIdx < len(e.weights.RawAttnK) { rawK = e.weights.RawAttnK[layerIdx] }
		if layerIdx < len(e.weights.RawAttnV) { rawV = e.weights.RawAttnV[layerIdx] }
		if layerIdx < len(e.weights.RawAttnO) { rawO = e.weights.RawAttnO[layerIdx] }

		q := e.weights.MatVec(qWeight, rawQ, normed)
		k := e.weights.MatVec(kWeight, rawK, normed)
		v := e.weights.MatVec(vWeight, rawV, normed)

		// Q/K normalization if present (e.g. Qwen 3.5, Gemma)
		if layerIdx < len(e.weights.AttnQNorm) && len(e.weights.AttnQNorm[layerIdx]) > 0 && len(q) == len(e.weights.AttnQNorm[layerIdx]) {
			qNormed := make([]float32, len(q))
			simd.RMSNorm(q, e.weights.AttnQNorm[layerIdx], qNormed, 1, len(q), e.config.Eps)
			q = qNormed
		}
		if layerIdx < len(e.weights.AttnKNorm) && len(e.weights.AttnKNorm[layerIdx]) > 0 && len(k) == len(e.weights.AttnKNorm[layerIdx]) {
			kNormed := make([]float32, len(k))
			simd.RMSNorm(k, e.weights.AttnKNorm[layerIdx], kNormed, 1, len(k), e.config.Eps)
			k = kNormed
		}

		attn := attentionCPU(q, k, v, e.config.Heads, e.config.KVHeads, e.config.HeadDim)
		out = e.weights.MatVec(oWeight, rawO, attn)
	} else if e.weights.HasSSM(layerIdx) {
		var qkvWeight, gateWeight, ssmOutWeight []float32
		var rawQKV, rawGate, rawSSMOut *gguf.TensorInfo
		if layerIdx < len(e.weights.AttnQKV) { qkvWeight = e.weights.AttnQKV[layerIdx] }
		if layerIdx < len(e.weights.AttnGate) { gateWeight = e.weights.AttnGate[layerIdx] }
		if layerIdx < len(e.weights.SSMOut) { ssmOutWeight = e.weights.SSMOut[layerIdx] }
		if layerIdx < len(e.weights.RawAttnQKV) { rawQKV = e.weights.RawAttnQKV[layerIdx] }
		if layerIdx < len(e.weights.RawAttnGate) { rawGate = e.weights.RawAttnGate[layerIdx] }
		if layerIdx < len(e.weights.RawSSMOut) { rawSSMOut = e.weights.RawSSMOut[layerIdx] }

		qkv := e.weights.MatVec(qkvWeight, rawQKV, normed)
		if len(gateWeight) > 0 || rawGate != nil {
			gate := e.weights.MatVec(gateWeight, rawGate, normed)
			for i := range gate {
				if i < len(qkv) {
					qkv[i] = qkv[i] * (gate[i] / (1.0 + float32(math.Exp(-float64(gate[i])))))
				}
			}
		}
		out = e.weights.MatVec(ssmOutWeight, rawSSMOut, qkv)
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
	if e.weights.HasFFN(layerIdx) {
		normedFFN := make([]float32, len(residual))
		simd.RMSNorm(residual, e.weights.FfnNorm[layerIdx], normedFFN, 1, len(residual), e.config.Eps)

		var gateWeight, upWeight, downWeight []float32
		var rawGate, rawUp, rawDown *gguf.TensorInfo
		if layerIdx < len(e.weights.FfnGate) { gateWeight = e.weights.FfnGate[layerIdx] }
		if layerIdx < len(e.weights.FfnUp) { upWeight = e.weights.FfnUp[layerIdx] }
		if layerIdx < len(e.weights.FfnDown) { downWeight = e.weights.FfnDown[layerIdx] }
		if layerIdx < len(e.weights.RawFfnGate) { rawGate = e.weights.RawFfnGate[layerIdx] }
		if layerIdx < len(e.weights.RawFfnUp) { rawUp = e.weights.RawFfnUp[layerIdx] }
		if layerIdx < len(e.weights.RawFfnDown) { rawDown = e.weights.RawFfnDown[layerIdx] }

		gate := e.weights.MatVec(gateWeight, rawGate, normedFFN)
		up := e.weights.MatVec(upWeight, rawUp, normedFFN)

		swiGLU := make([]float32, len(gate))
		simd.SwiGLU(gate, up, swiGLU)

		down := e.weights.MatVec(downWeight, rawDown, swiGLU)

		result := make([]float32, len(residual))
		for i := range result {
			result[i] = residual[i] + down[i]
		}
		return result
	}

	return residual
}

func rmsNormCPU(input, weight []float32, eps float32) []float32 {
	result := make([]float32, len(input))
	simd.RMSNorm(input, weight, result, 1, len(input), eps)
	return result
}

func sigmoid(x float32) float32 {
	if x < -30 {
		return 0
	}
	if x > 30 {
		return 1
	}
	return float32(1.0 / (1.0 + math.Exp(-float64(x))))
}

func vecDot(a, b []float32) float32 {
	n := len(a)
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := 0
	for ; i <= n-8; i += 8 {
		s0 += a[i+0] * b[i+0]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
		s4 += a[i+4] * b[i+4]
		s5 += a[i+5] * b[i+5]
		s6 += a[i+6] * b[i+6]
		s7 += a[i+7] * b[i+7]
	}
	sum := ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))
	for ; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

func vecFMA(dst, src []float32, weight float32) {
	n := len(dst)
	i := 0
	for ; i <= n-8; i += 8 {
		dst[i+0] += weight * src[i+0]
		dst[i+1] += weight * src[i+1]
		dst[i+2] += weight * src[i+2]
		dst[i+3] += weight * src[i+3]
		dst[i+4] += weight * src[i+4]
		dst[i+5] += weight * src[i+5]
		dst[i+6] += weight * src[i+6]
		dst[i+7] += weight * src[i+7]
	}
	for ; i < n; i++ {
		dst[i] += weight * src[i]
	}
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

		// Support GQA / MQA mapping
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

func attentionCPUScalar(q, k, v []float32, numHeads, kvHeads, headDim int) []float32 {
	headSize := headDim
	seqLen := len(q) / (numHeads * headDim)
	scale := 1.0 / math.Sqrt(float64(headDim))
	result := make([]float32, len(q))

	for h := 0; h < numHeads; h++ {
		qHead := q[h*headSize*seqLen : (h+1)*headSize*seqLen]
		kvPerHead := numHeads / kvHeads
		kh := h / kvPerHead
		kHead := k[kh*headSize*seqLen : (kh+1)*headSize*seqLen]
		vHead := v[kh*headSize*seqLen : (kh+1)*headSize*seqLen]

		for i := 0; i < seqLen; i++ {
			var maxScore float64
			for j := 0; j <= i; j++ {
				dot := 0.0
				for d := 0; d < headDim; d++ {
					dot += float64(qHead[i*headDim+d]) * float64(kHead[j*headDim+d])
				}
				score := dot * scale
				if j == 0 || score > maxScore {
					maxScore = score
				}
			}

			var sum float64
			for j := 0; j <= i; j++ {
				dot := 0.0
				for d := 0; d < headDim; d++ {
					dot += float64(qHead[i*headDim+d]) * float64(kHead[j*headDim+d])
				}
				sum += math.Exp(dot*scale - maxScore)
			}

			for d := 0; d < headDim; d++ {
				var attnSum float64
				for j := 0; j <= i; j++ {
					dot := 0.0
					for dd := 0; dd < headDim; dd++ {
						dot += float64(qHead[i*headDim+dd]) * float64(kHead[j*headDim+dd])
					}
					attnSum += math.Exp(dot*scale-maxScore) * float64(vHead[j*headDim+d])
				}
				result[h*seqLen*headDim+i*headDim+d] = float32(attnSum / sum)
			}
		}
	}

	return result
}

func matMulVec(matrix []float32, vector []float32) []float32 {
	rows := len(matrix) / len(vector)
	result := make([]float32, rows)
	for i := 0; i < rows; i++ {
		sum := 0.0
		for j := 0; j < len(vector); j++ {
			sum += float64(matrix[i*len(vector)+j]) * float64(vector[j])
		}
		result[i] = float32(sum)
	}
	return result
}

func (e *CPUEngine) Close() {
	if e.cache != nil {
		e.cache.Free()
	}
	if e.ctx != nil {
		e.ctx.Free()
	}
	if e.model != nil {
		_ = e.model.Close()
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
		rows := int(raw.Dimensions[0])
		if len(raw.Dimensions) > 1 {
			rows = int(raw.Dimensions[1])
		}
		if raw.Type == gguf.GGMLTypeQ8_0 {
			return gguf.MatVecMulQ8_0(raw.Data, x, rows, cols)
		}
		data, err := decodeTensorData(raw)
		if err == nil && len(data) > 0 {
			return simd.MatVecMul(data, x, len(data)/cols, cols)
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
		if w.RawTokenEmb.Type == gguf.GGMLTypeQ8_0 {
			const blockSize = 32
			const blockSizeBytes = 34
			blocksPerRow := hiddenSize / blockSize
			rowBytes := blocksPerRow * blockSizeBytes
			offset := tokenId * rowBytes
			if offset+rowBytes <= len(w.RawTokenEmb.Data) {
				return gguf.DequantizeQ8_0(w.RawTokenEmb.Data[offset:offset+rowBytes], hiddenSize)
			}
		} else if w.RawTokenEmb.Type == gguf.GGMLTypeF32 {
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
	hasQ := (layerIdx < len(w.AttnQ) && len(w.AttnQ[layerIdx]) > 0) || (layerIdx < len(w.RawAttnQ) && w.RawAttnQ[layerIdx] != nil)
	hasK := (layerIdx < len(w.AttnK) && len(w.AttnK[layerIdx]) > 0) || (layerIdx < len(w.RawAttnK) && w.RawAttnK[layerIdx] != nil)
	hasV := (layerIdx < len(w.AttnV) && len(w.AttnV[layerIdx]) > 0) || (layerIdx < len(w.RawAttnV) && w.RawAttnV[layerIdx] != nil)
	hasO := (layerIdx < len(w.AttnO) && len(w.AttnO[layerIdx]) > 0) || (layerIdx < len(w.RawAttnO) && w.RawAttnO[layerIdx] != nil)
	return hasQ && hasK && hasV && hasO
}

func (w *CPUWeights) HasSSM(layerIdx int) bool {
	hasQKV := (layerIdx < len(w.AttnQKV) && len(w.AttnQKV[layerIdx]) > 0) || (layerIdx < len(w.RawAttnQKV) && w.RawAttnQKV[layerIdx] != nil)
	hasOut := (layerIdx < len(w.SSMOut) && len(w.SSMOut[layerIdx]) > 0) || (layerIdx < len(w.RawSSMOut) && w.RawSSMOut[layerIdx] != nil)
	return hasQKV && hasOut
}

func (w *CPUWeights) HasFFN(layerIdx int) bool {
	hasNorm := layerIdx < len(w.FfnNorm) && len(w.FfnNorm[layerIdx]) > 0
	hasGate := (layerIdx < len(w.FfnGate) && len(w.FfnGate[layerIdx]) > 0) || (layerIdx < len(w.RawFfnGate) && w.RawFfnGate[layerIdx] != nil)
	hasUp := (layerIdx < len(w.FfnUp) && len(w.FfnUp[layerIdx]) > 0) || (layerIdx < len(w.RawFfnUp) && w.RawFfnUp[layerIdx] != nil)
	hasDown := (layerIdx < len(w.FfnDown) && len(w.FfnDown[layerIdx]) > 0) || (layerIdx < len(w.RawFfnDown) && w.RawFfnDown[layerIdx] != nil)
	return hasNorm && hasGate && hasUp && hasDown
}

func applyTempCPU(logits []float32, temp float64) []float32 {
	result := make([]float32, len(logits))
	for i, l := range logits {
		result[i] = float32(float64(l) / temp)
	}
	return result
}

func applyTopKCPU(logits []float32, k int) []float32 {
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

	for i := k; i < len(logits); i++ {
		logits[indices[i]] = float32(-math.Inf(1))
	}

	return logits
}

func applyTopPCPU(logits []float32, p float64) []float32 {
	probs := softmaxCPU(logits)
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

func softmaxCPU(logits []float32) []float32 {
	probs := make([]float32, len(logits))
	copy(probs, logits)
	simd.SoftmaxAVX2(probs)
	return probs
}

func sampleFromDistCPU(probs []float32, r *rand.Rand) int {
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

func (e *CPUEngine) ForwardDraft(tokens []int) ([][]float32, error) {
	if len(tokens) == 0 || e.weights == nil || (len(e.weights.TokenEmb) == 0 && e.weights.RawTokenEmb == nil) {
		return nil, nil
	}
	hiddenSize := e.config.Dim
	if len(e.weights.TokenEmb) > 0 && len(e.weights.TokenEmb[0]) > 0 {
		hiddenSize = len(e.weights.TokenEmb[0])
	}
	draftCount := 4
	drafts := make([][]float32, draftCount)
	for i := range drafts {
		drafts[i] = make([]float32, hiddenSize)
	}
	lastToken := tokens[len(tokens)-1]
	tokEmb := e.weights.GetTokenEmbedding(lastToken, hiddenSize)
	for i := 0; i < draftCount; i++ {
		copy(drafts[i], tokEmb)
	}
	return drafts, nil
}

func (e *CPUEngine) RollbackKV(seqID string, newPos int) error {
	if e.cache == nil {
		return fmt.Errorf("cache not initialized")
	}
	return e.cache.RollbackKV(seqID, newPos)
}

func (e *CPUEngine) LoadAdapter(path, id string) error {
	return nil // Not supported on CPU for now
}
