//go:build !cuda && !tpu && !metal

package engine

import (
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
}

func init() {
	RegisterEngine("cpu", NewCPUEngine)
}

func NewCPUEngine(modelPath string, cfg config.Config) (Engine, error) {
	f, err := gguf.LoadFile(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load GGUF: %w", err)
	}

	// Load Dimensions from GGUF into config
	if v, ok := f.KV["llama.block_count"].(uint32); ok {
		cfg.Layers = int(v)
	}
	if v, ok := f.KV["llama.embedding_length"].(uint32); ok {
		cfg.Dim = int(v)
	}
	if v, ok := f.KV["llama.attention.head_count"].(uint32); ok {
		cfg.Heads = int(v)
	}
	if v, ok := f.KV["llama.attention.head_count_kv"].(uint32); ok {
		cfg.KVHeads = int(v)
	} else {
		cfg.KVHeads = cfg.Heads // Default MHA
	}
	if cfg.Heads > 0 {
		cfg.HeadDim = cfg.Dim / cfg.Heads
	}

	weights, err := loadCPUWeights(f)
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
			if len(w.Output) == 0 {
				w.Output = data
			}
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
	vocabSize := 0
	if len(e.weights.TokenEmb) > 0 && len(e.weights.TokenEmb[0]) > 0 {
		vocabSize = len(e.weights.TokenEmb[0]) / hiddenSize
	}

	hidden := make([]float32, hiddenSize)

	if len(tokens) > 0 && vocabSize > 0 {
		lastToken := tokens[len(tokens)-1]
		if lastToken < vocabSize {
			startIdx := lastToken * hiddenSize
			for i := 0; i < hiddenSize && startIdx+i < len(e.weights.TokenEmb[0]); i++ {
				hidden[i] = e.weights.TokenEmb[0][startIdx+i]
			}
		}
	}

	for layerIdx := 0; layerIdx < e.config.Layers; layerIdx++ {
		hidden = e.applyLayerCPU(hidden, layerIdx)
	}

	if e.weights.OutputNorm != nil {
		hidden = rmsNormCPU(hidden, e.weights.OutputNorm, e.config.Eps)
	}

	if e.weights.Output != nil {
		logits := matMulVec(e.weights.Output, hidden)
		return logits
	}

	return hidden
}

func (e *CPUEngine) applyLayerCPU(input []float32, layerIdx int) []float32 {
	normed := rmsNormCPU(input, e.weights.AttnNorm[layerIdx], e.config.Eps)

	q := matMulVec(e.weights.AttnQ[layerIdx], normed)
	k := matMulVec(e.weights.AttnK[layerIdx], normed)
	v := matMulVec(e.weights.AttnV[layerIdx], normed)

	attn := attentionCPU(q, k, v, e.config.Heads, e.config.KVHeads, e.config.HeadDim)

	out := matMulVec(e.weights.AttnO[layerIdx], attn)

	residual := make([]float32, len(input))
	for i := range residual {
		residual[i] = input[i] + out[i]
	}

	normedFFN := rmsNormCPU(residual, e.weights.FfnNorm[layerIdx], e.config.Eps)

	gate := matMulVec(e.weights.FfnGate[layerIdx], normedFFN)
	up := matMulVec(e.weights.FfnUp[layerIdx], normedFFN)

	swiGLU := make([]float32, len(gate))
	for i := range swiGLU {
		swiGLU[i] = gate[i] * sigmoid(gate[i]) * up[i]
	}

	down := matMulVec(e.weights.FfnDown[layerIdx], swiGLU)

	result := make([]float32, len(residual))
	for i := range result {
		result[i] = residual[i] + down[i]
	}

	return result
}

func rmsNormCPU(input, weight []float32, eps float32) []float32 {
	result := make([]float32, len(input))
	sum := 0.0
	for _, v := range input {
		sum += float64(v * v)
	}
	sum /= float64(len(input))
	sum += float64(eps)
	norm := 1.0 / math.Sqrt(sum)

	for i := range result {
		result[i] = float32(float64(input[i]) * norm * float64(weight[i]))
	}
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

func attentionCPU(q, k, v []float32, numHeads, kvHeads, headDim int) []float32 {
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
				if j == i || score > maxScore {
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
	if len(tokens) == 0 || e.weights == nil || len(e.weights.TokenEmb) == 0 {
		return nil, nil
	}
	hiddenSize := len(e.weights.TokenEmb[0])
	draftCount := 4
	drafts := make([][]float32, draftCount)
	for i := range drafts {
		drafts[i] = make([]float32, hiddenSize)
	}
	lastToken := tokens[len(tokens)-1]
	if lastToken < len(e.weights.TokenEmb) {
		for i := 0; i < draftCount; i++ {
			copy(drafts[i], e.weights.TokenEmb[lastToken])
		}
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
