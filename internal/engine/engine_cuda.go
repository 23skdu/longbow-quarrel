//go:build linux && cuda

package engine

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"sort"
	"sync"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	cudaEngineInitialized = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "quarrel_cuda_engine_initialized_total",
		Help: "Total number of CUDA engine initializations",
	}, []string{"model", "architecture"})

	cudaEngineFailed = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "quarrel_cuda_engine_failed_total",
		Help: "Total number of CUDA engine initialization failures",
	}, []string{"model", "error_type"})

	cudaInferenceTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "quarrel_cuda_inference_total",
		Help: "Total number of CUDA inference calls",
	}, []string{"model"})

	cudaInferenceDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "quarrel_cuda_inference_duration_seconds",
		Help:    "Duration of CUDA inference calls",
		Buckets: []float64{0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0},
	}, []string{"model"})

	cudaTokensGenerated = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "quarrel_cuda_tokens_generated_total",
		Help: "Total number of tokens generated on CUDA",
	}, []string{"model"})

	cudaTokensPerSecond = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "quarrel_cuda_tokens_per_second",
		Help:    "Tokens generated per second",
		Buckets: []float64{10, 50, 100, 200, 500, 1000},
	}, []string{"model"})

	cudaLayerLatency = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "quarrel_cuda_layer_latency_seconds",
		Help:    "Latency per transformer layer",
		Buckets: []float64{0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1},
	}, []string{"model", "layer"})

	cudaMemoryUsage = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "quarrel_cuda_memory_bytes",
		Help: "Current CUDA memory usage",
	}, []string{"model"})

	cudaKVCacheHits = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "quarrel_cuda_kv_cache_hits_total",
		Help: "Total number of KV cache hits",
	}, []string{"model"})

	cudaKVCacheMisses = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "quarrel_cuda_kv_cache_misses_total",
		Help: "Total number of KV cache misses",
	}, []string{"model"})

	cudaDequantizationTime = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "quarrel_cuda_dequantization_seconds",
		Help:    "Time spent dequantizing weights",
		Buckets: []float64{0.0001, 0.001, 0.01, 0.1},
	}, []string{"model", "quantization_type"})

	cudaAttentionTime = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "quarrel_cuda_attention_seconds",
		Help:    "Time spent in attention computation",
		Buckets: []float64{0.0001, 0.001, 0.01, 0.1, 0.5},
	}, []string{"model"})

	cudaSamplingTime = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "quarrel_cuda_sampling_seconds",
		Help:    "Time spent in sampling",
		Buckets: []float64{0.00001, 0.0001, 0.001, 0.01},
	}, []string{"model"})

	cudaBatchSize = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "quarrel_cuda_batch_size",
		Help:    "Number of tokens in a batch",
		Buckets: []float64{1, 2, 4, 8, 16, 32},
	}, []string{"model"})
)

type cudaEngine struct {
	model        *gguf.GGUFFile
	ctx          *device.Context
	cuda         *CUDAStorage
	config       config.Config
	scratch      *CUDAScratch
	tok          *tokenizer.Tokenizer
	cache        *PagedKVCache
	BatchManager *ContinuousBatchManager
	stopChan     chan struct{}
	doneChan     chan struct{}
}

func NewcudaEngine(modelPath string, cfg config.Config) (Engine, error) {
	f, err := gguf.LoadFile(modelPath)
	if err != nil {
		cudaEngineFailed.WithLabelValues("unknown", "gguf_load_failed").Inc()
		return nil, fmt.Errorf("failed to load GGUF: %w", err)
	}

	arch := "unknown"
	if v, ok := f.KV["general.architecture"].(string); ok {
		arch = v
	}

	ctx, err := device.NewContext()
	if err != nil {
		f.Close()
		cudaEngineFailed.WithLabelValues(arch, "context_creation_failed").Inc()
		return nil, fmt.Errorf("failed to create CUDA context: %w", err)
	}

	cudaModel, err := ctx.NewCUDAModel(f, true, cfg.KVCacheSize)
	if err != nil {
		ctx.Free()
		f.Close()
		cudaEngineFailed.WithLabelValues(arch, "model_load_failed").Inc()
		return nil, fmt.Errorf("failed to load model to GPU: %w", err)
	}

	cache := &PagedKVCache{}
	if err := cache.Init(ctx, cfg); err != nil {
		ctx.Free()
		cudaModel.Free()
		f.Close()
		cudaEngineFailed.WithLabelValues(arch, "cache_init_failed").Inc()
		return nil, fmt.Errorf("failed to initialize paged cache: %w", err)
	}

	cudaEngineInitialized.WithLabelValues(arch, arch).Inc()
	cudaMemoryUsage.WithLabelValues(arch).Set(float64(device.CUDAAllocatedBytes()))

	tok, err := tokenizer.New(modelPath)
	if err != nil {
		log.Printf("Warning: failed to load tokenizer: %v", err)
	}

	layers := 1
	if v, ok := f.KV["llama.block_count"].(uint32); ok {
		layers = int(v)
	}

	vocabSize := 49152
	if v, ok := f.KV["llama.vocab_size"].(uint32); ok {
		vocabSize = int(v)
	}

	heads := 32
	if val, ok := getKV(f, "llama.attention.head_count", "gemma4.attention.head_count", "qwen2.attention.head_count", "qwen3moe.attention.head_count"); ok {
		heads = int(toFloat64(val))
	}

	kvHeads := heads
	if val, ok := getKV(f, "llama.attention.head_count_kv", "gemma4.attention.head_count_kv", "gemma4.attention.kv_head_count", "qwen3moe.attention.head_count_kv"); ok {
		if arr, ok := val.([]interface{}); ok {
			maxVal := 0
			for _, v := range arr {
				iv := int(toFloat64(v))
				if iv > maxVal {
					maxVal = iv
				}
			}
			kvHeads = maxVal
		} else {
			kvHeads = int(toFloat64(val))
		}
	}
	if kvHeads <= 0 {
		kvHeads = heads
	}

	dim := 2048
	if val, ok := getKV(f, "llama.embedding_length", "gemma4.embedding_length", "qwen2.embedding_length", "qwen3moe.embedding_length"); ok {
		dim = int(toFloat64(val))
	}

	headDim := dim / heads
	hiddenDim := dim * 4
	if val, ok := getKV(f, "llama.feed_forward_length", "gemma4.feed_forward_length", "qwen2.feed_forward_length", "qwen3moe.feed_forward_length"); ok {
		hiddenDim = int(toFloat64(val))
	}

	ropeTheta := 10000.0
	if val, ok := getKV(f, "llama.rope.freq_base", "qwen3moe.rope.freq_base", "gemma4.rope.freq_base", "qwen2.rope.freq_base"); ok {
		ropeTheta = toFloat64(val)
	}

	eps := float32(1e-5)
	if val, ok := getKV(f, "llama.attention.layer_norm_rms_epsilon", "qwen3moe.attention.layer_norm_rms_epsilon", "gemma4.attention.layer_norm_rms_epsilon"); ok {
		eps = float32(toFloat64(val))
	}

	seqLen := 2048
	if val, ok := getKV(f, "llama.context_length", "qwen3moe.context_length", "gemma4.context_length", "qwen2.context_length"); ok {
		seqLen = int(toFloat64(val))
	}

	isGemma4 := arch == "gemma4"
	e := &cudaEngine{
		model: f,
		ctx:   ctx,
		cuda:  cudaModel,
		config: config.Config{
			Architecture:  arch,
			Dim:           dim,
			HiddenDim:     hiddenDim,
			Layers:        layers,
			Heads:         heads,
			KVHeads:       kvHeads,
			HeadDim:       headDim,
			VocabSize:     vocabSize,
			SeqLen:        seqLen,
			Eps:           eps,
			RopeTheta:     float32(ropeTheta),
			PrecisionMode: config.PrecisionAuto,
			KVCacheSize:   cfg.KVCacheSize,
			IsGemma4:      isGemma4,
		},
		tok:          tok,
		cache:        cache,
		BatchManager: NewContinuousBatchManager(),
		stopChan:     make(chan struct{}),
		doneChan:     make(chan struct{}),
	}

	// Initialize scratch space
	e.scratch = ctx.NewLayerScratch(1, dim, hiddenDim, heads, kvHeads, headDim, seqLen, vocabSize, 0, 0)

	if isGemma4 {
		e.config.Gemma4SlidingWindowSize = 512
		e.config.Gemma4SlidingRoPETheta = 10000.0
		e.config.Gemma4FullRoPETheta = 1000000.0
		e.config.Gemma4PartialRoPEFactor = 0.25
		e.config.Gemma4SlidingHeadDim = 256
		e.config.Gemma4FullHeadDim = 512
	}

	logger.Log.Info("CUDA engine initialized with PagedKVCache", "model", modelPath, "heads", heads, "kv_heads", kvHeads)
	
	go e.runBatchLoop()

	return e, nil
}

func (e *cudaEngine) Config() config.Config {
	return e.config
}

func (e *cudaEngine) Close() {
	if e.scratch != nil {
		e.scratch.Free()
	}
	if e.cache != nil {
		e.cache.Free()
	}
	if e.cuda != nil {
		e.cuda.Free()
	}
	if e.ctx != nil {
		e.ctx.Free()
	}
	if e.model != nil {
		e.model.Close()
	}
}

func (e *cudaEngine) SwapModel(newModelPath string, newConfig config.Config) error {
	return fmt.Errorf("SwapModel not implemented for refactored CUDA engine")
}

func (e *cudaEngine) Infer(inputTokens []int, tokensToGenerate int, samplerConfig SamplerConfig) ([]int, error) {
	return e.inferInternal(inputTokens, tokensToGenerate, samplerConfig, nil, nil)
}

func (e *cudaEngine) InferWithLogits(inputTokens []int, tokensToGenerate int, samplerConfig SamplerConfig) ([]int, []float32, error) {
	var lastLogits []float32
	result, err := e.inferInternal(inputTokens, tokensToGenerate, samplerConfig, nil, func(l []float32) {
		lastLogits = l
	})
	return result, lastLogits, err
}

func (e *cudaEngine) InferWithCallback(inputTokens []int, tokensToGenerate int, samplerConfig SamplerConfig, callback func(int)) ([]int, error) {
	return e.inferInternal(inputTokens, tokensToGenerate, samplerConfig, callback, nil)
}

func (e *cudaEngine) InferWithCallbackLogits(inputTokens []int, tokensToGenerate int, samplerConfig SamplerConfig, tokenCallback func(int), logitsCallback func([]float32)) ([]int, error) {
	return e.inferInternal(inputTokens, tokensToGenerate, samplerConfig, tokenCallback, logitsCallback)
}

func (e *cudaEngine) runBatchLoop() {
	defer close(e.doneChan)
	for {
		select {
		case <-e.stopChan:
			return
		default:
		}

		// 1. Pull active sequences
		desc, _ := e.BatchManager.Step(16, e.cache, nil)
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
		for i, seq := range desc.Sequences {
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

func (e *cudaEngine) ForwardBatch(desc *BatchDescriptor) ([]*device.Tensor, error) {
	// For CUDA, we'll implement a fallback batch processing similar to CPU for the first iteration.
	// This ensures the batching loop works. Real performance comes from batched kernels in the next phase.
	batchSize := len(desc.Sequences)
	results := make([]*device.Tensor, batchSize)

	for i, seq := range desc.Sequences {
		start := desc.Offsets[i]
		var end int
		if i < batchSize - 1 {
			end = desc.Offsets[i+1]
		} else {
			end = len(desc.Tokens)
		}
		
		seqTokens := desc.Tokens[start:end]
		
		// Run existing forward
		logits, err := e.forward(seqTokens)
		if err != nil {
			return nil, err
		}
		
		res := e.ctx.NewTensorWithType(1, e.config.VocabSize, device.DataTypeF32)
		res.LoadFrom(logits)
		results[i] = res
	}

	return results, nil
}

func (e *cudaEngine) inferInternal(inputTokens []int, tokensToGenerate int, samplerConfig SamplerConfig, tokenCallback func(int), logitsCallback func([]float32)) ([]int, error) {
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

	result := make([]int, 0, tokensToGenerate)
	sampler := NewSampler(samplerConfig)
	startTime := time.Now()

	cudaInferenceTotal.WithLabelValues(e.config.Architecture).Inc()
	cudaBatchSize.WithLabelValues(e.config.Architecture).Observe(float64(len(inputTokens)))

	allTokens := append([]int{}, inputTokens...)
	var logits []float32
	var err error

	// Prompt processing
	for pos, token := range inputTokens {
		logits, err = e.forward(token, pos, allTokens)
		if err != nil {
			return nil, fmt.Errorf("forward pass failed at prompt pos %d: %w", pos, err)
		}
	}

	// Generation loop
	for gen := 0; gen < tokensToGenerate; gen++ {
		if logitsCallback != nil {
			logitsCallback(logits)
		}

		nextToken := sampler.Sample(logits, allTokens)
		allTokens = append(allTokens, nextToken)
		result = append(result, nextToken)
		cudaTokensGenerated.WithLabelValues(e.config.Architecture).Inc()

		if tokenCallback != nil {
			tokenCallback(nextToken)
		}

		if len(allTokens) >= e.config.SeqLen {
			break
		}

		logits, err = e.forward(nextToken, len(allTokens)-1, allTokens)
		if err != nil {
			return nil, fmt.Errorf("forward pass failed at gen step %d: %w", gen, err)
		}
	}

	elapsed := time.Since(startTime)
	log.Printf("Generated %d tokens in %.2fs (%.1f t/s)", len(result), elapsed.Seconds(), float64(len(result))/elapsed.Seconds())

	return result, nil
}

func (e *cudaEngine) forward(token int, pos int, allTokens []int) ([]float32, error) {
	ctx := e.cuda.Ctx
	dim := e.config.Dim
	hiddenDim := e.config.HiddenDim
	heads := e.config.Heads
	kvHeads := e.config.KVHeads
	headDim := e.config.HeadDim
	ropeTheta := e.config.RopeTheta
	eps := e.config.Eps

	// 1. Initial Embedding (on GPU)
	hidden, err := e.cuda.GetEmbeddingTensor(token)
	if err != nil {
		return nil, fmt.Errorf("failed to get embedding: %w", err)
	}
	defer hidden.ReturnToPool()

	// 2. Transformer Layer Loop
	for layer := 0; layer < e.config.Layers; layer++ {
		// --- Attention Sublayer ---
		// RMSNorm
		attnNormW, _ := e.cuda.GetWeightTensor(fmt.Sprintf("blk.%d.attn_norm.weight", layer))
		normed := e.scratch.Normed
		ctx.RMSNorm(hidden, attnNormW, normed, 1, dim, eps)

		// Q, K, V Projections
		qW, _ := e.cuda.GetWeightTensor(fmt.Sprintf("blk.%d.attn_q.weight", layer))
		kW, _ := e.cuda.GetWeightTensor(fmt.Sprintf("blk.%d.attn_k.weight", layer))
		vW, _ := e.cuda.GetWeightTensor(fmt.Sprintf("blk.%d.attn_v.weight", layer))

		q, _ := ctx.MatmulF16(normed, qW)
		k, _ := ctx.MatmulF16(normed, kW)
		v, _ := ctx.MatmulF16(normed, vW)

		// RoPE
		ctx.FusedRoPE(q, []int{pos}, 1, heads, 1, headDim, ropeTheta)
		ctx.FusedRoPE(k, []int{pos}, 1, kvHeads, 1, headDim, ropeTheta)

		// Fused Attention
		kCache := e.cuda.GetKCache(layer)
		vCache := e.cuda.GetVCache(layer)
		attnOut := e.scratch.Attn

		windowSize := 0
		if e.config.IsGemma4 && (layer%6) != 5 {
			windowSize = e.config.Gemma4SlidingWindowSize
		}

		scale := float32(1.0 / math.Sqrt(float64(headDim)))
		ctx.FusedAttention(q, k, v, attnOut, kCache, vCache, 1, heads, 1, pos+1, headDim, scale, 1, windowSize)

		// Cleanup intermediate projections
		q.ReturnToPool()
		k.ReturnToPool()
		v.ReturnToPool()

		// Output Projection
		oW, _ := e.cuda.GetWeightTensor(fmt.Sprintf("blk.%d.attn_output.weight", layer))
		attnProj, _ := ctx.MatmulF16(attnOut, oW)

		// Residual Add
		ctx.Add(hidden, attnProj, hidden, dim)
		attnProj.ReturnToPool()

		// --- MLP Sublayer ---
		// RMSNorm
		ffnNormW, _ := e.cuda.GetWeightTensor(fmt.Sprintf("blk.%d.ffn_norm.weight", layer))
		ctx.RMSNorm(hidden, ffnNormW, normed, 1, dim, eps)

		// Fused MLP
		ffnGateW, _ := e.cuda.GetWeightTensor(fmt.Sprintf("blk.%d.ffn_gate.weight", layer))
		ffnUpW, _ := e.cuda.GetWeightTensor(fmt.Sprintf("blk.%d.ffn_up.weight", layer))
		ffnDownW, _ := e.cuda.GetWeightTensor(fmt.Sprintf("blk.%d.ffn_down.weight", layer))
		
		mlpOut := e.scratch.Down
		ctx.FusedMLP(normed, ffnGateW, ffnUpW, ffnDownW, mlpOut, 1, dim, hiddenDim)

		// Residual Add
		ctx.Add(hidden, mlpOut, hidden, dim)
	}

	// 3. Final Output Layer
	outputNormW, _ := e.cuda.GetWeightTensor("output_norm.weight")
	normedFinal := e.scratch.Normed
	ctx.RMSNorm(hidden, outputNormW, normedFinal, 1, dim, eps)

	outputW, _ := e.cuda.GetWeightTensor("output.weight")
	if outputW == nil {
		outputW, _ = e.cuda.GetWeightTensor("token_embd.weight")
	}

	logitsTensor, err := ctx.MatmulF16(normedFinal, outputW)
	if err != nil {
		return nil, fmt.Errorf("failed to compute logits: %w", err)
	}
	defer logitsTensor.ReturnToPool()

	ctx.Synchronize()
	return logitsTensor.ToHostF32(), nil
}

type Sampler struct {
	config SamplerConfig
	rng    *rand.Rand
}

func NewSampler(config SamplerConfig) *Sampler {
	seed := config.Seed
	if seed == 0 {
		seed = time.Now().UnixNano()
	}
	return &Sampler{
		config: config,
		rng:    rand.New(rand.NewSource(seed)),
	}
}

func (s *Sampler) Sample(logits []float32, history []int) int {
	if len(logits) == 0 {
		return 0
	}

	// Repetition penalty
	if s.config.RepPenalty > 1.0 && len(history) > 0 {
		seen := make(map[int]bool)
		for _, t := range history {
			seen[t] = true
		}
		for t := range seen {
			if t < len(logits) {
				if logits[t] > 0 {
					logits[t] /= float32(s.config.RepPenalty)
				} else {
					logits[t] *= float32(s.config.RepPenalty)
				}
			}
		}
	}

	// Temperature=0: Greedy
	if s.config.Temperature <= 0 {
		maxIdx := 0
		maxVal := logits[0]
		for i, v := range logits {
			if v > maxVal {
				maxVal = v
				maxIdx = i
			}
		}
		return maxIdx
	}

	// Temperature scaling
	for i := range logits {
		logits[i] /= float32(s.config.Temperature)
	}

	// Softmax
	maxLogit := logits[0]
	for _, v := range logits {
		if v > maxLogit {
			maxLogit = v
		}
	}
	sum := float64(0)
	probs := make([]float64, len(logits))
	for i, v := range logits {
		probs[i] = math.Exp(float64(v - maxLogit))
		sum += probs[i]
	}
	for i := range probs {
		probs[i] /= sum
	}

	// Top-K
	if s.config.TopK > 0 && s.config.TopK < len(probs) {
		type score struct {
			idx int
			val float64
		}
		scores := make([]score, len(probs))
		for i, v := range probs {
			scores[i] = score{i, v}
		}
		sort.Slice(scores, func(i, j int) bool { return scores[i].val > scores[j].val })
		
		sumK := 0.0
		for i := 0; i < s.config.TopK; i++ {
			sumK += scores[i].val
		}
		r := s.rng.Float64() * sumK
		acc := 0.0
		for i := 0; i < s.config.TopK; i++ {
			acc += scores[i].val
			if r <= acc {
				return scores[i].idx
			}
		}
		return scores[0].idx
	}

	// Basic sampling
	r := s.rng.Float64()
	acc := 0.0
	for i, v := range probs {
		acc += v
		if r <= acc {
			return i
		}
	}
	return 0
}

func (e *cudaEngine) ForwardDraft(tokens []int) ([][]float32, error) {
	// Stub for speculative decoding support
	return nil, nil
}

func (e *cudaEngine) GetSeqCachePos(seqID string) int {
	// For CUDA engine, we currently don't support paged cache with multiple sequences.
	// We'll return 0 for now to satisfy the interface.
	return 0
}

func (e *cudaEngine) RollbackKV(seqID string, newPos int) error {
	if e.cache == nil {
		return fmt.Errorf("cache not initialized")
	}
	return e.cache.RollbackKV(seqID, newPos)
}

func init() {
	RegisterEngine("cuda", NewcudaEngine)
}

