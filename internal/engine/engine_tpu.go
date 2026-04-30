//go:build linux && tpu

package engine

import (
	"errors"
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
	tpuEngineInitialized = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "quarrel_tpu_engine_initialized_total",
		Help: "Total number of TPU engine initializations",
	}, []string{"model", "architecture"})

	tpuEngineFailed = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "quarrel_tpu_engine_failed_total",
		Help: "Total number of TPU engine initialization failures",
	}, []string{"model", "error_type"})

	tpuInferenceTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "quarrel_tpu_inference_total",
		Help: "Total number of TPU inference calls",
	}, []string{"model"})

	tpuInferenceDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "quarrel_tpu_inference_duration_seconds",
		Help:    "Duration of TPU inference calls",
		Buckets: []float64{0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0},
	}, []string{"model"})

	tpuTokensGenerated = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "quarrel_tpu_tokens_generated_total",
		Help: "Total number of tokens generated on TPU",
	}, []string{"model"})

	tpuTokensPerSecond = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "quarrel_tpu_tokens_per_second",
		Help:    "Tokens generated per second",
		Buckets: []float64{10, 50, 100, 200, 500, 1000},
	}, []string{"model"})

	tpuLayerLatency = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "quarrel_tpu_layer_latency_seconds",
		Help:    "Latency per transformer layer",
		Buckets: []float64{0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1},
	}, []string{"model", "layer"})

	tpuMemoryUsage = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "quarrel_tpu_memory_bytes",
		Help: "Current TPU memory usage",
	}, []string{"model"})

	tpuDequantizationTime = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "quarrel_tpu_dequantization_seconds",
		Help:    "Time spent dequantizing weights",
		Buckets: []float64{0.0001, 0.001, 0.01, 0.1},
	}, []string{"model", "quantization_type"})

	tpuAttentionTime = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "quarrel_tpu_attention_seconds",
		Help:    "Time spent in attention computation",
		Buckets: []float64{0.0001, 0.001, 0.01, 0.1, 0.5},
	}, []string{"model"})

	tpuSamplingTime = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "quarrel_tpu_sampling_seconds",
		Help:    "Time spent in sampling",
		Buckets: []float64{0.00001, 0.0001, 0.001, 0.01},
	}, []string{"model"})

	tpuBatchSize = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "quarrel_tpu_batch_size",
		Help:    "Number of tokens in a batch",
		Buckets: []float64{1, 2, 4, 8, 16, 32},
	}, []string{"model"})
)

type tpuEngine struct {
	model        *gguf.GGUFFile
	ctx          *device.Context
	tpu          *device.TPUModel
	config       config.Config
	scratch      *device.Tensor
	tok          *tokenizer.Tokenizer
	cache        *PagedKVCache
	BatchManager *ContinuousBatchManager
	stopChan     chan struct{}
	doneChan     chan struct{}
	mu           sync.RWMutex
}

func NewTPUEngine(modelPath string, cfg config.Config) (Engine, error) {
	f, err := gguf.LoadFile(modelPath)
	if err != nil {
		tpuEngineFailed.WithLabelValues("unknown", "gguf_load_failed").Inc()
		return nil, fmt.Errorf("failed to load GGUF: %w", err)
	}

	arch := "unknown"
	if v, ok := f.KV["general.architecture"].(string); ok {
		arch = v
	}

	ctx, err := device.NewTPUContext()
	if err != nil {
		f.Close()
		tpuEngineFailed.WithLabelValues(arch, "context_creation_failed").Inc()
		return nil, fmt.Errorf("failed to create TPU context: %w", err)
	}

	tpuModel, err := ctx.LoadTPUModel(f, true, cfg.KVCacheSize)
	if err != nil {
		ctx.Free()
		f.Close()
		tpuEngineFailed.WithLabelValues(arch, "model_load_failed").Inc()
		return nil, fmt.Errorf("failed to load model to TPU: %w", err)
	}

	cache := &PagedKVCache{}
	if err := cache.Init(ctx, cfg); err != nil {
		ctx.Free()
		tpuModel.Free()
		f.Close()
		tpuEngineFailed.WithLabelValues(arch, "cache_init_failed").Inc()
		return nil, fmt.Errorf("failed to initialize paged cache: %w", err)
	}

	tpuEngineInitialized.WithLabelValues(arch, arch).Inc()
	tpuMemoryUsage.WithLabelValues(arch).Set(float64(device.TPUAllocatedBytes()))

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

	e := &tpuEngine{
		model: f,
		ctx:   ctx,
		tpu:   tpuModel,
		config: config.Config{
			Architecture: arch,
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
		},
		tok:          tok,
		cache:        cache,
		BatchManager: NewContinuousBatchManager(),
		stopChan:     make(chan struct{}),
		doneChan:    make(chan struct{}),
	}

	logger.Log.Info("TPU engine initialized", "model", modelPath, "heads", heads, "kv_heads", kvHeads)

	go e.runBatchLoop()

	return e, nil
}

func (e *tpuEngine) Config() config.Config {
	return e.config
}

func (e *tpuEngine) Close() {
	if e.stopChan != nil {
		close(e.stopChan)
		<-e.doneChan
	}
	if e.cache != nil {
		e.cache.Free()
	}
	if e.ctx != nil {
		e.ctx.Free()
	}
	if e.model != nil {
		e.model.Close()
	}
}

func (e *tpuEngine) SwapModel(newModelPath string, newConfig config.Config) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.stopChan != nil {
		close(e.stopChan)
		<-e.doneChan
	}

	if e.cache != nil {
		e.cache.Free()
		e.cache = nil
	}

	if e.ctx != nil {
		e.ctx.Free()
		e.ctx = nil
	}

	if e.model != nil {
		e.model.Close()
		e.model = nil
	}

	f, err := gguf.LoadFile(newModelPath)
	if err != nil {
		return fmt.Errorf("failed to load new GGUF: %w", err)
	}
	e.model = f

	ctx, err := device.NewTPUContext()
	if err != nil {
		return fmt.Errorf("failed to create TPU context: %w", err)
	}
	e.ctx = ctx

	e.config = newConfig

	stopChan := make(chan struct{})
	doneChan := make(chan struct{})
	e.stopChan = stopChan
	e.doneChan = doneChan
	go e.runBatchLoop()

	return nil
}

func (e *tpuEngine) Infer(inputTokens []int, tokensToGenerate int, samplerConfig SamplerConfig) ([]int, error) {
	return e.inferInternal(inputTokens, tokensToGenerate, samplerConfig, nil, nil)
}

func (e *tpuEngine) InferWithLogits(inputTokens []int, tokensToGenerate int, samplerConfig SamplerConfig) ([]int, []float32, error) {
	var lastLogits []float32
	result, err := e.inferInternal(inputTokens, tokensToGenerate, samplerConfig, nil, func(l []float32) {
		lastLogits = l
	})
	return result, lastLogits, err
}

func (e *tpuEngine) InferWithCallback(inputTokens []int, tokensToGenerate int, samplerConfig SamplerConfig, callback func(int)) ([]int, error) {
	return e.inferInternal(inputTokens, tokensToGenerate, samplerConfig, callback, nil)
}

func (e *tpuEngine) InferWithCallbackLogits(inputTokens []int, tokensToGenerate int, samplerConfig SamplerConfig, tokenCallback func(int), logitsCallback func([]float32)) ([]int, error) {
	return e.inferInternal(inputTokens, tokensToGenerate, samplerConfig, tokenCallback, logitsCallback)
}

func (e *tpuEngine) runBatchLoop() {
	defer close(e.doneChan)
	for {
		select {
		case <-e.stopChan:
			return
		default:
		}

		desc, _ := e.BatchManager.Step(16, e.cache, nil)
		if desc == nil || len(desc.Sequences) == 0 {
			time.Sleep(10 * time.Millisecond)
			continue
		}

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

		for i, seq := range desc.Sequences {
			logits := results[i].ToHostF32()
			results[i].Free()

			if seq.LogitsCallback != nil {
				seq.LogitsCallback(logits)
			}

			sampler := NewSampler(seq.Config)
			token := sampler.Sample(logits, seq.Tokens)

			seq.Tokens = append(seq.Tokens, token)
			seq.Pos++

			if seq.TokenCallback != nil {
				seq.TokenCallback(token)
			}

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

func (e *tpuEngine) ForwardBatch(desc *BatchDescriptor) ([]*device.Tensor, error) {
	ctx := e.ctx
	batchSize := len(desc.Sequences)
	numTokens := len(desc.Tokens)

	tokenPosTensor := ctx.NewTensor(1, numTokens)
	tokenPositions := make([]float32, numTokens)
	for i, start := range desc.Offsets {
		end := numTokens
		if i < batchSize-1 {
			end = desc.Offsets[i+1]
		}
		for j := 0; j < end-start; j++ {
			tokenPositions[start+j] = float32(desc.ContextLens[i] + j)
		}
	}
	if err := tokenPosTensor.LoadFrom(tokenPositions); err != nil {
		return nil, nil, err
	}
	defer tokenPosTensor.ReturnToPool()

	inputTokensTensor := ctx.NewTensor(1, numTokens)
	inputTokensF := make([]float32, numTokens)
	for i, t := range desc.Tokens {
		inputTokensF[i] = float32(t)
	}
	if err := inputTokensTensor.LoadFrom(inputTokensF); err != nil {
		return nil, nil, err
	}
	defer inputTokensTensor.ReturnToPool()

	hidden, _ := ctx.MatmulF16(inputTokensTensor, e.tpu.TokenEmb)
	defer hidden.ReturnToPool()

	blockTableTensor := ctx.NewTensor(batchSize, 0)

	eps := e.config.Eps
	dim := e.config.Dim
	heads := e.config.Heads
	kvHeads := e.config.KVHeads
	headDim := e.config.HeadDim
	ropeTheta := e.config.RopeTheta
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	for layer := 0; layer < e.config.Layers; layer++ {
		attnNormW, _ := e.ctx.GetWeightTensor(fmt.Sprintf("blk.%d.attn_norm.weight", layer))
		normed := ctx.NewTensor(numTokens, dim)
		ctx.RMSNorm(hidden, attnNormW, normed, numTokens, dim, eps)

		qW, _ := e.ctx.GetWeightTensor(fmt.Sprintf("blk.%d.attn_q.weight", layer))
		kW, _ := e.ctx.GetWeightTensor(fmt.Sprintf("blk.%d.attn_k.weight", layer))
		vW, _ := e.ctx.GetWeightTensor(fmt.Sprintf("blk.%d.attn_v.weight", layer))

		q, _ := ctx.MatmulF16(normed, qW)
		k, _ := ctx.MatmulF16(normed, kW)
		v, _ := ctx.MatmulF16(normed, vW)
		normed.ReturnToPool()

		ctx.FusedRoPE(q, []int{posToInt(desc.ContextLens)}, batchSize, heads, 1, headDim, ropeTheta)
		ctx.FusedRoPE(k, []int{posToInt(desc.ContextLens)}, batchSize, kvHeads, 1, headDim, ropeTheta)

		kCache := e.tpu.KCache[layer]
		vCache := e.tpu.VCache[layer]
		attnOut := ctx.NewTensor(numTokens, dim)

		posData := make([]int, numTokens)
		for i := range posData {
			posData[i] = i
		}

		ctx.StoreKV(k, v, kCache, vCache, posData, numTokens)
		ctx.FusedAttention(q, k, v, attnOut, kCache, vCache, batchSize, heads, 1, numTokens, headDim, scale, 0)

		q.ReturnToPool()
		k.ReturnToPool()
		v.ReturnToPool()

		oW, _ := e.ctx.GetWeightTensor(fmt.Sprintf("blk.%d.attn_output.weight", layer))
		attnProj, _ := ctx.MatmulF16(attnOut, oW)
		attnOut.ReturnToPool()

		ctx.Add(hidden, attnProj, hidden, numTokens*dim)
		attnProj.ReturnToPool()
	}

	lastHiddenIndices := make([]int, batchSize)
	for i := 0; i < batchSize; i++ {
		lastHiddenIndices[i] = numTokens - 1
		if i < batchSize-1 {
			lastHiddenIndices[i] = desc.Offsets[i+1] - 1
		}
	}

	lastHidden := ctx.NewTensor(batchSize, dim)
	hidden.ReturnToPool()

	outputNormW, _ := e.ctx.GetWeightTensor("output_norm.weight")
	normedFinal := ctx.NewTensor(batchSize, dim)
	ctx.RMSNorm(hidden, outputNormW, normedFinal, batchSize, dim, eps)
	hidden.ReturnToPool()

	outputW, _ := e.ctx.GetWeightTensor("output.weight")
	if outputW == nil {
		outputW, _ = e.ctx.GetWeightTensor("token_embd.weight")
	}

	logitsTensor, _ := ctx.MatmulF16(normedFinal, outputW)
	normedFinal.ReturnToPool()

	results := make([]*device.Tensor, batchSize)
	for i := 0; i < batchSize; i++ {
		res := ctx.NewTensor(1, e.config.VocabSize)
		results[i] = res
	}
	logitsTensor.ReturnToPool()

	return results, nil
}

func posToInt(positions []int) int {
	if len(positions) == 0 {
		return 0
	}
	return positions[0]
}

func (e *tpuEngine) inferInternal(inputTokens []int, tokensToGenerate int, samplerConfig SamplerConfig, tokenCallback func(int), logitsCallback func([]float32)) ([]int, error) {
	resChan := make(chan []int, 1)
	errChan := make(chan error, 1)

	req := &InferenceRequest{
		ID:            uint64(time.Now().UnixNano()),
		Prompt:       inputTokens,
		MaxTokens:     len(inputTokens) + tokensToGenerate,
		Config:        samplerConfig,
		Result:        resChan,
		Err:           errChan,
		TokenCallback: tokenCallback,
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

func (e *tpuEngine) forward(token int, pos int, allTokens []int) ([]float32, error) {
	ctx := e.ctx
	dim := e.config.Dim
	hiddenDim := e.config.HiddenDim
	heads := e.config.Heads
	kvHeads := e.config.KVHeads
	headDim := e.config.HeadDim
	ropeTheta := e.config.RopeTheta
	eps := e.config.Eps

	embTensor := ctx.NewTensor(1, dim)
	if err := embTensor.LoadFrom(float32(token)); err != nil {
		return nil, fmt.Errorf("failed to create embedding tensor: %w", err)
	}
	defer embTensor.ReturnToPool()

	hidden, err := ctx.MatmulF16(embTensor, e.tpu.TokenEmb)
	if err != nil {
		return nil, fmt.Errorf("failed to get embedding: %w", err)
	}
	defer hidden.ReturnToPool()

	for layer := 0; layer < e.config.Layers; layer++ {
		attnNormW, _ := e.ctx.GetWeightTensor(fmt.Sprintf("blk.%d.attn_norm.weight", layer))
		normed := ctx.NewTensor(1, dim)
		ctx.RMSNorm(hidden, attnNormW, normed, 1, dim, eps)

		qW, _ := e.ctx.GetWeightTensor(fmt.Sprintf("blk.%d.attn_q.weight", layer))
		kW, _ := e.ctx.GetWeightTensor(fmt.Sprintf("blk.%d.attn_k.weight", layer))
		vW, _ := e.ctx.GetWeightTensor(fmt.Sprintf("blk.%d.attn_v.weight", layer))

		q, _ := ctx.MatmulF16(normed, qW)
		k, _ := ctx.MatmulF16(normed, kW)
		v, _ := ctx.MatmulF16(normed, vW)
		normed.ReturnToPool()

		ctx.FusedRoPE(q, []int{pos}, 1, heads, 1, headDim, ropeTheta)
		ctx.FusedRoPE(k, []int{pos}, 1, kvHeads, 1, headDim, ropeTheta)

		kCache := e.tpu.KCache[layer]
		vCache := e.tpu.VCache[layer]
		attnOut := ctx.NewTensor(1, dim)

		scale := float32(1.0 / math.Sqrt(float64(headDim)))
		ctx.FusedAttention(q, k, v, attnOut, kCache, vCache, 1, heads, 1, pos+1, headDim, scale, 0)

		q.ReturnToPool()
		k.ReturnToPool()
		v.ReturnToPool()

		oW, _ := e.ctx.GetWeightTensor(fmt.Sprintf("blk.%d.attn_output.weight", layer))
		attnProj, _ := ctx.MatmulF16(attnOut, oW)
		attnOut.ReturnToPool()

		ctx.Add(hidden, attnProj, hidden, dim)
		attnProj.ReturnToPool()

		ffnNormW, _ := e.ctx.GetWeightTensor(fmt.Sprintf("blk.%d.ffn_norm.weight", layer))
		ctx.RMSNorm(hidden, ffnNormW, normed, 1, dim, eps)

		ffnGateW, _ := e.ctx.GetWeightTensor(fmt.Sprintf("blk.%d.ffn_gate.weight", layer))
		ffnUpW, _ := e.ctx.GetWeightTensor(fmt.Sprintf("blk.%d.ffn_up.weight", layer))
		ffnDownW, _ := e.ctx.GetWeightTensor(fmt.Sprintf("blk.%d.ffn_down.weight", layer))

		mlpOut := ctx.NewTensor(1, hiddenDim)
		ctx.FusedMLP(normed, ffnGateW, ffnUpW, ffnDownW, mlpOut, 1, dim, hiddenDim)
		normed.ReturnToPool()

		ctx.Add(hidden, mlpOut, hidden, dim)
		mlpOut.ReturnToPool()
	}

	outputNormW, _ := e.ctx.GetWeightTensor("output_norm.weight")
	normedFinal := ctx.NewTensor(1, dim)
	ctx.RMSNorm(hidden, outputNormW, normedFinal, 1, dim, eps)
	hidden.ReturnToPool()

	outputW, _ := e.ctx.GetWeightTensor("output.weight")
	if outputW == nil {
		outputW, _ = e.ctx.GetWeightTensor("token_embd.weight")
	}

	logitsTensor, err := ctx.MatmulF16(normedFinal, outputW)
	if err != nil {
		return nil, fmt.Errorf("failed to compute logits: %w", err)
	}
	defer logitsTensor.ReturnToPool()

	ctx.Synchronize()
	return logitsTensor.ToHostF32(), nil
}

func (e *tpuEngine) ForwardDraft(tokens []int) ([][]float32, error) {
	return nil, errors.New("ForwardDraft not implemented for TPU")
}

func (e *tpuEngine) GetSeqCachePos(seqID string) int {
	return 0
}

func (e *tpuEngine) RollbackKV(seqID string, newPos int) error {
	if e.cache == nil {
		return errors.New("cache not initialized")
	}
	return e.cache.RollbackKV(seqID, newPos)
}

func init() {
	RegisterEngine("tpu", NewTPUEngine)
}