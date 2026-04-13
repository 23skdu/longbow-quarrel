//go:build linux && cuda

package engine

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"sort"
	"sync"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
	"github.com/23skdu/longbow-quarrel/internal/metrics"
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
	model            *gguf.GGUFFile
	tokenizer        *tokenizer.Tokenizer
	config           config.Config
	cuda             *device.CUDAModel
	scratch          *device.LayerScratch
	dequantizedCache map[string]*device.CUDATensor
	mu               sync.RWMutex
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

	ctx, err := device.NewCUDAContext()
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

	cudaEngineInitialized.WithLabelValues(arch, arch).Inc()
	cudaMemoryUsage.WithLabelValues(arch).Set(float64(device.CUDAAllocatedBytes()))

	tok, err := tokenizer.New(modelPath)
	if err != nil {
		log.Printf("Warning: failed to load tokenizer: %v", err)
	}

	arch = "unknown"
	if v, ok := f.KV["general.architecture"].(string); ok {
		arch = v
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

	log.Printf("=== CUDA Engine ===")
	log.Printf("Architecture: %s", arch)
	log.Printf("Layers: %d, Dim: %d, Heads: %d, HeadDim: %d, KVHeads: %d", layers, dim, heads, headDim, kvHeads)
	log.Printf("Vocab: %d, HiddenDim: %d", vocabSize, hiddenDim)
	log.Printf("RoPE Theta: %.0f, Eps: %e", ropeTheta, eps)
	log.Printf("GPU Memory: %.1f MB", float64(device.CUDAAllocatedBytes())/1e6)

	// Detect Gemma4 architecture
	isGemma4 := arch == "gemma4"
	if isGemma4 {
		log.Printf("Gemma4 architecture detected - enabling hybrid attention")
	}

	e := &cudaEngine{
		model:            f,
		tokenizer:        tok,
		dequantizedCache: make(map[string]*device.CUDATensor),
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
		cuda: cudaModel,
	}

	// Set Gemma4-specific config
	if isGemma4 {
		e.config.Gemma4SlidingWindowSize = 512
		e.config.Gemma4SlidingRoPETheta = 10000.0
		e.config.Gemma4FullRoPETheta = 1000000.0
		e.config.Gemma4PartialRoPEFactor = 0.25
		e.config.Gemma4SlidingHeadDim = 256
		e.config.Gemma4FullHeadDim = 512
		log.Printf("Gemma4 config: sliding_window=%d, sliding_theta=%.0f, full_theta=%.0f",
			e.config.Gemma4SlidingWindowSize, e.config.Gemma4SlidingRoPETheta, e.config.Gemma4FullRoPETheta)
	}

	qNormDim := 512
	kNormDim := 512
	e.scratch = ctx.NewLayerScratch(seqLen, dim, hiddenDim, heads, kvHeads, headDim, seqLen, vocabSize, qNormDim, kNormDim)

	return e, nil
}
func (e *cudaEngine) Config() config.Config {
	return e.config
}

func (e *cudaEngine) SwapModel(newModelPath string, newConfig config.Config) error {
	startTime := time.Now()
	success := false

	defer func() {
		metrics.RecordModelHotSwap(time.Since(startTime), success)
	}()

	e.mu.Lock()
	defer e.mu.Unlock()

	// Free existing cache and weights
	if e.cuda != nil {
		for _, c := range e.cuda.KCache {
			if c != nil {
				c.Free()
			}
		}
		for _, c := range e.cuda.VCache {
			if c != nil {
				c.Free()
			}
		}
		e.cuda.Free()
	}
	if e.model != nil {
		e.model.Close()
	}

	// Load the new model
	f, err := gguf.LoadFile(newModelPath)
	if err != nil {
		return fmt.Errorf("failed to load GGUF: %w", err)
	}

	_ = f // Close will happen in deferred cleanup

	ctx, err := device.NewCUDAContext()
	if err != nil {
		f.Close()
		return fmt.Errorf("failed to create CUDA context: %w", err)
	}

	cudaModel, err := ctx.NewCUDAModel(f, true, newConfig.KVCacheSize)
	if err != nil {
		ctx.Free()
		f.Close()
		return fmt.Errorf("failed to load model to GPU: %w", err)
	}

	// Update config
	e.config = newConfig
	e.config.KVCacheSize = newConfig.KVCacheSize

	// Update components
	e.model = f
	e.cuda = cudaModel

	success = true
	return nil
}
func (e *cudaEngine) Close() {
	for _, t := range e.dequantizedCache {
		if t != nil {
			t.Free()
		}
	}
	e.dequantizedCache = make(map[string]*device.CUDATensor)

	if e.scratch != nil {
		e.scratch.Free()
	}
	if e.cuda != nil {
		if e.cuda.Ctx != nil {
			e.cuda.Ctx.Free()
		}
		e.cuda.Free()
	}
	if e.model != nil {
		e.model.Close()
	}
}

func (e *cudaEngine) getDequantedWeight(name string) (*device.CUDATensor, error) {
	e.mu.RLock()
	if cached, ok := e.dequantizedCache[name]; ok && cached != nil {
		e.mu.RUnlock()
		return cached, nil
	}
	e.mu.RUnlock()

	d, err := e.cuda.GetDequantedWeight(name)
	if err != nil {
		log.Printf("DEBUG getDequantedWeight(%s) returning error: %v", name, err)
		return nil, err
	}

	e.mu.Lock()
	e.dequantizedCache[name] = d
	e.mu.Unlock()

	return d, nil
}

func (e *cudaEngine) Infer(inputTokens []int, tokensToGenerate int, samplerConfig SamplerConfig) ([]int, error) {
	return e.InferWithCallbackLogits(inputTokens, tokensToGenerate, samplerConfig, nil, func(logits []float32) {
		if len(logits) > 0 {
			maxVal := float32(-999999)
			for i := 1; i < min(100, len(logits)); i++ {
				if logits[i] > maxVal {
					maxVal = logits[i]
				}
			}
			log.Printf("DEBUG: Top logit value: %f", maxVal)
		}
	})
}

func (e *cudaEngine) InferWithCallback(inputTokens []int, tokensToGenerate int, samplerConfig SamplerConfig, callback func(int)) ([]int, error) {
	if len(inputTokens) == 0 {
		return nil, fmt.Errorf("empty input tokens")
	}

	result := make([]int, 0, tokensToGenerate)

	sampler := NewSampler(samplerConfig)

	log.Printf("Starting inference: %d prompt tokens + %d to generate", len(inputTokens), tokensToGenerate)
	startTime := time.Now()

	cudaInferenceTotal.WithLabelValues(e.config.Architecture).Inc()
	cudaBatchSize.WithLabelValues(e.config.Architecture).Observe(float64(len(inputTokens)))

	inputLen := len(inputTokens)
	kvHits, kvMisses := 0, 0
	seqLen := inputLen + tokensToGenerate
	if seqLen > e.config.SeqLen {
		seqLen = e.config.SeqLen
	}

	allTokens := make([]int, 0, seqLen)
	allTokens = append(allTokens, inputTokens...)

	layerStart := time.Now()
	var logits []float32
	var err error
	for pos := 0; pos < inputLen; pos++ {
		token := inputTokens[pos]
		logits, err = e.forward(token, pos, allTokens)
		if err != nil {
			cudaEngineFailed.WithLabelValues(e.config.Architecture, "forward_failed").Inc()
			return nil, fmt.Errorf("forward pass failed at position %d: %w", pos, err)
		}
		layerLatency := time.Since(layerStart).Seconds()
		cudaLayerLatency.WithLabelValues(e.config.Architecture, "prompt").Observe(layerLatency)
		layerStart = time.Now()
	}

	for gen := 0; gen < tokensToGenerate && len(allTokens) < seqLen; gen++ {
		samplingStart := time.Now()
		nextToken := sampler.Sample(logits, allTokens)
		cudaSamplingTime.WithLabelValues(e.config.Architecture).Observe(time.Since(samplingStart).Seconds())

		allTokens = append(allTokens, nextToken)
		result = append(result, nextToken)
		cudaTokensGenerated.WithLabelValues(e.config.Architecture).Inc()

		if callback != nil {
			callback(nextToken)
		}

		// Prepare logits for next iteration
		currentPos := len(allTokens) - 1
		logits, err = e.forward(nextToken, currentPos, allTokens)
		if err != nil {
			cudaEngineFailed.WithLabelValues(e.config.Architecture, "forward_failed").Inc()
			return nil, fmt.Errorf("forward pass failed at generation step %d: %w", gen, err)
		}

		layerLatency := time.Since(layerStart).Seconds()
		cudaLayerLatency.WithLabelValues(e.config.Architecture, fmt.Sprintf("gen_%d", gen)).Observe(layerLatency)
		layerStart = time.Now()
	}

	elapsed := time.Since(startTime)
	tokensPerSecond := float64(len(result)) / elapsed.Seconds()

	log.Printf("Generated %d tokens in %.2fs (%.1f t/s)", len(result), elapsed.Seconds(), tokensPerSecond)

	cudaInferenceDuration.WithLabelValues(e.config.Architecture).Observe(elapsed.Seconds())
	cudaTokensPerSecond.WithLabelValues(e.config.Architecture).Observe(tokensPerSecond)
	cudaKVCacheHits.WithLabelValues(e.config.Architecture).Add(float64(kvHits))
	cudaKVCacheMisses.WithLabelValues(e.config.Architecture).Add(float64(kvMisses))

	if e.cuda != nil {
		cudaMemoryUsage.WithLabelValues(e.config.Architecture).Set(float64(device.CUDAAllocatedBytes()))
	}

	if e.tokenizer != nil && len(result) > 0 {
		text := e.tokenizer.Decode(result)
		log.Printf("Output: %s", text)
	}

	return result, nil
}

func (e *cudaEngine) forward(token int, pos int, allTokens []int) ([]float32, error) {
	heads := e.config.Heads
	kvHeads := e.config.KVHeads
	headDim := e.config.HeadDim
	eps := e.config.Eps
	ropeTheta := e.config.RopeTheta

	hidden, err := e.cuda.GetEmbedding(token)
	if err != nil {
		return nil, err
	}
	if token == 0 && pos == 0 && len(hidden) > 0 {
		log.Printf("DEBUG: GetEmbedding token=%d first5=%v", token, hidden[:5])
		sum := float32(0)
		for i := 0; i < min(10, len(hidden)); i++ {
			sum += hidden[i]
		}
		log.Printf("DEBUG: token=%d embedding sum (first 10): %f", token, sum)
	}

	hidden = append([]float32{}, hidden...)

	// Debug: after copy
	if token == 0 && pos == 0 {
		sum := float32(0)
		for i := 0; i < min(10, len(hidden)); i++ {
			sum += hidden[i]
		}
		log.Printf("DEBUG: after copy, hidden sum (first 10): %f", sum)
	}

	isGemma4 := e.config.IsGemma4
	gemma4SlidingWindowSize := e.config.Gemma4SlidingWindowSize
	gemma4SlidingRoPETheta := e.config.Gemma4SlidingRoPETheta
	gemma4FullRoPETheta := e.config.Gemma4FullRoPETheta
	gemma4SlidingHeadDim := e.config.Gemma4SlidingHeadDim
	gemma4FullHeadDim := e.config.Gemma4FullHeadDim

	for layer := 0; layer < e.config.Layers; layer++ {
		attnNormW, err := e.getDequantedWeight(fmt.Sprintf("blk.%d.attn_norm.weight", layer))
		if err != nil {
			continue
		}
		if attnNormW == nil {
			continue
		}
		attnNorm := attnNormW.ToHostF32()

		// Debug: print attn_norm weight stats
		if layer == 0 && token == 0 && pos == 0 {
			attnNormData := attnNormW.ToHostF32()
			sum := float32(0)
			log.Printf("DEBUG: attn_norm dequantized tensor rows=%d, cols=%d, len(data)=%d",
				attnNormW.Rows, attnNormW.Cols, len(attnNormData))
			for i := 0; i < min(10, len(attnNormData)); i++ {
				sum += attnNormData[i]
			}
			log.Printf("DEBUG: attn_norm dequantized first 10 sum: %f, values: %v", sum, attnNormData[:10])

			// Also check hidden before RMSNorm
			hiddenSum := float32(0)
			for i := 0; i < min(10, len(hidden)); i++ {
				hiddenSum += hidden[i]
			}
			log.Printf("DEBUG: hidden before rmsnorm first 10 sum: %f, values: %v", hiddenSum, hidden[:10])
		}

		hidden = e.rmsnorm(hidden, attnNorm, eps)

		// Debug: check hidden after attn_norm
		if layer == 0 && pos == 0 && token == 0 {
			sum := float32(0)
			for i := 0; i < min(10, len(hidden)); i++ {
				sum += hidden[i]
			}
			log.Printf("DEBUG: layer0 pos0 after attn_norm hidden sum: %f", sum)
		}

		qW, _ := e.getDequantedWeight(fmt.Sprintf("blk.%d.attn_q.weight", layer))
		kW, _ := e.getDequantedWeight(fmt.Sprintf("blk.%d.attn_k.weight", layer))
		vW, _ := e.getDequantedWeight(fmt.Sprintf("blk.%d.attn_v.weight", layer))
		oW, _ := e.getDequantedWeight(fmt.Sprintf("blk.%d.attn_output.weight", layer))

		if qW == nil || kW == nil || vW == nil || oW == nil {
			continue
		}

		var q, k, v []float32
		q, err = e.matmulGPU(hidden, fmt.Sprintf("blk.%d.attn_q.weight", layer))
		if err != nil {
			return nil, fmt.Errorf("failed to project q: %w", err)
		}
		k, err = e.matmulGPU(hidden, fmt.Sprintf("blk.%d.attn_k.weight", layer))
		if err != nil {
			return nil, fmt.Errorf("failed to project k: %w", err)
		}
		v, err = e.matmulGPU(hidden, fmt.Sprintf("blk.%d.attn_v.weight", layer))
		if err != nil {
			return nil, fmt.Errorf("failed to project v: %w", err)
		}

		// Gemma4: Apply Q/K normalization after projection
		if isGemma4 {
			qNormW, _ := e.getDequantedWeight(fmt.Sprintf("blk.%d.attn_q_norm.weight", layer))
			kNormW, _ := e.getDequantedWeight(fmt.Sprintf("blk.%d.attn_k_norm.weight", layer))
			if qNormW != nil && kNormW != nil {
				qNorm := qNormW.ToHostF32()
				kNorm := kNormW.ToHostF32()
				for i := range q {
					q[i] = q[i] * qNorm[i]
					k[i] = k[i] * kNorm[i]
				}
			}

			// Determine if this is a sliding window or full attention layer
			isSlidingWindowLayer := (layer % 6) != 5
			gemma4PartialFactor := e.config.Gemma4PartialRoPEFactor
			if isSlidingWindowLayer {
				currentHeadDim := gemma4SlidingHeadDim
				currentTheta := gemma4SlidingRoPETheta
				e.applyRoPEWithFactor(q, pos, int(currentTheta), currentHeadDim, gemma4PartialFactor)
				e.applyRoPEWithFactor(k, pos, int(currentTheta), currentHeadDim, gemma4PartialFactor)
			} else {
				// Full attention layer - use full RoPE theta but standard dim
				currentHeadDim := gemma4FullHeadDim
				currentTheta := gemma4FullRoPETheta
				e.applyRoPE(q, pos, int(currentTheta), currentHeadDim)
				e.applyRoPE(k, pos, int(currentTheta), currentHeadDim)
			}
		} else {
			e.applyRoPE(q, pos, int(ropeTheta), headDim)
			e.applyRoPE(k, pos, int(ropeTheta), headDim)
		}

		ffnNormW, _ := e.getDequantedWeight(fmt.Sprintf("blk.%d.ffn_norm.weight", layer))
		ffnGateW, _ := e.getDequantedWeight(fmt.Sprintf("blk.%d.ffn_gate.weight", layer))
		ffnUpW, _ := e.getDequantedWeight(fmt.Sprintf("blk.%d.ffn_up.weight", layer))
		ffnDownW, _ := e.getDequantedWeight(fmt.Sprintf("blk.%d.ffn_down.weight", layer))

		q3d := e.viewAsTensor(q, heads, headDim)
		k3d := e.viewAsTensor(k, kvHeads, headDim)
		v3d := e.viewAsTensor(v, kvHeads, headDim)

		kCache := e.cuda.GetKCache(layer)
		vCache := e.cuda.GetVCache(layer)

		// Debug: before attention
		if token == 0 && pos == 0 && layer == 0 {
			q3d := e.viewAsTensor(q, heads, headDim)
			k3d := e.viewAsTensor(k, kvHeads, headDim)
			v3d := e.viewAsTensor(v, kvHeads, headDim)
			log.Printf("DEBUG: before attention: q3d[0][0:5]=%v, k3d[0][0:5]=%v, v3d[0][0:5]=%v", q3d[0][:5], k3d[0][:5], v3d[0][:5])
		}

		var attnOut []float32
		attnWindowSize := 0
		if isGemma4 {
			isSlidingWindowLayer := (layer % 6) != 5
			if isSlidingWindowLayer {
				attnWindowSize = gemma4SlidingWindowSize
			}
		}
		if attnWindowSize > 0 {
			attnOut = e.attentionWithWindow(q3d, k3d, v3d, kCache, vCache, pos, heads, kvHeads, headDim, e.config.SeqLen, attnWindowSize)
		} else {
			attnOut = e.attention(q3d, k3d, v3d, kCache, vCache, pos, heads, kvHeads, headDim, e.config.SeqLen)
		}
		if attnOut == nil {
			attnOut = e.attentionFallback(q3d, k3d, v3d)
		}

		// Debug: after attention
		if token == 0 && pos == 0 && layer == 0 {
			sum := float32(0)
			for i := 0; i < min(10, len(attnOut)); i++ {
				sum += attnOut[i]
			}
			log.Printf("DEBUG: after attention: attnOut sum=%f, first5=%v", sum, attnOut[:5])
		}

		attnProj, err := e.matmulGPU(attnOut, fmt.Sprintf("blk.%d.attn_output.weight", layer))
		if err != nil {
			return nil, fmt.Errorf("failed to project attn_output: %w", err)
		}

		// Debug: after proj
		if token == 0 && pos == 0 && layer == 0 {
			sum := float32(0)
			for i := 0; i < min(10, len(attnProj)); i++ {
				sum += attnProj[i]
			}
			log.Printf("DEBUG: after proj: attnProj sum=%f, first5=%v", sum, attnProj[:5])
			hiddenSum := float32(0)
			for i := 0; i < min(10, len(hidden)); i++ {
				hiddenSum += hidden[i]
			}
			log.Printf("DEBUG: before add: hidden sum=%f, first5=%v", hiddenSum, hidden[:5])
		}

		for i := range hidden {
			hidden[i] += attnProj[i]
		}

		// Debug: after add
		if token == 0 && pos == 0 && layer == 0 {
			log.Printf("DEBUG: after add: hidden[0:5]=%v", hidden[:5])
		}

		if ffnNormW != nil && ffnGateW != nil && ffnUpW != nil && ffnDownW != nil {
			ffnNorm := ffnNormW.ToHostF32()
			hidden = e.rmsnorm(hidden, ffnNorm, eps)

			var ffnGate, ffnUp, ffnDown []float32
			ffnGate, err = e.matmulGPU(hidden, fmt.Sprintf("blk.%d.ffn_gate.weight", layer))
			if err != nil {
				return nil, fmt.Errorf("failed to project ffn_gate: %w", err)
			}
			ffnUp, err = e.matmulGPU(hidden, fmt.Sprintf("blk.%d.ffn_up.weight", layer))
			if err != nil {
				return nil, fmt.Errorf("failed to project ffn_up: %w", err)
			}

			for i := range ffnGate {
				ffnGate[i] = ffnGate[i] / (1 + float32(math.Exp(float64(-ffnGate[i]))))
			}

			for i := range ffnUp {
				ffnUp[i] *= ffnGate[i]
			}

			ffnDown, err = e.matmulGPU(ffnUp, fmt.Sprintf("blk.%d.ffn_down.weight", layer))
			if err != nil {
				return nil, fmt.Errorf("failed to project ffn_down: %w", err)
			}
			for i := range hidden {
				hidden[i] += ffnDown[i]
			}
		}

		// Debug: after layer
		if token == 0 && pos == 0 && layer < 3 {
			sum := float32(0)
			for i := 0; i < min(10, len(hidden)); i++ {
				sum += hidden[i]
			}
			log.Printf("DEBUG: after layer %d, hidden sum: %f", layer, sum)
		}
	}

	outputNormW, err := e.getDequantedWeight("output_norm.weight")
	if err != nil || outputNormW == nil {
		logits := make([]float32, e.config.VocabSize)
		for i := range logits {
			logits[i] = float32(-i)
		}
		return logits, nil
	}
	outputNorm := outputNormW.ToHostF32()
	hidden = e.rmsnorm(hidden, outputNorm, eps)

	var logits []float32

	outputW, err := e.getDequantedWeight("output.weight")
	log.Printf("DEBUG: output.weight get result: %v, err: %v", outputW, err)

	if outputW == nil {
		// Fallback: use token_embd.weight (tied embeddings)
		log.Printf("DEBUG: output.weight not found, trying token_embd.weight")

		// Try to get the raw weight (not dequantized) and dequantize on CPU
		emb, ok := e.cuda.GetWeight("token_embd.weight")
		if ok && emb != nil {
			log.Printf("DEBUG: Got token_embd raw weight, type=%v, rows=%d, cols=%d", emb.GGMLType, emb.Rows, emb.Cols)

			// Dequantize on CPU
			numElements := emb.Rows * emb.Cols
			var dequantized []float32
			switch emb.GGMLType {
			case gguf.GGMLTypeQ8_0:
				dequantized = gguf.DequantizeQ8_0(emb.HostData, numElements)
			case gguf.GGMLTypeQ4_K:
				dequantized = gguf.DequantizeQ4K(emb.HostData, numElements)
			case gguf.GGMLTypeQ6_K:
				dequantized = gguf.DequantizeQ6K(emb.HostData, numElements)
			case gguf.GGMLTypeF32:
				dequantized = make([]float32, numElements)
				for i := 0; i < numElements; i++ {
					dequantized[i] = math.Float32frombits(uint32(emb.HostData[i*4]) | uint32(emb.HostData[i*4+1])<<8 | uint32(emb.HostData[i*4+2])<<16 | uint32(emb.HostData[i*4+3])<<24)
				}
			}

			if len(dequantized) > 0 {
				log.Printf("DEBUG: CPU dequantized token_embd, first10: %v", dequantized[:10])
				logits = e.matmul(hidden, dequantized)
			}
		}
	}

	// If still no logits, use GPU path or fallback
	if len(logits) == 0 {
		if outputW != nil {
			log.Printf("DEBUG: Using GPU output weight")
			outputData := outputW.ToHostF32()
			logits = e.matmul(hidden, outputData)
		} else {
			log.Printf("DEBUG: Using fallback logits (uniform)")
			logits = make([]float32, e.config.VocabSize)
			for i := range logits {
				logits[i] = float32(-i)
			}
		}
	}

	// Debug: print logits stats
	if len(logits) > 0 {
		// First check hidden before output projection
		hiddenSum := float32(0)
		for i := 0; i < min(20, len(hidden)); i++ {
			hiddenSum += hidden[i]
		}
		log.Printf("DEBUG: hidden before output layer sum: %f, values: %v", hiddenSum, hidden[:20])

		// Check output weight
		if outputW != nil {
			outputData := outputW.ToHostF32()
			outSum := float32(0)
			for i := 0; i < min(20, len(outputData)); i++ {
				outSum += outputData[i]
			}
			log.Printf("DEBUG: output weight first 20 sum: %f, values: %v", outSum, outputData[:20])
		}

		minLogit := logits[0]
		maxLogit := logits[0]
		for i := 1; i < min(100, len(logits)); i++ {
			if logits[i] < minLogit {
				minLogit = logits[i]
			}
			if logits[i] > maxLogit {
				maxLogit = logits[i]
			}
		}
		log.Printf("DEBUG: logits min=%f, max=%f, first20=%v", minLogit, maxLogit, logits[:20])
	}

	return logits, nil
}

func (e *cudaEngine) rmsnorm(input, weight []float32, eps float32) []float32 {
	n := len(input)
	result := make([]float32, n)

	var sum float32 = 0
	for i := range input {
		sum += input[i] * input[i]
	}
	sum = float32(math.Sqrt(float64(sum)/float64(n) + float64(eps)))

	for i := range result {
		result[i] = input[i] / sum * weight[i]
	}

	return result
}

func (e *cudaEngine) matmul(a, b []float32) []float32 {
	aRows := 1
	aCols := len(a)
	bRows := aCols
	bCols := len(b) / bRows

	result := make([]float32, aRows*bCols)

	for i := 0; i < aCols; i++ {
		for j := 0; j < bCols; j++ {
			result[j] += a[i] * b[i*bCols+j]
		}
	}

	return result
}

func (e *cudaEngine) matmulGPU(inputData []float32, weightName string) ([]float32, error) {
	if e.cuda == nil || e.cuda.Ctx == nil {
		return nil, fmt.Errorf("CUDA context not available")
	}

	weightTensor, err := e.cuda.GetWeightTensor(weightName)
	if err != nil {
		return nil, fmt.Errorf("failed to get weight tensor: %w", err)
	}

	inputRows := 1
	inputCols := len(inputData)
	weightRows := weightTensor.Rows()

	if inputCols != weightRows {
		return nil, fmt.Errorf("matmul dimension mismatch: input cols=%d, weight rows=%d", inputCols, weightRows)
	}

	inputTensor, err := e.cuda.Ctx.NewTensor(inputRows, inputCols, device.DataTypeF16)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %w", err)
	}
	defer inputTensor.Free()

	if err := inputTensor.LoadFrom(inputData); err != nil {
		return nil, fmt.Errorf("failed to load input data: %w", err)
	}

	outputTensor, err := e.cuda.Ctx.LinearF16(inputTensor, weightTensor)
	if err != nil {
		return nil, fmt.Errorf("failed to run GPU matmul: %w", err)
	}
	defer outputTensor.Free()

	e.cuda.Ctx.Synchronize()

	return outputTensor.ToHostF32(), nil
}

func (e *cudaEngine) attentionGPU(q, k, v []float32, kCache, vCache *device.CUDATensor, pos, heads, kvHeads, headDim, seqLen, windowSize int) ([]float32, error) {
	if e.cuda == nil || e.cuda.Ctx == nil {
		return nil, fmt.Errorf("CUDA context not available")
	}

	numHeads := heads
	dim := headDim
	batch := 1
	seqLenQ := 1
	kvSeqLen := pos + 1
	scale := float32(1.0 / math.Sqrt(float64(dim)))

	qTensor, err := e.cuda.Ctx.NewTensorFP32(batch*numHeads, headDim)
	if err != nil {
		return nil, err
	}
	defer qTensor.Free()
	qTensor.LoadFrom(q)

	kTensor, err := e.cuda.Ctx.NewTensorFP32(batch*kvHeads, headDim)
	if err != nil {
		return nil, err
	}
	defer kTensor.Free()
	kTensor.LoadFrom(k)

	vTensor, err := e.cuda.Ctx.NewTensorFP32(batch*kvHeads, headDim)
	if err != nil {
		return nil, err
	}
	defer vTensor.Free()
	vTensor.LoadFrom(v)

	outTensor, err := e.cuda.Ctx.NewTensorFP32(batch*numHeads, headDim)
	if err != nil {
		return nil, err
	}
	defer outTensor.Free()

	useCache := 0
	if kCache != nil && vCache != nil {
		useCache = 1
	}

	e.cuda.Ctx.FusedAttention(qTensor, kTensor, vTensor, outTensor, kCache, vCache, batch, numHeads, seqLenQ, kvSeqLen, headDim, scale, useCache, 0)
	e.cuda.Ctx.Synchronize()

	return outTensor.ToHostF32(), nil
}

func (e *cudaEngine) applyRoPEGPU(tensor []float32, pos int, theta float32, headDim int) error {
	if e.cuda == nil || e.cuda.Ctx == nil {
		return fmt.Errorf("CUDA context not available")
	}

	tensorGPU, err := e.cuda.Ctx.NewTensorFP32(1, len(tensor))
	if err != nil {
		return err
	}
	defer tensorGPU.Free()
	tensorGPU.LoadFrom(tensor)

	posIds := []int{pos}
	e.cuda.Ctx.FusedRoPE(tensorGPU, posIds, 1, 1, 1, headDim, theta)
	e.cuda.Ctx.Synchronize()

	copy(tensor, tensorGPU.ToHostF32())
	return nil
}

func (e *cudaEngine) applyRoPE(tensor []float32, pos int, theta, dim int) {
	e.applyRoPEWithFactor(tensor, pos, theta, dim, 1.0)
}

func (e *cudaEngine) applyRoPEWithFactor(tensor []float32, pos int, theta, dim int, partialFactor float32) {
	if len(tensor)%2 != 0 {
		return
	}

	numHeads := len(tensor) / dim
	halfDim := dim / 2
	for h := 0; h < numHeads; h++ {
		offset := h * dim
		for i := 0; i < halfDim; i++ {
			idx1 := offset + i
			idx2 := offset + i + halfDim

			freq := float64(pos) * math.Pow(float64(theta), -2.0*float64(i)/float64(dim))
			cos := float32(math.Cos(freq))
			sin := float32(math.Sin(freq))

			x1 := tensor[idx1]
			x2 := tensor[idx2]
			tensor[idx1] = x1*cos - x2*sin
			tensor[idx2] = x1*sin + x2*cos
		}
	}
}

func (e *cudaEngine) viewAsTensor(data []float32, heads, headDim int) [][]float32 {
	result := make([][]float32, heads)
	for h := 0; h < heads; h++ {
		result[h] = make([]float32, headDim)
		copy(result[h], data[h*headDim:(h+1)*headDim])
	}
	return result
}

func (e *cudaEngine) storeKV(kCache, vCache *device.CUDATensor, pos int, k, v [][]float32) {
	if kCache == nil || vCache == nil || len(k) == 0 || len(k[0]) == 0 {
		return
	}

	if e.cuda == nil || e.cuda.Ctx == nil {
		return
	}

	heads := len(k)
	headDim := len(k[0])

	kFlat := make([]float32, heads*headDim)
	for h := 0; h < heads; h++ {
		copy(kFlat[h*headDim:(h+1)*headDim], k[h])
	}

	vFlat := make([]float32, heads*headDim)
	for h := 0; h < heads; h++ {
		copy(vFlat[h*headDim:(h+1)*headDim], v[h])
	}

	kTemp, err := e.cuda.Ctx.NewTensorFP32(heads, headDim)
	if err != nil {
		return
	}
	defer kTemp.Free()
	_ = kTemp.LoadFrom(kFlat)

	vTemp, err := e.cuda.Ctx.NewTensorFP32(heads, headDim)
	if err != nil {
		return
	}
	defer vTemp.Free()
	_ = vTemp.LoadFrom(vFlat)

	e.cuda.Ctx.CopyF16(kTemp, kCache)
	e.cuda.Ctx.CopyF16(vTemp, vCache)
	e.cuda.Ctx.Synchronize()
}

func (e *cudaEngine) attentionWithWindow(q, k, v [][]float32, kCache, vCache *device.CUDATensor, pos, heads, kvHeads, headDim, seqLen, windowSize int) []float32 {
	if len(q) == 0 || len(k) == 0 || len(v) == 0 {
		return nil
	}

	dim := len(q[0])
	numHeads := len(q)
	scale := float32(1.0 / math.Sqrt(float64(dim)))

	seqLenK := pos + 1
	actualWindowSize := seqLenK
	if windowSize > 0 && windowSize < seqLenK {
		actualWindowSize = windowSize
	}

	allK := make([][]float32, kvHeads)
	allV := make([][]float32, kvHeads)

	for h := 0; h < kvHeads; h++ {
		allK[h] = make([]float32, seqLenK*dim)
		allV[h] = make([]float32, seqLenK*dim)
		for i := 0; i < seqLenK; i++ {
			copy(allK[h][i*dim:(i+1)*dim], k[h])
			copy(allV[h][i*dim:(i+1)*dim], v[h])
		}
	}

	attn := make([][]float32, numHeads)
	for h := 0; h < numHeads; h++ {
		attn[h] = make([]float32, dim)
	}

	scores := make([]float32, actualWindowSize)

	for h := 0; h < numHeads; h++ {
		kvH := h / (numHeads / kvHeads)

		startIdx := 0
		if seqLenK > actualWindowSize {
			startIdx = seqLenK - actualWindowSize
		}

		for i := 0; i < actualWindowSize; i++ {
			srcIdx := startIdx + i
			var dot float32 = 0
			for d := 0; d < dim; d++ {
				dot += q[h][d] * allK[kvH][srcIdx*dim+d]
			}
			scores[i] = dot * scale
		}

		maxScore := float32(-math.MaxFloat32)
		for i := range scores {
			if scores[i] > maxScore {
				maxScore = scores[i]
			}
		}
		for i := range scores {
			scores[i] = float32(math.Exp(float64(scores[i] - maxScore)))
		}

		sum := float32(0)
		for i := range scores {
			sum += scores[i]
		}
		if sum > 0 {
			for i := range scores {
				scores[i] /= sum
			}
		}

		for d := 0; d < dim; d++ {
			var out float32 = 0
			for i := 0; i < actualWindowSize; i++ {
				srcIdx := startIdx + i
				out += scores[i] * allV[kvH][srcIdx*dim+d]
			}
			attn[h][d] = out
		}
	}

	result := make([]float32, numHeads*dim)
	for h := 0; h < numHeads; h++ {
		copy(result[h*dim:(h+1)*dim], attn[h])
	}
	return result
}

func (e *cudaEngine) attention(q, k, v [][]float32, kCache, vCache *device.CUDATensor, pos, heads, kvHeads, headDim, seqLen int) []float32 {
	return e.attentionWithWindow(q, k, v, kCache, vCache, pos, heads, kvHeads, headDim, seqLen, 0)
}

func (e *cudaEngine) attentionFallback(q, k, v [][]float32) []float32 {
	numHeads := len(q)
	dim := len(q[0])
	scale := float32(1.0 / math.Sqrt(float64(dim)))

	kLen := len(k)
	vLen := len(v)

	attn := make([][]float32, numHeads)
	for h := 0; h < numHeads; h++ {
		attn[h] = make([]float32, dim)
	}

	for h := 0; h < numHeads; h++ {
		scores := make([]float32, kLen)
		for i := 0; i < kLen; i++ {
			var dot float32
			for d := 0; d < dim; d++ {
				dot += q[h][d] * k[i][d]
			}
			scores[i] = dot * scale
		}

		maxScore := float32(-math.MaxFloat32)
		for i := range scores {
			if scores[i] > maxScore {
				maxScore = scores[i]
			}
		}
		for i := range scores {
			scores[i] = float32(math.Exp(float64(scores[i] - maxScore)))
		}

		sum := float32(0)
		for i := range scores {
			sum += scores[i]
		}
		if sum > 0 {
			for i := range scores {
				scores[i] /= sum
			}
		}

		for d := 0; d < dim; d++ {
			var out float32
			for i := 0; i < vLen; i++ {
				out += scores[i] * v[i][d]
			}
			attn[h][d] = out
		}
	}

	result := make([]float32, numHeads*dim)
	for h := 0; h < numHeads; h++ {
		copy(result[h*dim:(h+1)*dim], attn[h])
	}

	return result
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

func (s *Sampler) applyRepetitionPenalty(logits []float32, history []int) {
	if len(history) == 0 || s.config.RepPenalty <= 1.0 {
		return
	}

	seen := make(map[int]bool)
	// Penalize tokens seen in the last 64 positions
	start := 0
	if len(history) > 64 {
		start = len(history) - 64
	}
	for _, tokenID := range history[start:] {
		if seen[tokenID] {
			continue
		}
		seen[tokenID] = true
		if tokenID < len(logits) {
			if logits[tokenID] > 0 {
				logits[tokenID] /= float32(s.config.RepPenalty)
			} else {
				logits[tokenID] *= float32(s.config.RepPenalty)
			}
		}
	}
}

func (s *Sampler) Sample(logits []float32, history []int) int {
	if len(logits) == 0 {
		return 0
	}

	s.applyRepetitionPenalty(logits, history)

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

	for i := range logits {
		logits[i] = float32(float64(logits[i]) / s.config.Temperature)
	}

	topK := s.config.TopK
	if topK <= 0 || topK > len(logits) {
		topK = len(logits)
	}

	type tokenScore struct {
		token int
		score float32
	}

	scored := make([]tokenScore, len(logits))
	for i := range logits {
		scored[i] = tokenScore{i, logits[i]}
	}

	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})

	// Log top 5 candidates for debugging
	log.Printf("DEBUG Sampling: top5: [%d:%.2f %d:%.2f %d:%.2f %d:%.2f %d:%.2f]",
		scored[0].token, scored[0].score,
		scored[1].token, scored[1].score,
		scored[2].token, scored[2].score,
		scored[3].token, scored[3].score,
		scored[4].token, scored[4].score)

	topKScore := scored[0].token
	if topK > 1 && topK < len(scored) {
		cutoff := scored[topK-1].score
		cutoff = float32(math.Max(float64(cutoff), 0))

		sum := float32(0)
		for i := 0; i < topK; i++ {
			if scored[i].score >= cutoff {
				scored[i].score = float32(math.Exp(float64(scored[i].score - cutoff)))
				sum += scored[i].score
			} else {
				scored[i].score = 0
			}
		}

		if s.config.TopP > 0 && s.config.TopP < 1.0 {
			sumP := float32(0)
			for i := 0; i < len(scored) && scored[i].score > 0; i++ {
				sumP += scored[i].score
				if sumP >= float32(s.config.TopP)*sum {
					for j := i + 1; j < len(scored); j++ {
						scored[j].score = 0
					}
					break
				}
			}
		}

		if sum > 0 {
			for i := range scored {
				scored[i].score /= sum
			}
		}

		r := float32(s.rng.Float64())
		accum := float32(0)
		for i := range scored {
			accum += scored[i].score
			if r <= accum {
				topKScore = scored[i].token
				break
			}
		}
	}

	return topKScore
}

func init() {
	log.SetOutput(os.Stderr)
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)
	RegisterEngine("cuda", NewcudaEngine)
}

// =============================================================================
// GPU-based Fused Operations (keep data on GPU for maximum performance)
// =============================================================================

func (e *cudaEngine) fusedAttentionGPU(q, k, v *device.CUDATensor, output, kCache, vCache *device.CUDATensor, batch, heads, seqLen, kvSeqLen, headDim int) {
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	useCache := 0
	if kCache != nil && vCache != nil {
		useCache = 1
	}
	e.cuda.Ctx.FusedAttention(q, k, v, output, kCache, vCache, batch, heads, seqLen, kvSeqLen, headDim, scale, useCache, 0)
}

func (e *cudaEngine) flashAttentionGPU(q, k, v *device.CUDATensor, output *device.CUDATensor, batch, heads, seqLen, kvSeqLen, headDim int) {
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	e.cuda.Ctx.FlashFusedAttention(q, k, v, output, batch, heads, seqLen, kvSeqLen, headDim, scale, 0)
}

func (e *cudaEngine) fusedRoPEGPU(tensor *device.CUDATensor, posIds []int, batch, heads, seqLen, headDim int) {
	theta := float32(e.config.RopeTheta)
	e.cuda.Ctx.FusedRoPE(tensor, posIds, batch, heads, seqLen, headDim, theta)
}

func (e *cudaEngine) fusedSwiGLUGPU(input, gateWeight, upWeight, downWeight, output *device.CUDATensor, batch, dim, hiddenDim int) {
	e.cuda.Ctx.FusedSwiGLU(input, gateWeight, upWeight, downWeight, output, batch, dim, hiddenDim)
}

func (e *cudaEngine) fusedMLPGPU(input, gateWeight, upWeight, downWeight, output *device.CUDATensor, batch, dim, hiddenDim int) {
	e.cuda.Ctx.FusedMLP(input, gateWeight, upWeight, downWeight, output, batch, dim, hiddenDim)
}

func (e *cudaEngine) fusedRMSNormAddGPU(input, hidden, weight, output *device.CUDATensor, batch, dim int) {
	e.cuda.Ctx.FusedRMSNormAdd(input, hidden, weight, output, batch, dim, e.config.Eps)
}

func (e *cudaEngine) InferWithLogits(inputTokens []int, tokensToGenerate int, samplerConfig SamplerConfig) ([]int, []float32, error) {
	var lastLogits []float32
	tokens, err := e.InferWithCallbackLogits(inputTokens, tokensToGenerate, samplerConfig, nil, func(logits []float32) {
		lastLogits = make([]float32, len(logits))
		copy(lastLogits, logits)
	})
	return tokens, lastLogits, err
}

func (e *cudaEngine) InferWithCallbackLogits(inputTokens []int, tokensToGenerate int, samplerConfig SamplerConfig, tokenCallback func(int), logitsCallback func([]float32)) ([]int, error) {
	if len(inputTokens) == 0 {
		return nil, fmt.Errorf("empty input tokens")
	}

	result := make([]int, 0, tokensToGenerate)

	sampler := NewSampler(samplerConfig)

	log.Printf("Starting inference: %d prompt tokens + %d to generate", len(inputTokens), tokensToGenerate)
	startTime := time.Now()

	cudaInferenceTotal.WithLabelValues(e.config.Architecture).Inc()
	cudaBatchSize.WithLabelValues(e.config.Architecture).Observe(float64(len(inputTokens)))

	inputLen := len(inputTokens)
	kvHits, kvMisses := 0, 0
	seqLen := inputLen + tokensToGenerate
	if seqLen > e.config.SeqLen {
		seqLen = e.config.SeqLen
	}

	allTokens := make([]int, 0, seqLen)
	allTokens = append(allTokens, inputTokens...)

	layerStart := time.Now()
	var logits []float32
	var err error
	for pos := 0; pos < inputLen; pos++ {
		token := inputTokens[pos]
		logits, err = e.forward(token, pos, allTokens)
		if err != nil {
			cudaEngineFailed.WithLabelValues(e.config.Architecture, "forward_failed").Inc()
			return nil, fmt.Errorf("forward pass failed at position %d: %w", pos, err)
		}
		layerLatency := time.Since(layerStart).Seconds()
		cudaLayerLatency.WithLabelValues(e.config.Architecture, "prompt").Observe(layerLatency)
		layerStart = time.Now()
	}

	for gen := 0; gen < tokensToGenerate && len(allTokens) < seqLen; gen++ {
		if logitsCallback != nil {
			logitsCallback(logits)
		}

		samplingStart := time.Now()
		nextToken := sampler.Sample(logits, allTokens)
		cudaSamplingTime.WithLabelValues(e.config.Architecture).Observe(time.Since(samplingStart).Seconds())

		allTokens = append(allTokens, nextToken)
		result = append(result, nextToken)
		cudaTokensGenerated.WithLabelValues(e.config.Architecture).Inc()

		if tokenCallback != nil {
			tokenCallback(nextToken)
		}

		// Prepare logits for next iteration
		currentPos := len(allTokens) - 1
		logits, err = e.forward(nextToken, currentPos, allTokens)
		if err != nil {
			cudaEngineFailed.WithLabelValues(e.config.Architecture, "forward_failed").Inc()
			return nil, fmt.Errorf("forward pass failed at generation step %d: %w", gen, err)
		}

		layerLatency := time.Since(layerStart).Seconds()
		cudaLayerLatency.WithLabelValues(e.config.Architecture, fmt.Sprintf("gen_%d", gen)).Observe(layerLatency)
		layerStart = time.Now()
	}

	elapsed := time.Since(startTime)
	tokensPerSecond := float64(len(result)) / elapsed.Seconds()

	log.Printf("Generated %d tokens in %.2fs (%.1f t/s)", len(result), elapsed.Seconds(), tokensPerSecond)

	cudaInferenceDuration.WithLabelValues(e.config.Architecture).Observe(elapsed.Seconds())
	cudaTokensPerSecond.WithLabelValues(e.config.Architecture).Observe(tokensPerSecond)
	cudaKVCacheHits.WithLabelValues(e.config.Architecture).Add(float64(kvHits))
	cudaKVCacheMisses.WithLabelValues(e.config.Architecture).Add(float64(kvMisses))

	if e.cuda != nil {
		cudaMemoryUsage.WithLabelValues(e.config.Architecture).Set(float64(device.CUDAAllocatedBytes()))
	}

	if e.tokenizer != nil && len(result) > 0 {
		text := e.tokenizer.Decode(result)
		log.Printf("Output: %s", text)
	}

	return result, nil
}

func (e *cudaEngine) GetSeqCachePos(seqID string) int {
	return 0
}

func (e *cudaEngine) ForwardDraft(tokens []int) ([][]float32, error) {
	return nil, fmt.Errorf("ForwardDraft not implemented for CUDA engine")
}

func (e *cudaEngine) RollbackKV(seqID string, newPos int) error {
	return nil
}
