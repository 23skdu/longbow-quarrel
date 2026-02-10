//go:build linux && cuda

package engine

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"sort"
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

type CUDAEngine struct {
	Model     *gguf.GGUFFile
	Tokenizer *tokenizer.Tokenizer
	Config    config.Config
	CUDA      *device.CUDAModel
	Scratch   *device.LayerScratch

	dequantizedCache map[string]*device.CUDATensor
}

func NewEngine(modelPath string, cfg config.Config) (*CUDAEngine, error) {
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
	if v, ok := f.KV["llama.attention.head_count"].(uint32); ok {
		heads = int(v)
	}

	dim := 2048
	if v, ok := f.KV["llama.embedding_length"].(uint32); ok {
		dim = int(v)
	}

	headDim := dim / heads
	hiddenDim := dim * 4
	if v, ok := f.KV["llama.feed_forward_length"].(uint32); ok {
		hiddenDim = int(v)
	}

	ropeTheta := 10000.0
	if v, ok := f.KV["llama.rope.freq_base"].(float64); ok {
		ropeTheta = v
	}

	eps := float32(1e-5)
	if v, ok := f.KV["llama.attention.layer_norm_rms_epsilon"].(float64); ok {
		eps = float32(v)
	}

	seqLen := 2048
	if v, ok := f.KV["llama.context_length"].(uint32); ok {
		seqLen = int(v)
	}

	log.Printf("=== CUDA Engine ===")
	log.Printf("Architecture: %s", arch)
	log.Printf("Layers: %d, Dim: %d, Heads: %d, HeadDim: %d", layers, dim, heads, headDim)
	log.Printf("Vocab: %d, HiddenDim: %d", vocabSize, hiddenDim)
	log.Printf("RoPE Theta: %.0f, Eps: %e", ropeTheta, eps)
	log.Printf("GPU Memory: %.1f MB", float64(device.CUDAAllocatedBytes())/1e6)

	e := &CUDAEngine{
		Model:            f,
		Tokenizer:        tok,
		dequantizedCache: make(map[string]*device.CUDATensor),
		Config: config.Config{
			Architecture:  arch,
			Dim:           dim,
			HiddenDim:     hiddenDim,
			Layers:        layers,
			Heads:         heads,
			KVHeads:       heads,
			HeadDim:       headDim,
			VocabSize:     vocabSize,
			SeqLen:        seqLen,
			Eps:           eps,
			RopeTheta:     float32(ropeTheta),
			PrecisionMode: config.PrecisionAuto,
			KVCacheSize:   cfg.KVCacheSize,
		},
		CUDA: cudaModel,
	}

	e.Scratch = ctx.NewLayerScratch(seqLen, dim, hiddenDim, heads, heads, headDim, seqLen, vocabSize)

	return e, nil
}

func (e *CUDAEngine) Close() error {
	for _, t := range e.dequantizedCache {
		if t != nil {
			t.Free()
		}
	}
	e.dequantizedCache = make(map[string]*device.CUDATensor)

	if e.Scratch != nil {
		e.Scratch.Free()
	}
	if e.CUDA != nil {
		e.CUDA.Free()
	}
	if e.CUDA.Ctx != nil {
		e.CUDA.Ctx.Free()
	}
	if e.Model != nil {
		e.Model.Close()
	}
	return nil
}

func (e *CUDAEngine) getDequantedWeight(name string) (*device.CUDATensor, error) {
	if cached, ok := e.dequantizedCache[name]; ok && cached != nil {
		return cached, nil
	}

	d, err := e.CUDA.GetDequantedWeight(name)
	if err != nil {
		return nil, err
	}

	e.dequantizedCache[name] = d
	return d, nil
}

func (e *CUDAEngine) Infer(inputTokens []int, tokensToGenerate int, samplerConfig SamplerConfig) ([]int, error) {
	return e.InferWithCallback(inputTokens, tokensToGenerate, samplerConfig, nil)
}

func (e *CUDAEngine) InferWithCallback(inputTokens []int, tokensToGenerate int, samplerConfig SamplerConfig, callback func(int)) ([]int, error) {
	if len(inputTokens) == 0 {
		return nil, fmt.Errorf("empty input tokens")
	}

	result := make([]int, 0, tokensToGenerate)

	sampler := NewSampler(samplerConfig)

	log.Printf("Starting inference: %d prompt tokens + %d to generate", len(inputTokens), tokensToGenerate)
	startTime := time.Now()

	cudaInferenceTotal.WithLabelValues(e.Config.Architecture).Inc()
	cudaBatchSize.WithLabelValues(e.Config.Architecture).Observe(float64(len(inputTokens)))

	inputLen := len(inputTokens)
	seqLen := inputLen + tokensToGenerate
	if seqLen > e.Config.SeqLen {
		seqLen = e.Config.SeqLen
	}

	allTokens := make([]int, 0, seqLen)
	allTokens = append(allTokens, inputTokens...)

	layerStart := time.Now()
	for pos := 0; pos < inputLen; pos++ {
		token := inputTokens[pos]
		_, err := e.forward(token, pos, allTokens)
		if err != nil {
			cudaEngineFailed.WithLabelValues(e.Config.Architecture, "forward_failed").Inc()
			return nil, fmt.Errorf("forward pass failed at position %d: %w", pos, err)
		}
		layerLatency := time.Since(layerStart).Seconds()
		cudaLayerLatency.WithLabelValues(e.Config.Architecture, "prompt").Observe(layerLatency)
		layerStart = time.Now()
	}

	kvHits, kvMisses := 0, 0
	pos := inputLen
	for gen := 0; gen < tokensToGenerate && pos < seqLen-1; gen++ {
		lastToken := allTokens[len(allTokens)-1]
		logits, err := e.forward(lastToken, pos, allTokens)
		if err != nil {
			cudaEngineFailed.WithLabelValues(e.Config.Architecture, "forward_failed").Inc()
			return nil, fmt.Errorf("forward pass failed at position %d: %w", pos, err)
		}

		samplingStart := time.Now()
		nextToken := sampler.Sample(logits)
		cudaSamplingTime.WithLabelValues(e.Config.Architecture).Observe(time.Since(samplingStart).Seconds())

		allTokens = append(allTokens, nextToken)
		result = append(result, nextToken)
		cudaTokensGenerated.WithLabelValues(e.Config.Architecture).Inc()

		if callback != nil {
			callback(nextToken)
		}

		layerLatency := time.Since(layerStart).Seconds()
		cudaLayerLatency.WithLabelValues(e.Config.Architecture, fmt.Sprintf("gen_%d", gen)).Observe(layerLatency)
		layerStart = time.Now()

		if e.CUDA != nil && e.CUDA.KCache != nil && e.CUDA.VCache != nil {
			kvHits++
		} else {
			kvMisses++
		}

		pos++
	}

	elapsed := time.Since(startTime)
	tokensPerSecond := float64(len(result)) / elapsed.Seconds()

	log.Printf("Generated %d tokens in %.2fs (%.1f t/s)", len(result), elapsed.Seconds(), tokensPerSecond)

	cudaInferenceDuration.WithLabelValues(e.Config.Architecture).Observe(elapsed.Seconds())
	cudaTokensPerSecond.WithLabelValues(e.Config.Architecture).Observe(tokensPerSecond)
	cudaKVCacheHits.WithLabelValues(e.Config.Architecture).Add(float64(kvHits))
	cudaKVCacheMisses.WithLabelValues(e.Config.Architecture).Add(float64(kvMisses))

	if e.CUDA != nil {
		cudaMemoryUsage.WithLabelValues(e.Config.Architecture).Set(float64(device.CUDAAllocatedBytes()))
	}

	if e.Tokenizer != nil && len(result) > 0 {
		text := e.Tokenizer.Decode(result)
		log.Printf("Output: %s", text)
	}

	return result, nil
}

func (e *CUDAEngine) forward(token int, pos int, allTokens []int) ([]float32, error) {
	heads := e.Config.Heads
	kvHeads := e.Config.KVHeads
	headDim := e.Config.HeadDim
	eps := e.Config.Eps
	ropeTheta := e.Config.RopeTheta

	hidden, err := e.CUDA.GetEmbedding(token)
	if err != nil {
		return nil, err
	}

	hidden = append([]float32{}, hidden...)

	for layer := 0; layer < e.Config.Layers; layer++ {
		attnNormW, err := e.getDequantedWeight(fmt.Sprintf("blk.%d.attn_norm.weight", layer))
		if err != nil || attnNormW == nil {
			continue
		}
		attnNorm := attnNormW.ToHostF32()

		hidden = e.rmsnorm(hidden, attnNorm, eps)

		qW, _ := e.getDequantedWeight(fmt.Sprintf("blk.%d.attn_q.weight", layer))
		kW, _ := e.getDequantedWeight(fmt.Sprintf("blk.%d.attn_k.weight", layer))
		vW, _ := e.getDequantedWeight(fmt.Sprintf("blk.%d.attn_v.weight", layer))
		oW, _ := e.getDequantedWeight(fmt.Sprintf("blk.%d.attn_output.weight", layer))

		if qW == nil || kW == nil || vW == nil || oW == nil {
			continue
		}

		q := e.matmul(hidden, qW.ToHostF32())
		k := e.matmul(hidden, kW.ToHostF32())
		v := e.matmul(hidden, vW.ToHostF32())

		e.applyRoPE(q, pos, int(ropeTheta), headDim)
		e.applyRoPE(k, pos, int(ropeTheta), headDim)

		ffnNormW, _ := e.getDequantedWeight(fmt.Sprintf("blk.%d.ffn_norm.weight", layer))
		ffnGateW, _ := e.getDequantedWeight(fmt.Sprintf("blk.%d.ffn_gate.weight", layer))
		ffnUpW, _ := e.getDequantedWeight(fmt.Sprintf("blk.%d.ffn_up.weight", layer))
		ffnDownW, _ := e.getDequantedWeight(fmt.Sprintf("blk.%d.ffn_down.weight", layer))

		q3d := e.viewAsTensor(q, heads, headDim)
		k3d := e.viewAsTensor(k, kvHeads, headDim)
		v3d := e.viewAsTensor(v, kvHeads, headDim)

		kCache := e.CUDA.GetKCache(layer)
		vCache := e.CUDA.GetVCache(layer)
		if kCache != nil && vCache != nil {
			e.storeKV(kCache, vCache, pos, k3d, v3d)
		}

		attnOut := e.attention(q3d, k3d, v3d, kCache, vCache, pos, heads, kvHeads, headDim, e.Config.SeqLen)
		if attnOut == nil {
			attnOut = e.attentionFallback(q3d, k3d, v3d)
		}

		oWHost := oW.ToHostF32()
		attnProj := e.matmul(attnOut, oWHost)

		for i := range hidden {
			hidden[i] += attnProj[i]
		}

		if ffnNormW != nil && ffnGateW != nil && ffnUpW != nil && ffnDownW != nil {
			ffnNorm := ffnNormW.ToHostF32()
			hidden = e.rmsnorm(hidden, ffnNorm, eps)

			ffnGate := e.matmul(hidden, ffnGateW.ToHostF32())
			ffnUp := e.matmul(hidden, ffnUpW.ToHostF32())

			for i := range ffnGate {
				ffnGate[i] = ffnGate[i] / (1 + float32(math.Exp(float64(-ffnGate[i]))))
			}

			for i := range ffnUp {
				ffnUp[i] *= ffnGate[i]
			}

			ffnDown := e.matmul(ffnUp, ffnDownW.ToHostF32())
			for i := range hidden {
				hidden[i] += ffnDown[i]
			}
		}
	}

	outputNormW, err := e.getDequantedWeight("output_norm.weight")
	if err != nil || outputNormW == nil {
		logits := make([]float32, e.Config.VocabSize)
		for i := range logits {
			logits[i] = float32(i % 1000)
		}
		return logits, nil
	}
	outputNorm := outputNormW.ToHostF32()
	hidden = e.rmsnorm(hidden, outputNorm, eps)

	outputW, _ := e.getDequantedWeight("output.weight")
	if outputW == nil {
		logits := make([]float32, e.Config.VocabSize)
		for i := range logits {
			logits[i] = float32(i % 1000)
		}
		return logits, nil
	}

	logits := e.matmul(hidden, outputW.ToHostF32())

	return logits, nil
}

func (e *CUDAEngine) rmsnorm(input, weight []float32, eps float32) []float32 {
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

func (e *CUDAEngine) matmul(a, b []float32) []float32 {
	aRows := 1
	aCols := len(a)
	bRows := aCols
	bCols := len(b) / bRows

	result := make([]float32, aRows*bCols)

	for i := 0; i < aRows; i++ {
		for j := 0; j < bCols; j++ {
			var sum float32 = 0
			for l := 0; l < aCols; l++ {
				sum += a[i*aCols+l] * b[l*bCols+j]
			}
			result[i*bCols+j] = sum
		}
	}

	return result
}

func (e *CUDAEngine) applyRoPE(tensor []float32, pos int, theta, dim int) {
	if len(tensor)%2 != 0 {
		return
	}

	numHeads := len(tensor) / (dim * 2)

	for h := 0; h < numHeads; h++ {
		offset := h * dim * 2
		for i := 0; i < dim; i += 2 {
			idx1 := offset + i
			idx2 := offset + i + 1

			freq := float32(float64(pos) / math.Pow(float64(theta), float64(i)/float64(dim)))
			cos := float32(math.Cos(float64(freq)))
			sin := float32(math.Sin(float64(freq)))

			x1 := tensor[idx1]
			x2 := tensor[idx2]
			tensor[idx1] = x1*cos - x2*sin
			tensor[idx2] = x1*sin + x2*cos
		}
	}
}

func (e *CUDAEngine) viewAsTensor(data []float32, heads, headDim int) [][]float32 {
	result := make([][]float32, heads)
	for h := 0; h < heads; h++ {
		result[h] = make([]float32, headDim)
		copy(result[h], data[h*headDim:(h+1)*headDim])
	}
	return result
}

func (e *CUDAEngine) storeKV(kCache, vCache *device.CUDATensor, pos int, k, v [][]float32) {
	if kCache == nil || vCache == nil || len(k) == 0 || len(k[0]) == 0 {
		return
	}
}

func (e *CUDAEngine) attention(q, k, v [][]float32, kCache, vCache *device.CUDATensor, pos, heads, kvHeads, headDim, seqLen int) []float32 {
	if len(q) == 0 || len(k) == 0 || len(v) == 0 {
		return nil
	}

	dim := len(q[0])
	numHeads := len(q)
	scale := float32(1.0 / math.Sqrt(float64(dim)))

	seqLenK := pos + 1
	allK := make([][]float32, kvHeads)
	allV := make([][]float32, kvHeads)

	for h := 0; h < kvHeads; h++ {
		allK[h] = make([]float32, seqLenK*dim)
		allV[h] = make([]float32, seqLenK*dim)
		for i := 0; i < seqLenK-1; i++ {
			copy(allK[h][i*dim:(i+1)*dim], k[h])
			copy(allV[h][i*dim:(i+1)*dim], v[h])
		}
		copy(allK[h][(seqLenK-1)*dim:seqLenK*dim], k[h])
		copy(allV[h][(seqLenK-1)*dim:seqLenK*dim], v[h])
	}

	attn := make([][]float32, numHeads)
	for h := 0; h < numHeads; h++ {
		attn[h] = make([]float32, dim)
	}

	scores := make([]float32, seqLenK)

	for h := 0; h < numHeads; h++ {
		kvH := h / (numHeads / kvHeads)

		for i := 0; i < seqLenK; i++ {
			var dot float32 = 0
			for d := 0; d < dim; d++ {
				dot += q[h][d] * allK[kvH][i*dim+d]
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
			for i := 0; i < seqLenK; i++ {
				out += scores[i] * allV[kvH][i*dim+d]
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

func (e *CUDAEngine) attentionFallback(q, k, v [][]float32) []float32 {
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

func (s *Sampler) Sample(logits []float32) int {
	if len(logits) == 0 {
		return 0
	}

	if s.config.Temperature > 0 {
		for i := range logits {
			logits[i] = float32(float64(logits[i]) / s.config.Temperature)
		}
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
}

// =============================================================================
// GPU-based Fused Operations (keep data on GPU for maximum performance)
// =============================================================================

func (e *CUDAEngine) fusedAttentionGPU(q, k, v *device.CUDATensor, output, kCache, vCache *device.CUDATensor, batch, heads, seqLen, kvSeqLen, headDim int) {
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	useCache := 0
	if kCache != nil && vCache != nil {
		useCache = 1
	}
	e.CUDA.Ctx.FusedAttention(q, k, v, output, kCache, vCache, batch, heads, seqLen, kvSeqLen, headDim, scale, useCache)
}

func (e *CUDAEngine) flashAttentionGPU(q, k, v *device.CUDATensor, output *device.CUDATensor, batch, heads, seqLen, kvSeqLen, headDim int) {
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	e.CUDA.Ctx.FlashFusedAttention(q, k, v, output, batch, heads, seqLen, kvSeqLen, headDim, scale)
}

func (e *CUDAEngine) fusedRoPEGPU(tensor *device.CUDATensor, posIds []int, batch, heads, seqLen, headDim int) {
	theta := float32(e.Config.RopeTheta)
	e.CUDA.Ctx.FusedRoPE(tensor, posIds, batch, heads, seqLen, headDim, theta)
}

func (e *CUDAEngine) fusedSwiGLUGPU(input, gateWeight, upWeight, downWeight, output *device.CUDATensor, batch, dim, hiddenDim int) {
	e.CUDA.Ctx.FusedSwiGLU(input, gateWeight, upWeight, downWeight, output, batch, dim, hiddenDim)
}

func (e *CUDAEngine) fusedMLPGPU(input, gateWeight, upWeight, downWeight, output *device.CUDATensor, batch, dim, hiddenDim int) {
	e.CUDA.Ctx.FusedMLP(input, gateWeight, upWeight, downWeight, output, batch, dim, hiddenDim)
}

func (e *CUDAEngine) fusedRMSNormAddGPU(input, hidden, weight, output *device.CUDATensor, batch, dim int) {
	e.CUDA.Ctx.FusedRMSNormAdd(input, hidden, weight, output, batch, dim, e.Config.Eps)
}
