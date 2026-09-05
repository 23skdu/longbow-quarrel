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

	PromptCache  *PromptCache
	seqKVCaches  map[string]*CPUKVCache
	kvMu         sync.Mutex

	BatchManager *ContinuousBatchManager
	stopChan     chan struct{}
	doneChan     chan struct{}
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
		PromptCache:  NewPromptCache(),
		seqKVCaches:  make(map[string]*CPUKVCache),
		BatchManager: NewContinuousBatchManager(),
		stopChan:     make(chan struct{}),
		doneChan:     make(chan struct{}),
	}

	go e.runBatchLoop()

	return e, nil
}

func (e *CPUEngine) Config() config.Config {
	return e.config
}

func (e *CPUEngine) GetSeqCachePos(seqID string) int {
	e.kvMu.Lock()
	defer e.kvMu.Unlock()
	if c, ok := e.seqKVCaches[seqID]; ok && c != nil {
		kvDim := e.config.KVHeads * e.config.HeadDim
		if kvDim > 0 && len(c.Keys) > 0 && len(c.Keys[0]) > 0 {
			return len(c.Keys[0]) / kvDim
		}
	}
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

		// 1. Pull active sequences with prompt prefix cache support
		desc, _ := e.BatchManager.Step(16, e.cache, e.PromptCache)
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

			// Dynamic EOS & termination handling
			isEOS := (e.tok != nil && e.tok.IsEOS(token)) || token == e.config.EOSTokenID || token == 2
			if isEOS || len(seq.Tokens) >= seq.MaxTokens {
				select {
				case seq.Result <- seq.Tokens:
				default:
				}
				e.BatchManager.CompleteSequence(seq.ID, e.cache)
				e.kvMu.Lock()
				delete(e.seqKVCaches, fmt.Sprintf("seq-%d", seq.ID))
				e.kvMu.Unlock()
			}
		}
	}
}

func (e *CPUEngine) ForwardBatch(desc *BatchDescriptor) ([]*device.Tensor, error) {
	batchSize := len(desc.Sequences)
	results := make([]*device.Tensor, batchSize)

	var wg sync.WaitGroup
	wg.Add(batchSize)

	for i := range desc.Sequences {
		go func(idx int) {
			defer wg.Done()

			seq := desc.Sequences[idx]
			seqIDStr := fmt.Sprintf("seq-%d", seq.ID)

			start := desc.Offsets[idx]
			var end int
			if idx < batchSize-1 {
				end = desc.Offsets[idx+1]
			} else {
				end = len(desc.Tokens)
			}

			seqTokens := desc.Tokens[start:end]
			basePos := desc.ContextLens[idx]

			e.kvMu.Lock()
			if e.seqKVCaches == nil {
				e.seqKVCaches = make(map[string]*CPUKVCache)
			}
			kvCache, ok := e.seqKVCaches[seqIDStr]
			if !ok || kvCache == nil {
				kvCache = NewCPUKVCache(e.config.Layers)
				e.seqKVCaches[seqIDStr] = kvCache
			}
			e.kvMu.Unlock()

			hiddenSize := e.config.Dim
			if hiddenSize <= 0 {
				hiddenSize = 576
			}

			var logits []float32
			for t, tok := range seqTokens {
				pos := basePos + t
				hidden := e.weights.GetTokenEmbedding(tok, hiddenSize)

				for layerIdx := 0; layerIdx < e.config.Layers; layerIdx++ {
					hidden = ApplyLayerCPUKV(e.weights, hidden, layerIdx, pos, kvCache, e.config)
				}

				// Only compute output norm and logits for the last token in chunk
				if t == len(seqTokens)-1 {
					if e.weights.OutputNorm != nil {
						normed := make([]float32, len(hidden))
						simd.RMSNorm(hidden, e.weights.OutputNorm, normed, 1, len(hidden), e.config.Eps)
						hidden = normed
					}

					if len(e.weights.Output) > 0 || e.weights.RawOutput != nil {
						logits = e.weights.MatVec(e.weights.Output, e.weights.RawOutput, hidden)
					} else {
						logits = hidden
					}
				}
			}

			res := e.ctx.NewTensorFP32(1, len(logits))
			_ = res.LoadFrom(logits)
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
	cpuEng := newEngine.(*CPUEngine)
	e.kvMu.Lock()
	e.model = cpuEng.model
	e.config = cpuEng.config
	e.weights = cpuEng.weights
	e.tok = cpuEng.tok
	e.ctx = cpuEng.ctx
	e.cache = cpuEng.cache
	e.PromptCache = cpuEng.PromptCache
	e.seqKVCaches = cpuEng.seqKVCaches
	e.BatchManager = cpuEng.BatchManager
	e.stopChan = cpuEng.stopChan
	e.doneChan = cpuEng.doneChan
	e.kvMu.Unlock()
	return nil
}

func (e *CPUEngine) forward(tokens []int) []float32 {
	hiddenSize := e.config.Dim
	if hiddenSize <= 0 {
		hiddenSize = 576
	}

	if len(tokens) == 0 {
		return make([]float32, hiddenSize)
	}

	kvCache := NewCPUKVCache(e.config.Layers)
	var hidden []float32

	for pos, tok := range tokens {
		hidden = e.weights.GetTokenEmbedding(tok, hiddenSize)
		for layerIdx := 0; layerIdx < e.config.Layers; layerIdx++ {
			hidden = ApplyLayerCPUKV(e.weights, hidden, layerIdx, pos, kvCache, e.config)
		}
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
	return ApplyLayerCPU(e.weights, input, layerIdx, e.config)
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

func (e *CPUEngine) Close() {
	if e.stopChan != nil {
		select {
		case <-e.stopChan:
		default:
			close(e.stopChan)
		}
	}
	if e.doneChan != nil {
		<-e.doneChan
	}
	if e.cache != nil {
		e.cache.Free()
	}
	if e.ctx != nil {
		e.ctx.Free()
	}
	if e.model != nil {
		_ = e.model.Close()
	}
	e.kvMu.Lock()
	e.seqKVCaches = nil
	e.kvMu.Unlock()
	logger.Log.Info("CPU engine closed")
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
	var cacheErr error
	if e.cache != nil {
		cacheErr = e.cache.RollbackKV(seqID, newPos)
	}
	e.kvMu.Lock()
	defer e.kvMu.Unlock()
	if c, ok := e.seqKVCaches[seqID]; ok && c != nil {
		kvDim := e.config.KVHeads * e.config.HeadDim
		if kvDim > 0 {
			targetLen := newPos * kvDim
			for l := range c.Keys {
				if targetLen <= len(c.Keys[l]) {
					c.Keys[l] = c.Keys[l][:targetLen]
				}
				if targetLen <= len(c.Values[l]) {
					c.Values[l] = c.Values[l][:targetLen]
				}
			}
		}
	}
	return cacheErr
}

func (e *CPUEngine) LoadAdapter(path, id string) error {
	return nil // Not supported on CPU for now
}
