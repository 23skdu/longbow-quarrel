package engine

import (
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/mathx"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
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
	ctx     *device.Context
	cache   *PagedKVCache

	PromptCache *PromptCache
	seqKVCaches map[string]*CPUKVCache
	kvMu        sync.Mutex

	// adapters tracks LoRA adapters merged into the CPU weights. The CPU
	// backend fuses the delta into the base weights at load time, so it keeps
	// its own registry rather than using LoRAManager.
	adapters   []AdapterInfo
	adaptersMu sync.Mutex

	// stopTokens is the configured stop-token set, built once at construction.
	stopTokens map[int]struct{}

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
	if modelCfg.HeadDim > 0 {
		cfg.HeadDim = modelCfg.HeadDim
	} else if cfg.HeadDim <= 0 {
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
	if modelCfg.Eps > 0 {
		cfg.Eps = modelCfg.Eps
	} else if cfg.Eps <= 0 {
		cfg.Eps = modelCfg.Eps
	}
	if modelCfg.RopeTheta > 0 {
		cfg.RopeTheta = modelCfg.RopeTheta
	} else if cfg.RopeTheta <= 0 {
		cfg.RopeTheta = modelCfg.RopeTheta
	}
	if cfg.Architecture == "" {
		cfg.Architecture = modelCfg.Architecture
	}
	if modelCfg.IsGemma4 {
		cfg.IsGemma4 = true
		cfg.Gemma4SlidingWindowSize = modelCfg.Gemma4SlidingWindowSize
		cfg.Gemma4SlidingRoPETheta = modelCfg.Gemma4SlidingRoPETheta
		cfg.Gemma4FullRoPETheta = modelCfg.Gemma4FullRoPETheta
		cfg.Gemma4PartialRoPEFactor = modelCfg.Gemma4PartialRoPEFactor
		cfg.Gemma4SlidingHeadDim = modelCfg.Gemma4SlidingHeadDim
		cfg.Gemma4FullHeadDim = modelCfg.Gemma4FullHeadDim
		cfg.Gemma4SharedKVLayers = modelCfg.Gemma4SharedKVLayers
		cfg.Gemma4SlidingPattern = modelCfg.Gemma4SlidingPattern
		cfg.FinalLogitSoftcapping = modelCfg.FinalLogitSoftcapping
		cfg.Eps = modelCfg.Eps
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
	// Register the EOS token declared by the model metadata, so the decode
	// loop only has to consult the tokenizer. The engines used to hardcode
	// `token == cfg.EOSTokenID` here, which ignored any additional EOS ids
	// the model declares.
	if cfg.EOSTokenID > 0 {
		tok.AddEOSTokenID(cfg.EOSTokenID)
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
		stopTokens:   stopTokenSet(cfg.StopTokens),
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

			// Update Sequence State
			chunkLen := len(desc.Tokens) - desc.Offsets[i]
			if i < len(desc.Offsets)-1 {
				chunkLen = desc.Offsets[i+1] - desc.Offsets[i]
			}

			// If this is an intermediate prefill chunk, just advance position and continue
			if i < len(desc.IsDecode) && !desc.IsDecode[i] && seq.Pos+chunkLen < seq.PromptLen {
				seq.Pos += chunkLen
				continue
			}

			if seq.LogitsCallback != nil {
				seq.LogitsCallback(logits)
			}

			sampler := NewSampler(seq.Config)
			token := sampler.Sample(logits, seq.Tokens)

			seq.Tokens = append(seq.Tokens, token)
			seq.Pos += chunkLen

			if seq.TokenCallback != nil {
				seq.TokenCallback(token)
			}

			// Dynamic EOS & termination handling. The tokenizer knows every
			// EOS id the model declares (including the conventional 2 and
			// the one from the model metadata).
			isEOS := e.tok != nil && e.tok.IsEOS(token)
			if isEOS || isStopToken(e.stopTokens, token) || len(seq.Tokens) >= seq.MaxTokens {
				select {
				case seq.Result <- seq.Tokens:
				default:
				}
				// Populate PromptCache so repeated/shared prompts skip prefill
				if e.PromptCache != nil && e.cache != nil {
					promptLen := seq.Pos
					if promptLen > 0 && promptLen <= len(seq.Tokens) {
						blocks := e.cache.GetSequenceBlocks(fmt.Sprintf("%d", seq.ID))
						if len(blocks) > 0 {
							e.PromptCache.Insert(seq.Tokens[:promptLen], blocks)
						}
					}
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
			if !ok {
				windowSize := e.config.WindowSize
				if windowSize > 0 {
					kvCache = NewCPUKVCacheWithWindow(e.config.Layers, windowSize)
				} else {
					kvCache = NewCPUKVCache(e.config.Layers)
				}
				e.seqKVCaches[seqIDStr] = kvCache
			}

			e.kvMu.Unlock()

			hiddenSize := e.config.Dim
			if hiddenSize <= 0 {
				hiddenSize = 576
			}

			// Pre-allocate Gemma4 layer buffers or standard CPU layer scratch once per sequence
			var gemma4Buf *Gemma4LayerBuf
			var layerScratch *CPULayerScratch
			if e.config.IsGemma4 {
				gemma4Buf = NewGemma4LayerBuf(hiddenSize, e.config.Gemma4FullHeadDim, e.config.Heads, e.config.HiddenDim, e.config.KVCacheSize)
			} else {
				layerScratch = NewCPULayerScratch(hiddenSize, e.config.Heads, e.config.HeadDim)
			}

			// The activation log is a debugging aid: it records the
			// embedding, the per-layer activations and the final logits
			// so that a broken layer can be found.
			var actLog *ActivationLogger
			if e.config.ActivationLogPath != "" {
				actLog = NewActivationLogger()
				actLog.Enable("", seqTokens)
			}
			var collector *LayerActivationCollector
			if actLog != nil {
				collector = &LayerActivationCollector{}
				layerScratch.Collector = collector
			}

			var logits []float32
			for t, tok := range seqTokens {
				pos := basePos + t
				hidden := e.weights.GetTokenEmbedding(tok, hiddenSize)
				if actLog != nil && t == 0 {
					actLog.LogEmbedding(hidden)
				}

				if e.config.IsGemma4 {
					scaleEmb := float32(mathx.Sqrt(float64(hiddenSize)))
					for j := range hidden {
						hidden[j] *= scaleEmb
					}
					ple := e.weights.ComputeGemma4PLE(tok, hidden, e.config.Layers)
					for layerIdx := 0; layerIdx < e.config.Layers; layerIdx++ {
						hidden = ApplyGemma4LayerCPU(e.weights, hidden, layerIdx, pos, kvCache, ple[layerIdx], e.config, gemma4Buf)
					}
				} else {
					for layerIdx := 0; layerIdx < e.config.Layers; layerIdx++ {
						hidden = ApplyLayerCPUKVWithScratch(e.weights, hidden, layerIdx, pos, kvCache, e.config, layerScratch)
						if actLog != nil {
							logLayerActivations(actLog, collector)
						}
					}
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

					if e.config.FinalLogitSoftcapping > 0 {
						capVal := e.config.FinalLogitSoftcapping
						for j := range logits {
							logits[j] = capVal * float32(mathx.Tanh(float64(logits[j]/capVal)))
						}
					}

					// Audit the distribution. A model that produces NaNs
					// or a flat distribution says nothing about it, and
					// the audit is cheap next to the forward pass.
					auditLogits(logits)

					if actLog != nil {
						actLog.LogLogits(logits, topTokenIDs(logits, 5))
						if err := actLog.SaveToFile(e.config.ActivationLogPath); err != nil {
							logger.Log.Warn("failed to write the activation log", "error", err)
						}
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

func (e *CPUEngine) Forward(tokens []int) []float32 {
	hiddenSize := e.config.Dim
	if hiddenSize <= 0 {
		hiddenSize = 576
	}

	if len(tokens) == 0 {
		return make([]float32, hiddenSize)
	}

	kvCache := NewCPUKVCache(e.config.Layers)
	var hidden []float32

	// Pre-allocate Gemma4 layer buffers or standard CPU layer scratch
	var gemma4Buf *Gemma4LayerBuf
	var layerScratch *CPULayerScratch
	if e.config.IsGemma4 {
		gemma4Buf = NewGemma4LayerBuf(hiddenSize, e.config.Gemma4FullHeadDim, e.config.Heads, e.config.HiddenDim, e.config.KVCacheSize)
	} else {
		layerScratch = NewCPULayerScratch(hiddenSize, e.config.Heads, e.config.HeadDim)
	}

	for pos, tok := range tokens {
		hidden = e.weights.GetTokenEmbedding(tok, hiddenSize)
		if e.config.IsGemma4 {
			scaleEmb := float32(mathx.Sqrt(float64(hiddenSize)))
			for j := range hidden {
				hidden[j] *= scaleEmb
			}
			ple := e.weights.ComputeGemma4PLE(tok, hidden, e.config.Layers)
			for layerIdx := 0; layerIdx < e.config.Layers; layerIdx++ {
				hidden = ApplyGemma4LayerCPU(e.weights, hidden, layerIdx, pos, kvCache, ple[layerIdx], e.config, gemma4Buf)
			}
		} else {
			for layerIdx := 0; layerIdx < e.config.Layers; layerIdx++ {
				hidden = ApplyLayerCPUKVWithScratch(e.weights, hidden, layerIdx, pos, kvCache, e.config, layerScratch)
			}
		}
	}

	if e.weights.OutputNorm != nil {
		normed := make([]float32, len(hidden))
		simd.RMSNorm(hidden, e.weights.OutputNorm, normed, 1, len(hidden), e.config.Eps)
		hidden = normed
	}

	var logits []float32
	if len(e.weights.Output) > 0 || e.weights.RawOutput != nil {
		logits = e.weights.MatVec(e.weights.Output, e.weights.RawOutput, hidden)
	} else {
		logits = hidden
	}

	if e.config.FinalLogitSoftcapping > 0 {
		capVal := e.config.FinalLogitSoftcapping
		for j := range logits {
			logits[j] = capVal * float32(mathx.Tanh(float64(logits[j]/capVal)))
		}
	}

	return logits
}

// The CPU backend used to carry a second, parallel sampler here
// (RMSNormCPU, Sigmoid, ApplyTempCPU, ApplyTopKCPU, ApplyTopPCPU,
// softmaxCPU, SampleFromDistCPU). Every backend samples through
// internal/engine/sampler.go instead, so the copies had no production
// caller and could silently diverge from the implementation that is
// actually used. They have been removed; the shared Sampler is the single
// implementation.

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

// ForwardDraft performs a multi-token speculative forward pass on the CPU engine,
// returning one logit vector per draft position (up to draftCount tokens beyond the prompt).
// This enables the SpeculativeManager to compute real acceptance ratios.
func (e *CPUEngine) ForwardDraft(tokens []int) ([][]float32, error) {
	if len(tokens) == 0 || e.weights == nil {
		return nil, nil
	}
	if len(e.weights.TokenEmb) == 0 && e.weights.RawTokenEmb == nil {
		return nil, nil
	}

	hiddenSize := e.config.Dim
	if hiddenSize <= 0 {
		hiddenSize = 576
	}

	// Use up to the last draftCount token positions from the input token sequence.
	// For speculative decoding, we want logits at positions [len(tokens)-draftK .. len(tokens)-1].
	const draftCount = 4
	startIdx := len(tokens) - draftCount
	if startIdx < 0 {
		startIdx = 0
	}
	draftTokens := tokens[startIdx:]

	// Fresh KV cache for the draft pass (includes the full prefix context)
	kvCache := NewCPUKVCache(e.config.Layers)

	var results [][]float32

	// Process all tokens, collecting logits for draft positions
	for pos, tok := range tokens {
		hidden := e.weights.GetTokenEmbedding(tok, hiddenSize)
		for layerIdx := 0; layerIdx < e.config.Layers; layerIdx++ {
			hidden = ApplyLayerCPUKV(e.weights, hidden, layerIdx, pos, kvCache, e.config)
		}

		// Only collect logits for draft positions (last draftCount tokens)
		isDraftPos := pos >= len(tokens)-len(draftTokens)
		if isDraftPos {
			normed := hidden
			if e.weights.OutputNorm != nil {
				normed = make([]float32, len(hidden))
				simd.RMSNorm(hidden, e.weights.OutputNorm, normed, 1, len(hidden), e.config.Eps)
			}
			var logits []float32
			if len(e.weights.Output) > 0 || e.weights.RawOutput != nil {
				logits = e.weights.MatVec(e.weights.Output, e.weights.RawOutput, normed)
			} else {
				logits = normed
			}
			results = append(results, logits)
		}
	}

	return results, nil
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

// LoadAdapter implements CPU LoRA merge-on-load.
// The LoRA delta (A × B × alpha/rank) is fused directly into the CPUWeights
// float32 matrices at load time, so there is zero inference-time overhead.
// Supports .gguf sidecar format (tensor names: "blk.N.attn_q.lora_A.weight" etc.).
func (e *CPUEngine) LoadAdapter(path, id string) error {
	f, err := gguf.LoadFile(path)
	if err != nil {
		return fmt.Errorf("cpu lora: failed to load %s: %w", path, err)
	}
	defer func() { _ = f.Close() }()

	alpha := float32(8.0)
	if a, ok := f.KV["adapter.lora_alpha"].(float32); ok {
		alpha = a
	}

	// Group tensors by base name (e.g., "blk.0.attn_q")
	typeA := make(map[string]*gguf.TensorInfo)
	typeB := make(map[string]*gguf.TensorInfo)
	for _, t := range f.Tensors {
		switch {
		case strings.HasSuffix(t.Name, ".lora_A.weight"):
			typeA[strings.TrimSuffix(t.Name, ".lora_A.weight")] = t
		case strings.HasSuffix(t.Name, ".lora_B.weight"):
			typeB[strings.TrimSuffix(t.Name, ".lora_B.weight")] = t
		}
	}

	for base, tA := range typeA {
		tB, ok := typeB[base]
		if !ok {
			continue
		}
		dimIn := int(tA.Dimensions[0])  // #nosec G115
		rank := int(tA.Dimensions[1])   // #nosec G115
		dimOut := int(tB.Dimensions[1]) // #nosec G115

		aData, err2 := decodeTensorData(tA)
		if err2 != nil || len(aData) != dimIn*rank {
			continue
		}
		bData, err2 := decodeTensorData(tB)
		if err2 != nil || len(bData) != rank*dimOut {
			continue
		}

		// delta[i][j] = sum_r(A[i][r] * B[r][j]) * alpha / rank
		// A: [dimIn x rank], B: [rank x dimOut]  → delta: [dimOut x dimIn] (row-major weight shape)
		scale := alpha / float32(rank)
		delta := make([]float32, dimOut*dimIn)
		for o := 0; o < dimOut; o++ {
			for i := 0; i < dimIn; i++ {
				var s float32
				for r := 0; r < rank; r++ {
					s += aData[i*rank+r] * bData[r*dimOut+o]
				}
				delta[o*dimIn+i] = s * scale
			}
		}

		// Map base name to CPUWeights field
		var layer int
		layerKey := ""
		if n, err3 := fmt.Sscanf(base, "blk.%d.", &layer); n == 1 && err3 == nil {
			// Extract projection name
			suffix := base[len(fmt.Sprintf("blk.%d.", layer)):]
			switch suffix {
			case "attn_q", "self_attn.q_proj":
				layerKey = "attn_q"
			case "attn_k", "self_attn.k_proj":
				layerKey = "attn_k"
			case "attn_v", "self_attn.v_proj":
				layerKey = "attn_v"
			case "attn_output", "self_attn.o_proj":
				layerKey = "attn_o"
			case "ffn_gate", "mlp.gate_proj":
				layerKey = "ffn_gate"
			case "ffn_up", "mlp.up_proj":
				layerKey = "ffn_up"
			case "ffn_down", "mlp.down_proj":
				layerKey = "ffn_down"
			}
		}

		if layerKey == "" || layer >= e.config.Layers {
			continue
		}

		// Merge delta into the target weight slice
		e.mergeLoRADelta(layer, layerKey, delta)
	}

	// Record the adapter so ListAdapters can report it.
	e.adaptersMu.Lock()
	e.adapters = append(e.adapters, AdapterInfo{ID: id, Path: path, NumWeights: len(typeA)})
	e.adaptersMu.Unlock()

	return nil
}

// ListAdapters returns metadata for the LoRA adapters merged into the CPU
// weights, sorted by ID.
func (e *CPUEngine) ListAdapters() []AdapterInfo {
	e.adaptersMu.Lock()
	defer e.adaptersMu.Unlock()

	out := make([]AdapterInfo, len(e.adapters))
	copy(out, e.adapters)
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// mergeLoRADelta adds a LoRA delta matrix into the corresponding CPUWeights slice.
// This is safe to call only before inference starts (no mutex needed at load time).
func (e *CPUEngine) mergeLoRADelta(layer int, key string, delta []float32) {
	if e.weights == nil {
		return
	}
	var target []float32
	switch key {
	case "attn_q":
		if layer < len(e.weights.AttnQ) {
			target = e.weights.AttnQ[layer]
		}
	case "attn_k":
		if layer < len(e.weights.AttnK) {
			target = e.weights.AttnK[layer]
		}
	case "attn_v":
		if layer < len(e.weights.AttnV) {
			target = e.weights.AttnV[layer]
		}
	case "attn_o":
		if layer < len(e.weights.AttnO) {
			target = e.weights.AttnO[layer]
		}
	case "ffn_gate":
		if layer < len(e.weights.FfnGate) {
			target = e.weights.FfnGate[layer]
		}
	case "ffn_up":
		if layer < len(e.weights.FfnUp) {
			target = e.weights.FfnUp[layer]
		}
	case "ffn_down":
		if layer < len(e.weights.FfnDown) {
			target = e.weights.FfnDown[layer]
		}
	}
	if target == nil || len(target) != len(delta) {
		return
	}
	for i := range target {
		target[i] += delta[i]
	}
}
