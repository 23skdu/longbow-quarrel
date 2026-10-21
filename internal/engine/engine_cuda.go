//go:build linux && amd64 && cuda && cgo

package engine

import (
	"fmt"
	"log"
	"runtime"
	"runtime/debug"
	"sync"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/mathx"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
	"github.com/23skdu/longbow-quarrel/internal/metrics"
	"github.com/23skdu/longbow-quarrel/internal/simd"
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

	cudaMemoryUsage = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "quarrel_cuda_memory_bytes",
		Help: "Current CUDA memory usage",
	}, []string{"model"})
)

type CUDAStorage = device.CUDAModel
type CUDAScratch = device.LayerScratch

type cudaEngine struct {
	mu           sync.Mutex
	model        *gguf.GGUFFile
	ctx          *device.Context
	cuda         *CUDAStorage
	cpuWeights   *CPUWeights
	numGPULayers int
	config       config.Config
	scratch      *CUDAScratch
	tok          *tokenizer.Tokenizer
	cache        *PagedKVCache
	PromptCache  *PromptCache
	seqKVCaches  map[string]*CPUKVCache
	kvMu         sync.Mutex
	BatchManager *ContinuousBatchManager
	stopChan     chan struct{}
	doneChan     chan struct{}
	lora         *LoRAManager
	governor     *device.MemoryGovernor
	seqTTFTStart map[uint64]time.Time
	ttftMu       sync.Mutex

	// stopTokens is the configured stop-token set, built once at construction.
	stopTokens map[int]struct{}

	// workspaces holds the persistent decode buffers, one per shape. A
	// captured CUDA graph holds the address of every buffer it touches,
	// so the tensors of a decode pass have to be allocated once and
	// reused: a forward that allocates them per call would replay into
	// whatever the pool handed out last time.
	workspaceMu sync.Mutex
	workspaces  map[string]*device.DecodeWorkspace
}

func NewcudaEngine(modelPath string, cfg config.Config) (Engine, error) {
	f, err := gguf.LoadFile(modelPath)
	if err != nil {
		cudaEngineFailed.WithLabelValues("unknown", "gguf_load_failed").Inc()
		return nil, fmt.Errorf("failed to load GGUF: %w", err)
	}
	if err := validateModelFile(f); err != nil {
		_ = f.Close()
		cudaEngineFailed.WithLabelValues("unknown", "invalid_model").Inc()
		return nil, err
	}

	arch := "unknown"
	if v, ok := f.KV["general.architecture"].(string); ok {
		arch = v
	}

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

	// The mixture-of-experts and the multi-head latent attention
	// configurations come from the model metadata. They were not
	// copied: a MoE model was loaded on the GPU as if it were a dense
	// one, and the router and the experts were never used.
	if modelCfg.IsMOE {
		cfg.IsMOE = true
		if cfg.ExpertCount <= 0 {
			cfg.ExpertCount = modelCfg.ExpertCount
		}
		if cfg.ExpertUsedCount <= 0 {
			cfg.ExpertUsedCount = modelCfg.ExpertUsedCount
		}
		if cfg.ExpertSharedCount <= 0 {
			cfg.ExpertSharedCount = modelCfg.ExpertSharedCount
		}
		if cfg.ExpertFeedForwardLength <= 0 {
			cfg.ExpertFeedForwardLength = modelCfg.ExpertFeedForwardLength
		}
		if cfg.ExpertSharedFeedForwardLength <= 0 {
			cfg.ExpertSharedFeedForwardLength = modelCfg.ExpertSharedFeedForwardLength
		}
	}
	if modelCfg.IsMLA {
		cfg.IsMLA = true
		if cfg.QLoRARank <= 0 {
			cfg.QLoRARank = modelCfg.QLoRARank
		}
		if cfg.KVLoRARank <= 0 {
			cfg.KVLoRARank = modelCfg.KVLoRARank
		}
		if cfg.QKNDim <= 0 {
			cfg.QKNDim = modelCfg.QKNDim
		}
		if cfg.QKRopeDim <= 0 {
			cfg.QKRopeDim = modelCfg.QKRopeDim
		}
		if cfg.VHeadDim <= 0 {
			cfg.VHeadDim = modelCfg.VHeadDim
		}
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
	if cfg.KVCacheSize > 0 {
		cfg.SeqLen = cfg.KVCacheSize
	} else if modelCfg.KVCacheSize > 0 {
		cfg.KVCacheSize = modelCfg.KVCacheSize
	}
	if modelCfg.EOSTokenID > 0 && cfg.EOSTokenID <= 0 {
		cfg.EOSTokenID = modelCfg.EOSTokenID
	}

	// The -devices flag selects the CUDA device the engine runs on. It
	// used to be ignored: the engine always created its context on the
	// current device.
	var ctx *device.Context
	if len(cfg.Devices) > 0 {
		ctx, err = device.NewContextOnDevice(cfg.Devices[0])
		if err != nil {
			return nil, fmt.Errorf("failed to select CUDA device %d: %w", cfg.Devices[0], err)
		}
	} else {
		ctx = device.NewContext()
	}

	numGPULayers := cfg.Layers
	if cfg.NumGPULayers >= 0 && cfg.NumGPULayers < cfg.Layers {
		numGPULayers = cfg.NumGPULayers
	}

	cudaModel, err := ctx.NewCUDAModel(f, true, cfg.KVCacheSize, numGPULayers)
	if err != nil {
		ctx.Free()
		_ = f.Close()
		cudaEngineFailed.WithLabelValues(arch, "model_load_failed").Inc()
		return nil, fmt.Errorf("failed to load model to GPU: %w", err)
	}

	// Always load CPU weights for hybrid offloading, recurrent SSM layers, or PLE fallback
	var cpuW *CPUWeights
	cpuW, _ = loadCPUWeights(f, cfg)

	metrics.RecordLayerOffload(arch, numGPULayers, cfg.Layers-numGPULayers)

	cache := &PagedKVCache{}
	if err := cache.Init(ctx, cfg); err != nil {
		ctx.Free()
		cudaModel.Free()
		_ = f.Close()
		cudaEngineFailed.WithLabelValues(arch, "cache_init_failed").Inc()
		return nil, fmt.Errorf("failed to initialize paged cache: %w", err)
	}

	// The memory governor compacts the physical blocks and offloads the
	// least recently used sequence when the device is under pressure. The
	// registration used to happen in the tests only, so the trigger the
	// decode loop fires did nothing.
	//
	// The governor measures the device memory against the limit in the
	// memory configuration, which defaults to eight gigabytes whatever the
	// card holds. On a GPU that is the wrong denominator: a 24 GB card
	// never reaches the threshold and a 6 GB card reaches it too early.
	if total, err := device.GetDeviceMemory(0); err == nil && total > 0 {
		device.SetMaxMemory(total)
	}
	cache.RegisterMemoryGovernor(device.GetMemoryGovernor())

	cudaEngineInitialized.WithLabelValues(arch, arch).Inc()
	cudaMemoryUsage.WithLabelValues(arch).Set(float64(device.CUDAAllocatedBytes()))

	runtime.GC()
	debug.FreeOSMemory()

	tok, err := tokenizer.NewFromGGUF(f)
	if err != nil {
		tok, err = tokenizer.New(modelPath)
		if err != nil {
			log.Printf("Warning: failed to load tokenizer: %v", err)
		}
	}
	// Register the EOS token declared by the model metadata so the decode
	// loop only has to consult the tokenizer.
	if tok != nil && cfg.EOSTokenID > 0 {
		tok.AddEOSTokenID(cfg.EOSTokenID)
	}

	heads := cfg.Heads
	kvHeads := cfg.KVHeads

	isGemma4 := arch == "gemma4"
	e := &cudaEngine{
		model:        f,
		ctx:          ctx,
		cuda:         cudaModel,
		cpuWeights:   cpuW,
		numGPULayers: numGPULayers,
		config:       cfg,
		tok:          tok,
		cache:        cache,
		PromptCache:  NewPromptCache(),
		seqKVCaches:  make(map[string]*CPUKVCache),
		BatchManager: NewContinuousBatchManager(),
		stopChan:     make(chan struct{}),
		doneChan:     make(chan struct{}),
		governor:     device.GetMemoryGovernor(),
		seqTTFTStart: make(map[uint64]time.Time),
		stopTokens:   stopTokenSet(cfg.StopTokens),
	}

	if isGemma4 {
		e.config.Gemma4SlidingWindowSize = 512
		e.config.Gemma4SlidingRoPETheta = 10000.0
		e.config.Gemma4FullRoPETheta = 1000000.0
		e.config.Gemma4PartialRoPEFactor = 0.25
		e.config.Gemma4SlidingHeadDim = 256
		e.config.Gemma4FullHeadDim = 512
		e.config.Gemma4SharedKVLayers = modelCfg.Gemma4SharedKVLayers
		e.config.Gemma4SlidingPattern = modelCfg.Gemma4SlidingPattern
		metrics.RecordGemma4HeadDims(e.config.Gemma4SlidingHeadDim, e.config.Gemma4FullHeadDim)
		metrics.RecordGemma4LayerPattern(e.config.Gemma4SharedKVLayers, e.config.Layers-e.config.Gemma4SharedKVLayers)
		metrics.RecordGemma4Context(e.config.SeqLen)
	}

	log.Printf("CUDA engine initialized with PagedKVCache: model=%s heads=%d kv_heads=%d", modelPath, heads, kvHeads)

	go e.runBatchLoop()

	return e, nil
}

func (e *cudaEngine) Config() config.Config {
	return e.config
}

func (e *cudaEngine) Close() {
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
	e.workspaceMu.Lock()
	for _, workspace := range e.workspaces {
		workspace.Free()
	}
	e.workspaces = nil
	e.workspaceMu.Unlock()
	if e.cuda != nil {
		e.cuda.Free()
	}
	if e.cpuWeights != nil {
		e.cpuWeights.Free()
	}
	if e.ctx != nil {
		e.ctx.Free()
	}
	if e.model != nil {
		_ = e.model.Close()
	}
}

func (e *cudaEngine) LoadAdapter(path, id string) error {
	if e.lora == nil {
		e.lora = NewLoRAManager()
	}
	return e.lora.LoadAdapter(e.ctx, path, id)
}

// ListAdapters returns metadata for the LoRA adapters loaded into the CUDA
// backend, sorted by ID.
func (e *cudaEngine) ListAdapters() []AdapterInfo {
	if e.lora == nil {
		return []AdapterInfo{}
	}
	return e.lora.ListAdapters()
}

func (e *cudaEngine) SwapModel(newModelPath string, newConfig config.Config) error {
	if e.stopChan != nil {
		close(e.stopChan)
		<-e.doneChan
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	if e.BatchManager != nil {
		e.BatchManager.AbortAll(ErrModelSwapped)
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
		_ = e.model.Close()
		e.model = nil
	}

	f, err := gguf.LoadFile(newModelPath)
	if err != nil {
		return fmt.Errorf("failed to load new GGUF: %w", err)
	}
	if err := validateModelFile(f); err != nil {
		_ = f.Close()
		return fmt.Errorf("failed to load new GGUF: %w", err)
	}
	e.model = f

	e.ctx = device.NewContext()

	e.config = newConfig

	stopChan := make(chan struct{})
	doneChan := make(chan struct{})
	e.stopChan = stopChan
	e.doneChan = doneChan
	go e.runBatchLoop()

	return nil
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

		// Memory governor: check pressure and trigger if needed
		if e.governor != nil {
			if action := e.governor.TriggerGovernor(0.85); action != "none" {
				metrics.RecordMemoryPressureEvent("gpu", action, 0.0)
			}
		}

		// 1. Pull active sequences
		maxBatch := 16
		if e.config.MaxBatchSize > 0 {
			maxBatch = e.config.MaxBatchSize
		}
		desc, _ := e.BatchManager.Step(maxBatch, e.cache, e.PromptCache)
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
			// Update Sequence State
			chunkLen := 1
			if i < len(desc.Offsets)-1 {
				chunkLen = desc.Offsets[i+1] - desc.Offsets[i]
			} else {
				chunkLen = len(desc.Tokens) - desc.Offsets[i]
			}

			// If this is an intermediate prefill chunk, just advance position and continue
			if i < len(desc.IsDecode) && !desc.IsDecode[i] && seq.Pos+chunkLen < seq.PromptLen {
				results[i].Free()
				seq.Pos += chunkLen
				continue
			}

			var token int
			if seq.LogitsCallback == nil && seq.Config.Grammar == nil && e.config.FinalLogitSoftcapping == 0 {
				prevTokens := make([]int32, len(seq.Tokens))
				for idx, t := range seq.Tokens {
					prevTokens[idx] = int32(t)
				}
				token = e.ctx.SampleTopKTopP(
					results[i],
					prevTokens,
					float32(seq.Config.RepPenalty),
					float32(seq.Config.FrequencyPenalty),
					float32(seq.Config.PresencePenalty),
					float32(seq.Config.Temperature),
					seq.Config.TopK,
					float32(seq.Config.TopP),
					uint64(time.Now().UnixNano()),
					e.config.VocabSize,
				)
				results[i].Free()
			} else {
				logits := results[i].ToHostF32()
				results[i].Free()

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

				if seq.LogitsCallback != nil {
					seq.LogitsCallback(logits)
				}
				sampler := NewSampler(seq.Config)
				token = sampler.Sample(logits, seq.Tokens)
			}

			// Record TTFT on first generated token
			if !seq.PrefillCompleted || len(seq.Tokens) == seq.PromptLen {
				e.ttftMu.Lock()
				seqTTFT, ok := e.seqTTFTStart[seq.ID]
				if ok {
					delete(e.seqTTFTStart, seq.ID)
				}
				e.ttftMu.Unlock()
				if ok {
					metrics.RecordTTFTLatency(time.Since(seqTTFT))
				}
			} else {
				metrics.RecordInterTokenLatency(time.Duration(0))
			}

			seq.Tokens = append(seq.Tokens, token)
			seq.Pos += chunkLen

			if seq.TokenCallback != nil {
				seq.TokenCallback(token)
			}

			// Termination
			// The tokenizer knows every EOS id the model declares.
			isEOS := e.tok != nil && e.tok.IsEOS(token)
			if isEOS || isStopToken(e.stopTokens, token) || isStopToken(stopTokenSet(seq.Config.StopTokens), token) || len(seq.Tokens) >= seq.MaxTokens {
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

// decodeWorkspace returns the persistent buffers for a shape, building them
// the first time the shape is seen.
// maxDecodeWorkspaces is the number of shapes the engine keeps the
// persistent decode buffers for.
const maxDecodeWorkspaces = 8

func (e *cudaEngine) decodeWorkspace(shape device.DecodeWorkspaceShape) (*device.DecodeWorkspace, error) {
	key := fmt.Sprintf("%d/%d/%d/%d/%d/%d",
		shape.NumTokens, shape.Dim, shape.HiddenDim, shape.BatchSize, shape.MaxBlocks, shape.VocabSize)

	e.workspaceMu.Lock()
	defer e.workspaceMu.Unlock()

	if e.workspaces == nil {
		e.workspaces = make(map[string]*device.DecodeWorkspace)
	}
	if workspace, ok := e.workspaces[key]; ok {
		return workspace, nil
	}

	workspace, err := device.NewDecodeWorkspace(e.ctx, shape)
	if err != nil {
		return nil, err
	}
	e.workspaces[key] = workspace

	// The cache is bounded: a server whose batches change shape all the
	// time would otherwise keep a workspace per shape forever.
	if len(e.workspaces) > maxDecodeWorkspaces {
		for oldKey, oldWorkspace := range e.workspaces {
			if oldKey == key {
				continue
			}
			oldWorkspace.Free()
			delete(e.workspaces, oldKey)
			break
		}
	}

	return workspace, nil
}

func (e *cudaEngine) ForwardBatch(desc *BatchDescriptor) ([]*device.Tensor, error) {
	ctx := e.ctx
	batchSize := len(desc.Sequences)
	numTokens := len(desc.Tokens)

	// 1. Prepare Metadata Tensors
	//
	// The tensors are the persistent ones of the workspace: a captured
	// graph holds their addresses, so they cannot be allocated per call.
	workspace, err := e.decodeWorkspace(device.DecodeWorkspaceShape{
		NumTokens: numTokens,
		Dim:       e.config.Dim,
		HiddenDim: e.config.HiddenDim,
		Heads:     e.config.Heads,
		KVHeads:   e.config.KVHeads,
		HeadDim:   e.config.HeadDim,
		BatchSize: batchSize,
		MaxBlocks: 1,
		VocabSize: e.config.VocabSize,
	})
	if err != nil {
		return nil, err
	}

	tokenPosTensor := workspace.TokenPos()
	tokenPositions := make([]int32, numTokens)
	for i, start := range desc.Offsets {
		end := numTokens
		if i < batchSize-1 {
			end = desc.Offsets[i+1]
		}
		for j := 0; j < end-start; j++ {
			// The index is the token's, and the position is the
			// sequence's context length plus the token's offset within
			// it. The index used to be the token's offset added to the
			// sequence's start, which put the positions of the second
			// sequence past the end of the tensor.
			tokenPositions[start+j] = int32(desc.ContextLens[i] + j)
		}
	}
	if err := tokenPosTensor.LoadFrom(tokenPositions); err != nil {
		return nil, err
	}

	tokenToSeqTensor := workspace.TokenToSeq()
	tokenToSeq := make([]int32, numTokens)
	for i, val := range desc.TokenToSeq {
		tokenToSeq[i] = int32(val)
	}
	if err := tokenToSeqTensor.LoadFrom(tokenToSeq); err != nil {
		return nil, err
	}

	// Ensure KV blocks are allocated for all active sequences
	for _, seq := range desc.Sequences {
		seqID := fmt.Sprintf("seq-%d", seq.ID)
		tokensNeeded := seq.Pos + 1
		if len(seq.Tokens) > tokensNeeded {
			tokensNeeded = len(seq.Tokens)
		}
		if err := e.cache.Allocate(seqID, tokensNeeded); err != nil {
			return nil, fmt.Errorf("failed to allocate KV cache blocks for %s: %w", seqID, err)
		}
	}

	// Pack block tables.
	//
	// The width has to cover both the blocks the sequences were asked
	// for and the blocks they were actually given, and it has to be at
	// least one: a sequence without a maximum token count used to leave
	// it at zero, and the packing below then wrote past the end of the
	// buffer and took the engine down.
	maxBlocks := 1
	for _, seq := range desc.Sequences {
		if nt := (seq.MaxTokens + e.cache.blockSize - 1) / e.cache.blockSize; nt > maxBlocks {
			maxBlocks = nt
		}
		seqID := fmt.Sprintf("seq-%d", seq.ID)
		if nt := len(e.cache.BlockTable(seqID)); nt > maxBlocks {
			maxBlocks = nt
		}
	}

	workspace, err = e.decodeWorkspace(device.DecodeWorkspaceShape{
		NumTokens: numTokens,
		Dim:       e.config.Dim,
		HiddenDim: e.config.HiddenDim,
		Heads:     e.config.Heads,
		KVHeads:   e.config.KVHeads,
		HeadDim:   e.config.HeadDim,
		BatchSize: batchSize,
		MaxBlocks: maxBlocks,
		VocabSize: e.config.VocabSize,
	})
	if err != nil {
		return nil, err
	}

	blockTableTensor := workspace.BlockTable()
	btData := make([]int32, batchSize*maxBlocks)
	for i, seq := range desc.Sequences {
		seqID := fmt.Sprintf("seq-%d", seq.ID)
		table := e.cache.BlockTable(seqID)
		for j, bidx := range table {
			if j >= maxBlocks {
				break
			}
			btData[i*maxBlocks+j] = bidx
		}
	}
	if err := blockTableTensor.LoadFrom(btData); err != nil {
		return nil, err
	}

	dim := e.config.Dim
	eps := e.config.Eps

	// 2. Initial Embedding
	// [numTokens, dim]
	hidden, err := e.cuda.GetBatchEmbedding(desc.Tokens, e.config.VocabSize)
	if err != nil {
		return nil, err
	}
	if e.config.IsGemma4 {
		hiddenHost := hidden.ToHostF32()
		scaleEmb := float32(mathx.Sqrt(float64(dim)))
		for j := range hiddenHost {
			hiddenHost[j] *= scaleEmb
		}
		_ = hidden.LoadFrom(hiddenHost)
	}

	var actLog *ActivationLogger
	if e.config.ActivationLogPath != "" {
		actLog = NewActivationLogger()
		actLog.Enable("", desc.Tokens)
		hHost := hidden.ToHostF32()
		if len(hHost) >= dim {
			actLog.LogEmbedding(hHost[:dim])
		}
	}

	// 3. Layer Loop
	heads := e.config.Heads
	kvHeads := e.config.KVHeads
	headDim := e.config.HeadDim
	ropeTheta := e.config.RopeTheta
	// Build integer position IDs for RoPE
	posIds := make([]int, numTokens)
	for i, start := range desc.Offsets {
		end := numTokens
		if i < batchSize-1 {
			end = desc.Offsets[i+1]
		}
		for j := 0; j < end-start; j++ {
			posIds[start+j] = desc.ContextLens[i] + j
		}
	}
	maxPos := 0
	for _, p := range posIds {
		if p > maxPos {
			maxPos = p
		}
	}

	gpuLayers := e.numGPULayers
	for layer := 0; layer < e.config.Layers; layer++ {
		qW, _ := e.cuda.GetWeightTensor(fmt.Sprintf("blk.%d.attn_q.weight", layer))
		needsCPU := (layer >= gpuLayers) || (e.cpuWeights != nil && (qW == nil ||
			(layer < len(e.cpuWeights.SSMA) && len(e.cpuWeights.SSMA[layer]) > 0)))

		if needsCPU && e.cpuWeights != nil {
			// Find contiguous run of CPU layers to avoid ping-pong GPU transfers
			endLayer := layer + 1
			for endLayer < e.config.Layers {
				eqW, _ := e.cuda.GetWeightTensor(fmt.Sprintf("blk.%d.attn_q.weight", endLayer))
				eNeedsCPU := (endLayer >= gpuLayers) || (eqW == nil ||
					(endLayer < len(e.cpuWeights.SSMA) && len(e.cpuWeights.SSMA[endLayer]) > 0))
				if !eNeedsCPU {
					break
				}
				endLayer++
			}

			startCPU := time.Now()
			transferStart := time.Now()
			hiddenHost := hidden.ToHostF32()
			transferDuration := time.Since(transferStart)

			// Pre-allocate Gemma4 layer buffers or standard CPU layer scratch for this CPU offload batch
			var gemma4Buf *Gemma4LayerBuf
			var layerScratch *CPULayerScratch
			if e.config.IsGemma4 {
				gemma4Buf = NewGemma4LayerBuf(dim, e.config.Gemma4FullHeadDim, heads, e.config.HiddenDim, e.config.KVCacheSize)
			} else {
				layerScratch = NewCPULayerScratch(dim, heads, headDim)
			}

			for sIdx, seq := range desc.Sequences {
				seqIDStr := fmt.Sprintf("seq-%d", seq.ID)
				e.kvMu.Lock()
				if e.seqKVCaches == nil {
					e.seqKVCaches = make(map[string]*CPUKVCache)
				}
				kvCache, ok := e.seqKVCaches[seqIDStr]
				if !ok {
					if e.config.WindowSize > 0 {
						kvCache = NewCPUKVCacheWithWindow(e.config.Layers, e.config.WindowSize)
					} else {
						kvCache = NewCPUKVCache(e.config.Layers)
					}
					e.seqKVCaches[seqIDStr] = kvCache
				}
				e.kvMu.Unlock()

				start := desc.Offsets[sIdx]
				end := numTokens
				if sIdx < batchSize-1 {
					end = desc.Offsets[sIdx+1]
				}
				basePos := desc.ContextLens[sIdx]
				for t := 0; t < end-start; t++ {
					tokIdx := start + t
					tokOffset := tokIdx * dim
					tokenHidden := hiddenHost[tokOffset : tokOffset+dim]
					pos := basePos + t

					if e.config.IsGemma4 {
						tok := desc.Tokens[tokIdx]
						ple := e.cpuWeights.ComputeGemma4PLE(tok, tokenHidden, e.config.Layers)
						for l := layer; l < endLayer; l++ {
							tokenHidden = ApplyGemma4LayerCPU(e.cpuWeights, tokenHidden, l, pos, kvCache, ple[l], e.config, gemma4Buf)
						}
					} else {
						for l := layer; l < endLayer; l++ {
							tokenHidden = ApplyLayerCPUKVWithScratch(e.cpuWeights, tokenHidden, l, pos, kvCache, e.config, layerScratch)
						}
					}
					copy(hiddenHost[tokOffset:tokOffset+dim], tokenHidden)
				}
			}
			loadStart := time.Now()
			_ = hidden.LoadFrom(hiddenHost)
			transferDuration += time.Since(loadStart)
			metrics.RecordLayerOffloadTransfer(e.config.Architecture, transferDuration)
			metrics.RecordLayerOffloadCPUDuration(e.config.Architecture, time.Since(startCPU))
			layer = endLayer - 1
			continue
		}

		// Standard Attention and MLP on GPU
		// Batched RMSNorm
		attnNormW, _ := e.cuda.GetWeightTensor(fmt.Sprintf("blk.%d.attn_norm.weight", layer))
		normed := ctx.NewTensorFP32(numTokens, dim)
		ctx.RMSNorm(hidden, attnNormW, normed, numTokens, dim, eps)

		var q, k, v *device.Tensor
		if e.config.IsMLA {
			qkNopeDim := e.config.QKNDim
			if qkNopeDim <= 0 {
				qkNopeDim = headDim
			}
			qkRopeDim := e.config.QKRopeDim
			if qkRopeDim <= 0 {
				qkRopeDim = headDim
			}
			vHeadDim := e.config.VHeadDim
			if vHeadDim <= 0 {
				vHeadDim = headDim
			}
			kvLoraRank := e.config.KVLoRARank
			if kvLoraRank <= 0 {
				kvLoraRank = dim / 2
			}

			qAW, _ := e.cuda.GetWeightTensor(fmt.Sprintf("blk.%d.attn_q_a.weight", layer))
			qBW, _ := e.cuda.GetWeightTensor(fmt.Sprintf("blk.%d.attn_q_b.weight", layer))
			var qAll *device.Tensor
			if qAW != nil && qBW != nil {
				if err := checkMLAQLoRARank(e.config.QLoRARank, qAW.Cols()); err != nil {
					return nil, fmt.Errorf("layer %d: %w", layer, err)
				}
				cQ, _ := ctx.MatmulF16(normed, qAW)
				qANormW, _ := e.cuda.GetWeightTensor(fmt.Sprintf("blk.%d.attn_q_a_norm.weight", layer))
				if qANormW != nil {
					ctx.RMSNorm(cQ, qANormW, cQ, numTokens, cQ.Cols(), eps)
				}
				qAll, _ = ctx.MatmulF16(cQ, qBW)
				cQ.ReturnToPool()
			} else {
				qAll, _ = ctx.MatmulF16(normed, qW)
			}

			var posTensor *device.Tensor
			if len(posIds) > 0 {
				posTensor = ctx.NewTensorI32(1, len(posIds))
				_ = posTensor.LoadFrom(posIds)
			}
			qNope, qRope := ctx.MLAProjectQuerySplitRoPE(qAll, posTensor, numTokens, heads, qkNopeDim, qkRopeDim, ropeTheta)
			qAll.ReturnToPool()
			if posTensor != nil {
				posTensor.ReturnToPool()
			}

			kvAW, _ := e.cuda.GetWeightTensor(fmt.Sprintf("blk.%d.attn_kv_a_mqa.weight", layer))
			kvBW, _ := e.cuda.GetWeightTensor(fmt.Sprintf("blk.%d.attn_kv_b.weight", layer))
			if kvAW != nil && kvBW != nil {
				kvDown, _ := ctx.MatmulF16(normed, kvAW)
				kvANormW, _ := e.cuda.GetWeightTensor(fmt.Sprintf("blk.%d.attn_kv_a_norm.weight", layer))
				if kvANormW != nil {
					ctx.RMSNorm(kvDown, kvANormW, kvDown, numTokens, kvLoraRank, eps)
				}
				kNope, vVal := ctx.MLADecompressKV(kvDown, kvBW, numTokens, kvLoraRank, heads, qkNopeDim, vHeadDim)
				kvDown.ReturnToPool()
				q = qNope
				qRope.ReturnToPool()
				k = kNope
				v = vVal
			} else {
				q = qNope
				qRope.ReturnToPool()
				kW, _ := e.cuda.GetWeightTensor(fmt.Sprintf("blk.%d.attn_k.weight", layer))
				vW, _ := e.cuda.GetWeightTensor(fmt.Sprintf("blk.%d.attn_v.weight", layer))
				k, _ = ctx.MatmulF16(normed, kW)
				v, _ = ctx.MatmulF16(normed, vW)
			}
			normed.ReturnToPool()
		} else {
			// Batched Projections & RoPE
			kW, _ := e.cuda.GetWeightTensor(fmt.Sprintf("blk.%d.attn_k.weight", layer))
			vW, _ := e.cuda.GetWeightTensor(fmt.Sprintf("blk.%d.attn_v.weight", layer))
			qNormW, _ := e.cuda.GetWeightTensor(fmt.Sprintf("blk.%d.attn_q_norm.weight", layer))
			kNormW, _ := e.cuda.GetWeightTensor(fmt.Sprintf("blk.%d.attn_k_norm.weight", layer))

			rotaryDim := headDim
			if e.config.IsGemma4 && e.config.Gemma4PartialRoPEFactor > 0 {
				rotaryDim = int(float32(headDim) * e.config.Gemma4PartialRoPEFactor)
			}

			var err error
			if qNormW != nil || kNormW != nil {
				q, err = ctx.MatmulF16(normed, qW)
				if err == nil {
					k, err = ctx.MatmulF16(normed, kW)
				}
				if err == nil {
					v, err = ctx.MatmulF16(normed, vW)
				}
				if err != nil {
					normed.ReturnToPool()
					return nil, err
				}
				ctx.QKNormRMSNorm(q, k, qNormW, kNormW, numTokens, heads, kvHeads, headDim, eps)
				ctx.EnsureRoPETables(maxPos+1, headDim, rotaryDim, ropeTheta)
				ctx.FusedRoPEPrecomputed(q, posIds, numTokens, heads, 1, headDim)
				ctx.FusedRoPEPrecomputed(k, posIds, numTokens, kvHeads, 1, headDim)
			} else {
				q, k, v, err = ctx.FusedQKVRope(normed, qW, kW, vW, posIds, numTokens, dim, heads, kvHeads, headDim, rotaryDim, ropeTheta)
			}
			normed.ReturnToPool()
			if err != nil {
				return nil, err
			}
		}

		// Paged Attention (BATCHED)
		kCache := e.cache.kPools[layer]
		vCache := e.cache.vPools[layer]

		attnOut := ctx.NewTensorFP32(numTokens, dim)

		// Update Cache (StoreKV)
		// We'll calculate physical positions for new tokens
		// The store kernel reads the physical positions as float32, the
		// same convention the paged cache and the Metal kernels use.
		physPosData := make([]float32, numTokens)
		for i, seqIdx := range desc.TokenToSeq {
			seq := desc.Sequences[seqIdx]
			seqID := fmt.Sprintf("seq-%d", seq.ID)
			table := e.cache.BlockTable(seqID)
			// The sequence is in use, so the governor must not pick it
			// as the least recently used one.
			e.cache.Touch(seqID)
			logicalPos := desc.ContextLens[seqIdx] + (i - desc.Offsets[seqIdx])
			blockIdx := logicalPos / e.cache.blockSize
			offset := logicalPos % e.cache.blockSize
			if blockIdx < len(table) {
				physPosData[i] = float32(table[blockIdx]*int32(e.cache.blockSize) + int32(offset))
			}
		}
		physPosTensor := workspace.PhysPos()
		if err := physPosTensor.LoadFrom(physPosData); err != nil {
			return nil, err
		}

		// The number of tokens, not the number of sequences: the kernel
		// walks the tokens and their physical positions. The batch size
		// was passed instead, so only the first token of each sequence
		// ever reached the cache and the rest of it stayed uninitialised.
		//
		// The store goes through the cache, which knows the precision of
		// the pool: a quantized pool needs the codes and their scale,
		// not the F16 values.
		e.cache.StoreKVPagedBatch(layer, k, v, physPosTensor)

		// Attention — use quantized path if cache is FP8/Q8_0
		if e.cache != nil && (e.cache.Precision == device.DataTypeFP8 || e.cache.Precision == device.DataTypeQ8_0) {
			kScales, vScales := e.cache.GetScaleTensors(layer)
			isFP8 := e.cache.Precision == device.DataTypeFP8
			scale := float32(1.0 / mathx.Sqrt(float64(headDim)))
			ctx.PagedAttentionQuantized(q, kCache, vCache, kScales, vScales, attnOut, tokenPosTensor, blockTableTensor, tokenToSeqTensor, maxBlocks, heads, kvHeads, headDim, e.cache.blockSize, batchSize, scale, isFP8)
		} else if numTokens <= 4 && maxBlocks >= 16 {
			// Long-context decode: Split-K FlashDecoding across SMs
			scale := float32(1.0 / mathx.Sqrt(float64(headDim)))
			splitK := 16
			if maxBlocks < 32 {
				splitK = 8
			}
			ctx.AttentionPagedSplitK(q, kCache, vCache, attnOut, tokenPosTensor, blockTableTensor, tokenToSeqTensor, maxBlocks, heads, kvHeads, headDim, e.cache.blockSize, numTokens, scale, splitK)
		} else {
			ctx.AttentionPagedBatch(q, kCache, vCache, attnOut, tokenPosTensor, blockTableTensor, maxBlocks, heads, kvHeads, headDim, e.cache.blockSize, tokenToSeqTensor, batchSize)
		}

		q.ReturnToPool()
		k.ReturnToPool()
		v.ReturnToPool()

		// Output Projection
		oW, _ := e.cuda.GetWeightTensor(fmt.Sprintf("blk.%d.attn_output.weight", layer))
		attnProj, mErr := ctx.MatmulF16(attnOut, oW)
		if mErr != nil {
			return nil, fmt.Errorf("layer %d: the attention output projection failed: %w", layer, mErr)
		}
		attnOut.ReturnToPool()

		// Residual Add
		ctx.Add(hidden, attnProj, hidden, numTokens*dim)
		attnProj.ReturnToPool()

		// Feed Forward (Batched)
		ffnNormW, _ := e.cuda.GetWeightTensor(fmt.Sprintf("blk.%d.ffn_norm.weight", layer))
		if ffnNormW != nil {
			normedFFN := ctx.NewTensorFP32(numTokens, dim)
			ctx.RMSNorm(hidden, ffnNormW, normedFFN, numTokens, dim, eps)

			ffnGateW, _ := e.cuda.GetWeightTensor(fmt.Sprintf("blk.%d.ffn_gate.weight", layer))
			ffnUpW, _ := e.cuda.GetWeightTensor(fmt.Sprintf("blk.%d.ffn_up.weight", layer))
			ffnDownW, _ := e.cuda.GetWeightTensor(fmt.Sprintf("blk.%d.ffn_down.weight", layer))

			if ffnGateW != nil && ffnUpW != nil && ffnDownW != nil {
				gate, _ := ctx.MatmulF16(normedFFN, ffnGateW)
				up, _ := ctx.MatmulF16(normedFFN, ffnUpW)
				normedFFN.ReturnToPool()

				ctx.FusedSwiGLU(gate, up, gate, numTokens, e.config.HiddenDim)
				up.ReturnToPool()

				down, _ := ctx.MatmulF16(gate, ffnDownW)
				gate.ReturnToPool()

				ctx.Add(hidden, down, hidden, numTokens*dim)
				down.ReturnToPool()
			} else if ffnRouterW, _ := e.cuda.GetWeightTensor(fmt.Sprintf("blk.%d.ffn_gate_inp.weight", layer)); ffnRouterW != nil {
				// MoE Feed-Forward
				topK := e.config.ExpertUsedCount
				if topK <= 0 {
					topK = 2
				}
				hiddenDimMoE := e.config.ExpertFeedForwardLength
				if hiddenDimMoE <= 0 {
					hiddenDimMoE = e.config.HiddenDim
				}

				logits := ctx.MOERouterLogits(normedFFN, ffnRouterW)
				expertIndices, expertWeights := ctx.MOETopKSelection(logits, topK)
				logits.ReturnToPool()

				ffnGateExps, _ := e.cuda.GetWeightTensor(fmt.Sprintf("blk.%d.ffn_gate_exps.weight", layer))
				ffnUpExps, _ := e.cuda.GetWeightTensor(fmt.Sprintf("blk.%d.ffn_up_exps.weight", layer))
				ffnDownExps, _ := e.cuda.GetWeightTensor(fmt.Sprintf("blk.%d.ffn_down_exps.weight", layer))

				if ffnGateExps != nil && ffnUpExps != nil && ffnDownExps != nil {
					activated := ctx.MOEExpertGateUpSwiGLU(normedFFN, ffnGateExps, ffnUpExps, expertIndices, expertWeights, hiddenDimMoE)
					down := ctx.MOEExpertForward(activated, ffnDownExps, expertIndices, expertWeights, dim)
					activated.ReturnToPool()

					// Check for shared expert
					ffnGateSh, _ := e.cuda.GetWeightTensor(fmt.Sprintf("blk.%d.ffn_gate_shexp.weight", layer))
					ffnUpSh, _ := e.cuda.GetWeightTensor(fmt.Sprintf("blk.%d.ffn_up_shexp.weight", layer))
					ffnDownSh, _ := e.cuda.GetWeightTensor(fmt.Sprintf("blk.%d.ffn_down_shexp.weight", layer))
					if ffnGateSh != nil && ffnUpSh != nil && ffnDownSh != nil {
						shGate, _ := ctx.MatmulF16(normedFFN, ffnGateSh)
						shUp, _ := ctx.MatmulF16(normedFFN, ffnUpSh)
						ctx.FusedSwiGLU(shGate, shUp, shGate, numTokens, shGate.Cols())
						shUp.ReturnToPool()
						shDown, _ := ctx.MatmulF16(shGate, ffnDownSh)
						shGate.ReturnToPool()
						ctx.Add(down, shDown, down, numTokens*dim)
						shDown.ReturnToPool()
					}

					ctx.Add(hidden, down, hidden, numTokens*dim)
					down.ReturnToPool()
				}
				expertIndices.ReturnToPool()
				expertWeights.ReturnToPool()
				normedFFN.ReturnToPool()
			} else {
				normedFFN.ReturnToPool()
			}
		}
		if actLog != nil {
			collector := &LayerActivationCollector{LayerIdx: layer}
			logLayerActivations(actLog, collector)
		}
	}

	// 4. Extract Last Logits
	results := make([]*device.Tensor, batchSize)
	hiddenHost := hidden.ToHostF32()
	for i := 0; i < batchSize; i++ {
		lastTokIdx := numTokens - 1
		if i < batchSize-1 {
			lastTokIdx = desc.Offsets[i+1] - 1
		}
		tokOffset := lastTokIdx * dim
		lastTokenHidden := make([]float32, dim)
		copy(lastTokenHidden, hiddenHost[tokOffset:tokOffset+dim])

		var finalLogits []float32
		if e.cpuWeights != nil {
			if e.cpuWeights.OutputNorm != nil {
				normed := make([]float32, dim)
				simd.RMSNorm(lastTokenHidden, e.cpuWeights.OutputNorm, normed, 1, dim, eps)
				lastTokenHidden = normed
			}
			if len(e.cpuWeights.Output) > 0 || e.cpuWeights.RawOutput != nil {
				finalLogits = e.cpuWeights.MatVec(e.cpuWeights.Output, e.cpuWeights.RawOutput, lastTokenHidden)
			} else {
				finalLogits = lastTokenHidden
			}
		}

		if actLog != nil && len(finalLogits) > 0 {
			actLog.LogLogits(finalLogits, topTokenIDs(finalLogits, 5))
			if err := actLog.SaveToFile(e.config.ActivationLogPath); err != nil {
				log.Printf("Warning: failed to save activation log: %v", err)
			}
		}

		res := ctx.NewTensorFP32(1, e.config.VocabSize)
		if err := res.LoadFrom(finalLogits); err != nil {
			return nil, err
		}
		results[i] = res
	}

	return results, nil
}

func (e *cudaEngine) inferInternal(inputTokens []int, tokensToGenerate int, samplerConfig SamplerConfig, tokenCallback func(int), logitsCallback func([]float32)) ([]int, error) {
	metrics.RecordContextLength(len(inputTokens))
	startTime := time.Now()

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

	e.ttftMu.Lock()
	e.seqTTFTStart[req.ID] = time.Now()
	e.ttftMu.Unlock()

	e.BatchManager.Submit(req)

	select {
	case tokens := <-resChan:
		genTokens := 0
		if len(tokens) > len(inputTokens) {
			genTokens = len(tokens) - len(inputTokens)
			metrics.RecordInference(genTokens, time.Since(startTime))
			return tokens[len(inputTokens):], nil
		}
		metrics.RecordInference(0, time.Since(startTime))
		return []int{}, nil
	case err := <-errChan:
		return nil, err
	}
}

func (e *cudaEngine) ForwardDraft(tokens []int) ([][]float32, error) {
	if e.model == nil || e.cuda == nil {
		return nil, fmt.Errorf("model not initialized")
	}
	if len(tokens) == 0 {
		return nil, fmt.Errorf("no tokens to forward")
	}

	// The batched forward, not the single-token one.
	//
	// The single-token forward used the model's legacy KV cache, which
	// the path never wrote, and the fused attention's cache stride did
	// not match that cache's layout. It produced NaNs, and the
	// speculative verification compares the target's distribution
	// against them. The batched forward uses the paged cache, which
	// works.
	seqID := 1
	tokenToSeq := make([]int, len(tokens))
	batch := &BatchDescriptor{
		Sequences:   []*Sequence{{ID: uint64(seqID), Tokens: tokens, PromptLen: len(tokens)}},
		Tokens:      tokens,
		Offsets:     []int{0},
		ContextLens: []int{0},
		TokenToSeq:  tokenToSeq,
	}

	results, err := e.ForwardBatch(batch)
	if err != nil {
		return nil, err
	}

	// The draft's blocks are not kept: the next verification starts from
	// the prompt again.
	defer e.cache.FreeSequence(fmt.Sprintf("seq-%d", seqID))

	logits := make([][]float32, 1)
	if len(results) > 0 && results[0] != nil {
		logits[0] = results[0].ToHostF32()
		results[0].Free()
	}
	if len(logits[0]) == 0 {
		return nil, fmt.Errorf("forward pass failed")
	}

	return logits, nil
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
