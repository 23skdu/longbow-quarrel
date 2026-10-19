//go:build !(darwin && metal) && !(linux && amd64 && cuda && cgo)

package engine

import (
	"context"
	"log"
	"sync"
	"sync/atomic"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/engine"
	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
)

// WrappedCPUEngine couples an inference engine with the tokenizer for one
// loaded model.
type WrappedCPUEngine struct {
	engine     engine.Engine
	tok        *tokenizer.Tokenizer
	model      string
	loadedAt   time.Time
	activeReqs int32
	shutdown   chan struct{}
}

// CPUEngineAdapter runs inference on the CPU backend.
//
// It used to be a stub that answered every request with an empty, successful
// response: the WebUI returned HTTP 200 with an empty completion and no
// error, which is the worst possible failure mode for a client. The CPU
// engine has always worked, so the adapter now loads a real model and runs
// real inference.
type CPUEngineAdapter struct {
	mu           sync.RWMutex
	engines      map[string]*WrappedCPUEngine
	requests     chan *InferenceRequest
	done         chan struct{}
	currentModel string
}

var (
	cpuInstance *CPUEngineAdapter
	cpuOnce     sync.Once
)

// getAdapterImpl returns the CPU adapter singleton.
func getAdapterImpl() EngineAdapter {
	cpuOnce.Do(func() {
		cpuInstance = &CPUEngineAdapter{
			engines:  make(map[string]*WrappedCPUEngine),
			requests: make(chan *InferenceRequest, 100),
			done:     make(chan struct{}),
		}
		log.Println("Using the CPU inference backend.")
		go cpuInstance.processRequests()
	})
	return cpuInstance
}

func (a *CPUEngineAdapter) processRequests() {
	for {
		select {
		case req := <-a.requests:
			a.handleRequest(req)
		case <-a.done:
			return
		}
	}
}

func (a *CPUEngineAdapter) handleRequest(req *InferenceRequest) {
	e, err := a.GetEngine(req.Model)
	if err != nil {
		log.Printf("Failed to get engine for model %s: %v", req.Model, err)
		a.sendError(req.ResponseChan, "ENGINE_ERROR", err.Error())
		return
	}

	atomic.AddInt32(&e.activeReqs, 1)
	defer atomic.AddInt32(&e.activeReqs, -1)

	select {
	case <-e.shutdown:
		a.sendError(req.ResponseChan, "MODEL_SHUTDOWN", "Model is being replaced")
		return
	default:
	}

	tokens := e.tok.Encode(req.Prompt)
	if len(tokens) == 0 {
		tokens = []int{1}
	}

	cfg := engine.SamplerConfig{
		Temperature: req.Temperature,
		TopK:        req.TopK,
		TopP:        req.TopP,
	}

	responseChan := make(chan InferenceResponse, req.MaxTokens)

	select {
	case req.ResponseChan <- responseChan:
	default:
		log.Printf("Request queue full, dropping request")
		a.sendError(req.ResponseChan, "QUEUE_FULL", "Request queue is full")
		return
	}

	go func() {
		defer close(responseChan)

		for i := 0; i < req.MaxTokens; i++ {
			result, err := e.engine.Infer(tokens, 1, cfg)
			if err != nil {
				log.Printf("Inference error: %v", err)
				responseChan <- InferenceResponse{Complete: true}
				return
			}

			if len(result) == 0 {
				responseChan <- InferenceResponse{Complete: true}
				return
			}

			tokens = append(tokens, result[0])

			select {
			case responseChan <- InferenceResponse{
				Token:    e.tok.Decode([]int{result[0]}),
				TokenID:  i,
				Complete: i == req.MaxTokens-1,
			}:
			default:
				log.Printf("Response channel full, stopping generation")
				return
			}

			if i == req.MaxTokens-1 {
				responseChan <- InferenceResponse{Complete: true}
			}
		}
	}()
}

// sendError delivers a completed response to the caller.
//
// It must *deliver* a finished inner channel on the outer channel the
// caller is waiting on, not take one out of it: the caller does
// `inner := <-outer` and then ranges over `inner`. The previous
// implementation received from the outer channel, which meant the caller
// never received anything and blocked forever.
func (a *CPUEngineAdapter) sendError(responseChanChan chan chan InferenceResponse, code, message string) {
	log.Printf("[%s] %s", code, message)
	if responseChanChan == nil {
		return
	}
	inner := make(chan InferenceResponse, 1)
	inner <- InferenceResponse{Complete: true, Error: message}
	close(inner)
	select {
	case responseChanChan <- inner:
	default:
	}
}

// GetEngine returns the engine for the given model, loading it on first use.
func (a *CPUEngineAdapter) GetEngine(modelPath string) (*WrappedCPUEngine, error) {
	a.mu.RLock()
	if e, ok := a.engines[modelPath]; ok {
		a.mu.RUnlock()
		return e, nil
	}
	a.mu.RUnlock()

	a.mu.Lock()
	defer a.mu.Unlock()

	if e, ok := a.engines[modelPath]; ok {
		return e, nil
	}

	log.Printf("Loading CPU engine for model: %s", modelPath)

	cfg := config.Config{
		KVCacheSize: 2048,
	}

	e, err := engine.NewEngine(modelPath, cfg)
	if err != nil {
		return nil, err
	}

	tok, err := tokenizer.New(modelPath)
	if err != nil {
		e.Close()
		return nil, err
	}

	wrapped := &WrappedCPUEngine{
		engine:   e,
		tok:      tok,
		model:    modelPath,
		loadedAt: time.Now(),
		shutdown: make(chan struct{}),
	}

	a.engines[modelPath] = wrapped
	return wrapped, nil
}

func (a *CPUEngineAdapter) ListModels() []ModelInfo {
	a.mu.RLock()
	defer a.mu.RUnlock()

	models := make([]ModelInfo, 0, len(a.engines))
	for path := range a.engines {
		models = append(models, ModelInfo{
			Name:   path,
			Path:   path,
			Loaded: true,
		})
	}
	return models
}

func (a *CPUEngineAdapter) Close() {
	close(a.done)
	a.mu.Lock()
	defer a.mu.Unlock()
	for _, e := range a.engines {
		e.engine.Close()
	}
}

func (a *CPUEngineAdapter) Infer(ctx context.Context, req *InferenceRequest) (<-chan chan InferenceResponse, error) {
	responseChan := make(chan chan InferenceResponse, 1)

	// The adapter delivers the inner (per-token) channel on this outer
	// channel, so the request has to carry it. The caller waits on exactly
	// this channel, which is why the assignment has to happen before the
	// request is queued.
	req.ResponseChan = responseChan

	select {
	case a.requests <- req:
		return responseChan, nil
	default:
		// The queue is full. Return an explicit error rather than a nil
		// channel, which a caller cannot distinguish from success.
		return nil, ErrQueueFull
	}
}

func (a *CPUEngineAdapter) UnloadModel(modelPath string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if e, ok := a.engines[modelPath]; ok {
		close(e.shutdown)
		e.engine.Close()
		delete(a.engines, modelPath)
	}
}

func (a *CPUEngineAdapter) LoadModel(modelPath string) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	if _, ok := a.engines[modelPath]; ok {
		return nil
	}

	log.Printf("Loading CPU model: %s", modelPath)

	cfg := config.Config{
		KVCacheSize: 2048,
	}

	e, err := engine.NewEngine(modelPath, cfg)
	if err != nil {
		return err
	}

	tok, err := tokenizer.New(modelPath)
	if err != nil {
		e.Close()
		return err
	}

	a.engines[modelPath] = &WrappedCPUEngine{
		engine:   e,
		tok:      tok,
		model:    modelPath,
		loadedAt: time.Now(),
		shutdown: make(chan struct{}),
	}
	return nil
}

func (a *CPUEngineAdapter) HotSwapModel(oldModelPath, newModelPath string) error {
	a.UnloadModel(oldModelPath)
	return a.LoadModel(newModelPath)
}
