//go:build linux && cuda

package engine

import (
	"context"
	"log"
	"sync"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/engine"
	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
)

type WrappedCUDAEngine struct {
	engine   *engine.CUDAEngine
	tok      *tokenizer.Tokenizer
	model    string
	loadedAt time.Time
}

type CUDAEngineAdapter struct {
	mu       sync.RWMutex
	engines  map[string]*WrappedCUDAEngine
	requests chan *InferenceRequest
	done     chan struct{}
}

var (
	cudaInstance *CUDAEngineAdapter
	cudaOnce     sync.Once
)

func getAdapterImpl() EngineAdapter {
	cudaOnce.Do(func() {
		cudaInstance = &CUDAEngineAdapter{
			engines:  make(map[string]*WrappedCUDAEngine),
			requests: make(chan *InferenceRequest, 100),
			done:     make(chan struct{}),
		}
		go cudaInstance.processRequests()
	})
	return cudaInstance
}

func (a *CUDAEngineAdapter) processRequests() {
	for {
		select {
		case req := <-a.requests:
			a.handleRequest(req)
		case <-a.done:
			return
		}
	}
}

func (a *CUDAEngineAdapter) handleRequest(req *InferenceRequest) {
	e, err := a.GetEngine(req.Model)
	if err != nil {
		log.Printf("Failed to get engine for model %s: %v", req.Model, err)
		a.sendError(req.ResponseChan, "ENGINE_ERROR", err.Error())
		return
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
				responseChan <- InferenceResponse{
					Complete: true,
				}
				return
			}

			if len(result) == 0 {
				responseChan <- InferenceResponse{
					Complete: true,
				}
				return
			}

			tokens = append(tokens, result[0])

			decodedToken := e.tok.Decode([]int{result[0]})

			select {
			case responseChan <- InferenceResponse{
				Token:    decodedToken,
				TokenID:  i,
				Complete: i == req.MaxTokens-1,
			}:
			default:
				log.Printf("Response channel full, stopping generation")
				return
			}

			if i == req.MaxTokens-1 {
				responseChan <- InferenceResponse{
					Complete: true,
				}
			}
		}
	}()
}

func (a *CUDAEngineAdapter) sendError(responseChanChan chan chan InferenceResponse, code, message string) {
	select {
	case responseChan := <-responseChanChan:
		responseChan <- InferenceResponse{
			Complete: true,
		}
		close(responseChan)
	default:
	}
}

func (a *CUDAEngineAdapter) GetEngine(modelPath string) (*WrappedCUDAEngine, error) {
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

	log.Printf("Loading CUDA engine for model: %s", modelPath)

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

	wrapped := &WrappedCUDAEngine{
		engine:   e,
		tok:      tok,
		model:    modelPath,
		loadedAt: time.Now(),
	}

	a.engines[modelPath] = wrapped
	return wrapped, nil
}

func (a *CUDAEngineAdapter) ListModels() []ModelInfo {
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

func (a *CUDAEngineAdapter) Close() {
	close(a.done)
	a.mu.Lock()
	defer a.mu.Unlock()
	for _, e := range a.engines {
		e.engine.Close()
	}
}

func (a *CUDAEngineAdapter) Infer(ctx context.Context, req *InferenceRequest) (<-chan chan InferenceResponse, error) {
	responseChan := make(chan chan InferenceResponse, 1)

	select {
	case a.requests <- req:
		return responseChan, nil
	default:
		return nil, nil
	}
}

func (a *CUDAEngineAdapter) UnloadModel(modelPath string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if e, ok := a.engines[modelPath]; ok {
		e.engine.Close()
		delete(a.engines, modelPath)
	}
}
