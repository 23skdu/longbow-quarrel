package engine

import (
	"context"
	"log"
	"sync"
	"time"
)

type InferenceRequest struct {
	Prompt       string
	Temperature  float64
	TopK         int
	TopP         float64
	MaxTokens    int
	Model        string
	Priority     int
	ResponseChan chan chan InferenceResponse
}

type InferenceResponse struct {
	Token    string
	TokenID  int
	Complete bool
}

type ModelInfo struct {
	Name         string
	Path         string
	Parameters   string
	Quantization string
	Loaded       bool
}

type EngineAdapter struct {
	mu       sync.RWMutex
	engines  map[string]*WrappedEngine
	requests chan *InferenceRequest
	done     chan struct{}
}

type WrappedEngine struct {
	model    string
	loadedAt time.Time
}

var (
	instance *EngineAdapter
	once     sync.Once
)

func GetAdapter() *EngineAdapter {
	once.Do(func() {
		instance = &EngineAdapter{
			engines:  make(map[string]*WrappedEngine),
			requests: make(chan *InferenceRequest, 100),
			done:     make(chan struct{}),
		}
		go instance.processRequests()
	})
	return instance
}

func (a *EngineAdapter) processRequests() {
	for {
		select {
		case req := <-a.requests:
			a.handleRequest(req)
		case <-a.done:
			return
		}
	}
}

func (a *EngineAdapter) handleRequest(req *InferenceRequest) {
	log.Printf("Inference request: model=%s, prompt=%s", req.Model, req.Prompt)

	responseChan := make(chan InferenceResponse, req.MaxTokens)

	select {
	case req.ResponseChan <- responseChan:
	default:
		log.Printf("Request queue full, dropping request")
		return
	}

	for i := 0; i < req.MaxTokens; i++ {
		select {
		case responseChan <- InferenceResponse{
			Token:    "token",
			TokenID:  i,
			Complete: i == req.MaxTokens-1,
		}:
		default:
			log.Printf("Response channel full, stopping generation")
			break
		}

		if i == req.MaxTokens-1 {
			responseChan <- InferenceResponse{
				Complete: true,
			}
		}
	}
	close(responseChan)
}

func (a *EngineAdapter) GetEngine(modelPath string) (*WrappedEngine, error) {
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

	log.Printf("Loading engine for model: %s", modelPath)

	wrapped := &WrappedEngine{
		model:    modelPath,
		loadedAt: time.Now(),
	}

	a.engines[modelPath] = wrapped
	return wrapped, nil
}

func (a *EngineAdapter) ListModels() []ModelInfo {
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

func (a *EngineAdapter) Close() {
	close(a.done)
	a.mu.Lock()
	defer a.mu.Unlock()
	a.engines = make(map[string]*WrappedEngine)
}

func (a *EngineAdapter) Infer(ctx context.Context, req *InferenceRequest) (<-chan chan InferenceResponse, error) {
	responseChan := make(chan chan InferenceResponse, 1)

	select {
	case a.requests <- req:
		return responseChan, nil
	default:
		return nil, nil
	}
}

func (a *EngineAdapter) UnloadModel(modelPath string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	delete(a.engines, modelPath)
}
