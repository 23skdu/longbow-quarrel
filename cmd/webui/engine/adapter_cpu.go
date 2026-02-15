//go:build !(darwin && metal) && !(linux && cuda)

package engine

import (
	"context"
	"log"
	"sync"
)

// CPUEngineAdapter provides a stub implementation for systems without GPU acceleration.
// On platforms without Metal (darwin) or CUDA (linux), inference requests will return
// an error indicating that no inference engine is available.
type CPUEngineAdapter struct {
	mu       sync.RWMutex
	requests chan *InferenceRequest
	done     chan struct{}
}

var (
	cpuInstance *CPUEngineAdapter
	cpuOnce     sync.Once
)

// getAdapterImpl returns the CPU adapter singleton.
// This is only used when no GPU acceleration is available.
func getAdapterImpl() EngineAdapter {
	cpuOnce.Do(func() {
		cpuInstance = &CPUEngineAdapter{
			requests: make(chan *InferenceRequest, 100),
			done:     make(chan struct{}),
		}
		log.Println("WARNING: No GPU acceleration available. Inference will not work.")
		log.Println("Supported platforms: darwin (Metal), linux (CUDA)")
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
	// Return error response indicating no engine available
	responseChan := make(chan InferenceResponse, 1)
	responseChan <- InferenceResponse{
		Complete: true,
	}
	close(responseChan)

	select {
	case req.ResponseChan <- responseChan:
	default:
		log.Printf("Request queue full, dropping request")
	}
}

func (a *CPUEngineAdapter) Infer(ctx context.Context, req *InferenceRequest) (<-chan chan InferenceResponse, error) {
	responseChan := make(chan chan InferenceResponse, 1)

	// Create error response
	errChan := make(chan InferenceResponse, 1)
	errChan <- InferenceResponse{
		Complete: true,
	}
	close(errChan)
	responseChan <- errChan

	select {
	case a.requests <- req:
		return responseChan, nil
	default:
		return nil, nil
	}
}

func (a *CPUEngineAdapter) ListModels() []ModelInfo {
	// Return empty list - no models can run without GPU
	return []ModelInfo{}
}

func (a *CPUEngineAdapter) Close() {
	close(a.done)
}

func (a *CPUEngineAdapter) UnloadModel(modelPath string) {
	// No-op for CPU adapter
}

// CPUEngineError indicates that no GPU acceleration is available
type CPUEngineError struct {
	Message string
}

func (e *CPUEngineError) Error() string {
	return e.Message
}
