package engine

import (
	"context"
	"errors"
)

// ErrQueueFull is returned by an adapter when the request queue is saturated.
//
// The adapters used to return a nil channel and a nil error in that case,
// which a caller cannot distinguish from a successful submission: the request
// simply never completed.
var ErrQueueFull = errors.New("request queue is full")

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
	// Error is set when the request could not be served. Without it a
	// failure was indistinguishable from a model that produced no tokens:
	// the client got HTTP 200 and an empty completion.
	Error string
}

type ModelInfo struct {
	Name         string
	Path         string
	Parameters   string
	Quantization string
	Loaded       bool
}

type EngineAdapter interface {
	Infer(ctx context.Context, req *InferenceRequest) (<-chan chan InferenceResponse, error)
	ListModels() []ModelInfo
	LoadModel(modelPath string) error
	UnloadModel(modelPath string)
	HotSwapModel(oldModelPath, newModelPath string) error
	Close()
}

func GetAdapter() EngineAdapter {
	return getAdapterImpl()
}
