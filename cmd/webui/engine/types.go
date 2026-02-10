package engine

import (
	"context"
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

type EngineAdapter interface {
	Infer(ctx context.Context, req *InferenceRequest) (<-chan chan InferenceResponse, error)
	ListModels() []ModelInfo
	Close()
}

func GetAdapter() EngineAdapter {
	return getAdapterImpl()
}
