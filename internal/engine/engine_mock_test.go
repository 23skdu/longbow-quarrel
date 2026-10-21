package engine

import (
	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/logger"
)

// MockEngine is a stub engine that returns fixed values.
//
// It lives in a test file: it is a test double, and a test double in a
// production package is compiled into every binary that imports the
// package.
//
// It is deliberately NOT registered with the production engine registry: it
// used to be, and because Infer returns `count` copies of token 42 with a
// nil error, an engine-construction failure on a host without CUDA or Metal
// silently produced plausible-looking garbage. Use NewMockEngine directly
// from a test instead.
type MockEngine struct {
	cfg config.Config
}

func NewMockEngine(modelPath string, cfg config.Config) (Engine, error) {
	logger.Log.Info("Mock engine initialized", "model", modelPath)
	return &MockEngine{cfg: cfg}, nil
}

func (e *MockEngine) Infer(tokens []int, count int, cfg SamplerConfig) ([]int, error) {
	result := make([]int, count)
	for i := range result {
		result[i] = 42 // Mock token
	}
	return result, nil
}

func (e *MockEngine) InferWithLogits(tokens []int, count int, cfg SamplerConfig) ([]int, []float32, error) {
	tokens_gen, _ := e.Infer(tokens, count, cfg)
	logits := make([]float32, 128) // Small mock vocab
	return tokens_gen, logits, nil
}

func (e *MockEngine) InferWithCallback(tokens []int, count int, cfg SamplerConfig, callback func(token int)) ([]int, error) {
	result := make([]int, count)
	for i := range result {
		result[i] = 42
		if callback != nil {
			callback(result[i])
		}
	}
	return result, nil
}

func (e *MockEngine) InferWithCallbackLogits(tokens []int, count int, cfg SamplerConfig, tokenCallback func(int), logitsCallback func([]float32)) ([]int, error) {
	result := make([]int, count)
	logits := make([]float32, 128)
	for i := range result {
		result[i] = 42
		if tokenCallback != nil {
			tokenCallback(result[i])
		}
		if logitsCallback != nil {
			logitsCallback(logits)
		}
	}
	return result, nil
}

func (e *MockEngine) Config() config.Config {
	return e.cfg
}

func (e *MockEngine) Close() {
	logger.Log.Info("Mock engine closed")
}

func (e *MockEngine) SwapModel(modelPath string, cfg config.Config) error {
	e.cfg = cfg
	return nil
}

func (e *MockEngine) LoadAdapter(path, id string) error {
	return nil
}

func (e *MockEngine) GetSeqCachePos(seqID string) int {
	return 0
}

func (e *MockEngine) ForwardDraft(tokens []int) ([][]float32, error) {
	return nil, nil
}

func (e *MockEngine) RollbackKV(seqID string, newPos int) error {
	return nil
}

func (e *MockEngine) ForwardBatch(desc *BatchDescriptor) ([]*device.Tensor, error) {
	return nil, nil // Mock implementation
}
