package engine

import (
	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/logger"
)

type mockEngine struct {
	cfg config.Config
}

func init() {
	RegisterEngine("mock", NewMockEngine)
}

func NewMockEngine(modelPath string, cfg config.Config) (Engine, error) {
	logger.Log.Info("Mock engine initialized", "model", modelPath)
	return &mockEngine{cfg: cfg}, nil
}

func (e *mockEngine) Infer(tokens []int, count int, cfg SamplerConfig) ([]int, error) {
	result := make([]int, count)
	for i := range result {
		result[i] = 42 // Mock token
	}
	return result, nil
}

func (e *mockEngine) InferWithLogits(tokens []int, count int, cfg SamplerConfig) ([]int, []float32, error) {
	tokens_gen, _ := e.Infer(tokens, count, cfg)
	logits := make([]float32, 128) // Small mock vocab
	return tokens_gen, logits, nil
}

func (e *mockEngine) InferWithCallback(tokens []int, count int, cfg SamplerConfig, callback func(token int)) ([]int, error) {
	result := make([]int, count)
	for i := range result {
		result[i] = 42
		if callback != nil {
			callback(result[i])
		}
	}
	return result, nil
}

func (e *mockEngine) InferWithCallbackLogits(tokens []int, count int, cfg SamplerConfig, tokenCallback func(int), logitsCallback func([]float32)) ([]int, error) {
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

func (e *mockEngine) Config() config.Config {
	return e.cfg
}

func (e *mockEngine) Close() {
	logger.Log.Info("Mock engine closed")
}

func (e *mockEngine) SwapModel(modelPath string, cfg config.Config) error {
	e.cfg = cfg
	return nil
}

func (e *mockEngine) GetSeqCachePos(seqID int) int {
	return 0
}
