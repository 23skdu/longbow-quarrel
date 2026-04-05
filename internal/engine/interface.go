package engine

import (
	"github.com/23skdu/longbow-quarrel/internal/config"
)

// Engine is the common interface for all inference backends (Metal, CUDA, CPU)
type Engine interface {
	Infer(tokens []int, count int, cfg SamplerConfig) ([]int, error)
	InferWithLogits(tokens []int, count int, cfg SamplerConfig) ([]int, []float32, error)
	InferWithCallback(tokens []int, count int, cfg SamplerConfig, callback func(token int)) ([]int, error)
	InferWithCallbackLogits(tokens []int, count int, cfg SamplerConfig, tokenCallback func(int), logitsCallback func([]float32)) ([]int, error)
	Config() config.Config
	Close()
	SwapModel(modelPath string, cfg config.Config) error
	GetSeqCachePos(seqID int) int
}

// EngineCreator defines the factory function for creating an engine
type EngineCreator func(modelPath string, cfg config.Config) (Engine, error)

var engineCreators = make(map[string]EngineCreator)

// RegisterEngine registers a new engine implementation with a name
func RegisterEngine(name string, creator EngineCreator) {
	engineCreators[name] = creator
}
