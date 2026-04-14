package engine

import (
	"fmt"
	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
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
	LoadAdapter(path, id string) error
	GetSeqCachePos(seqID string) int

	// Speculative Decoding Primitives
	ForwardDraft(tokens []int) ([][]float32, error)
	RollbackKV(seqID string, newPos int) error

	// Continuous Batching
	ForwardBatch(desc *BatchDescriptor) ([]*device.Tensor, error)
}

// EngineCreator defines the factory function for creating an engine
type EngineCreator func(modelPath string, cfg config.Config) (Engine, error)

var engineCreators = make(map[string]EngineCreator)

// RegisterEngine registers a new engine implementation with a name
func RegisterEngine(name string, creator EngineCreator) {
	engineCreators[name] = creator
}

// NewRegisteredEngine creates the best available engine implementation
func NewRegisteredEngine(modelPath string, cfg config.Config) (Engine, error) {
	// Priority order
	for _, name := range []string{"metal", "cuda", "cpu", "mock"} {
		if creator, ok := engineCreators[name]; ok {
			return creator(modelPath, cfg)
		}
	}
	return nil, fmt.Errorf("no registered engine found")
}

// NewEngine is the standard entry point to create an engine.
// It uses the registered factory to pick the best available backend.
func NewEngine(modelPath string, cfg config.Config) (Engine, error) {
	return NewRegisteredEngine(modelPath, cfg)
}
