//go:build !darwin || !metal

package engine

import (
	"errors"
	"github.com/23skdu/longbow-quarrel/internal/config"
)

// No init function here as metal shouldn't be registered on non-Metal platforms,
// OR we register it as a "no-op" for testing. 
// For now, let's just make it compilable for tests.

func (e *metalEngine) Infer(tokens []int, count int, cfg SamplerConfig) ([]int, error) {
	return nil, errors.New("metal engine not available on this platform")
}

func (e *metalEngine) InferWithLogits(tokens []int, count int, cfg SamplerConfig) ([]int, []float32, error) {
	return nil, nil, errors.New("metal engine not available on this platform")
}

func (e *metalEngine) InferWithCallback(tokens []int, count int, cfg SamplerConfig, callback func(token int)) ([]int, error) {
	return nil, errors.New("metal engine not available on this platform")
}

func (e *metalEngine) InferWithCallbackLogits(tokens []int, count int, cfg SamplerConfig, tokenCallback func(int), logitsCallback func([]float32)) ([]int, error) {
	return nil, errors.New("metal engine not available on this platform")
}

func (e *metalEngine) Close() {
	// No-op
}

func (e *metalEngine) SwapModel(modelPath string, cfg config.Config) error {
	return errors.New("metal engine not available on this platform")
}

func (e *metalEngine) initKVCache() error {
	return nil
}

func (e *metalEngine) loadModel(path string) error {
	return errors.New("metal engine not available on this platform")
}

func (e *metalEngine) detectMambaLayers(f interface{}, logger interface{}) {
	// No-op
}

