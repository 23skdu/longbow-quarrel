//go:build !(darwin && metal)

package engine

import (
	"errors"

	"github.com/23skdu/longbow-quarrel/internal/config"
)

// Engine is a stub for non-Metal environments.
type Engine struct {
	CachePos int
}

// NewEngine returns an error on non-Metal environments.
func NewEngine(modelPath string, cfg config.Config) (*Engine, error) {
	return nil, errors.New("engine requires Metal support (build tags: darwin, metal)")
}

// Close is a stub.
func (e *Engine) Close() error {
	return nil
}

// Infer is a stub.
func (e *Engine) Infer(inputs []int, genLen int, sampler SamplerConfig) ([]int, error) {
	return nil, errors.New("engine requires Metal support")
}

// InferWithCallback is a stub.
func (e *Engine) InferWithCallback(inputs []int, genLen int, sampler SamplerConfig, cb func(int)) ([]int, error) {
	return nil, errors.New("engine requires Metal support")
}
