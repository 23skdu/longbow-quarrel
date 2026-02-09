//go:build !(darwin && metal)

package engine

import (
	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/cpu"
)

type Engine struct {
	ctx *cpu.Context
	cfg config.Config
}

func NewEngine(modelPath string, cfg config.Config) (*Engine, error) {
	ctx := cpu.NewContext()
	return &Engine{
		ctx: ctx,
		cfg: cfg,
	}, nil
}

func (e *Engine) Close() error {
	e.ctx.Free()
	return nil
}

func (e *Engine) Infer(inputs []int, genLen int, sampler SamplerConfig) ([]int, error) {
	return nil, nil
}

func (e *Engine) InferWithCallback(inputs []int, genLen int, sampler SamplerConfig, cb func(int)) ([]int, error) {
	return nil, nil
}
