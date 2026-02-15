package engine

import (
	"github.com/23skdu/longbow-quarrel/internal/config"
)

type SamplerConfig struct {
	Temperature      float64
	TopK             int
	TopP             float64
	RepPenalty       float64
	Seed             int64
	DebugActivations bool
	QualityMode      bool
}

type Engine interface {
	Infer(tokens []int, count int, cfg SamplerConfig) ([]int, error)
	InferWithCallback(tokens []int, count int, cfg SamplerConfig, callback func(token int)) ([]int, error)
	Close()
}

type EngineCreator func(modelPath string, cfg config.Config) (Engine, error)

var engineCreators = make(map[string]EngineCreator)

func RegisterEngine(name string, creator EngineCreator) {
	engineCreators[name] = creator
}

func NewEngine(modelPath string, cfg config.Config) (Engine, error) {
	for _, creator := range engineCreators {
		engine, err := creator(modelPath, cfg)
		if err == nil {
			return engine, nil
		}
	}
	return nil, nil
}
