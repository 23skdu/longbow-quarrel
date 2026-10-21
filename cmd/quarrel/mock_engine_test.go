package main

import (
	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/engine"
)

// mockEngine is a stub engine for the tests of the binary.
//
// The repository used to carry a mock engine in the production package
// internal/engine, where it was compiled into every binary that imported
// the engine package. It is a test double, so it belongs in a test file.
type mockEngine struct {
	cfg config.Config
}

func (m *mockEngine) Infer(tokens []int, count int, cfg engine.SamplerConfig) ([]int, error) {
	result := make([]int, count)
	for i := range result {
		result[i] = 42
	}
	return result, nil
}

func (m *mockEngine) InferWithLogits(tokens []int, count int, cfg engine.SamplerConfig) ([]int, []float32, error) {
	generated, _ := m.Infer(tokens, count, cfg)
	return generated, make([]float32, 128), nil
}

func (m *mockEngine) InferWithCallback(tokens []int, count int, cfg engine.SamplerConfig, callback func(token int)) ([]int, error) {
	result, _ := m.Infer(tokens, count, cfg)
	_ = result
	if callback != nil {
		for _, token := range result {
			callback(token)
		}
	}
	return result, nil
}

func (m *mockEngine) InferWithCallbackLogits(tokens []int, count int, cfg engine.SamplerConfig, tokenCallback func(int), logitsCallback func([]float32)) ([]int, error) {
	result, _ := m.Infer(tokens, count, cfg)
	if tokenCallback != nil {
		for _, token := range result {
			tokenCallback(token)
		}
	}
	if logitsCallback != nil {
		logitsCallback(make([]float32, 128))
	}
	return result, nil
}

func (m *mockEngine) Config() config.Config { return m.cfg }
func (m *mockEngine) Close()                {}

func (m *mockEngine) SwapModel(modelPath string, cfg config.Config) error {
	m.cfg = cfg
	return nil
}

func (m *mockEngine) LoadAdapter(path, id string) error { return nil }
func (m *mockEngine) GetSeqCachePos(seqID string) int   { return 0 }

func (m *mockEngine) ForwardDraft(tokens []int) ([][]float32, error) {
	return nil, nil
}

func (m *mockEngine) RollbackKV(seqID string, newPos int) error { return nil }

func (m *mockEngine) ForwardBatch(desc *engine.BatchDescriptor) ([]*device.Tensor, error) {
	return nil, nil
}
