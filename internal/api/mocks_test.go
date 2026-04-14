package api

import (
	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/engine"
)

type mockEngine struct {
	cfg config.Config
}

func (m *mockEngine) Infer(tokens []int, count int, cfg engine.SamplerConfig) ([]int, error) {
	return []int{1, 2, 3}, nil
}

func (m *mockEngine) InferWithLogits(tokens []int, count int, cfg engine.SamplerConfig) ([]int, []float32, error) {
	return []int{1, 2, 3}, make([]float32, 128), nil
}

func (m *mockEngine) InferWithCallback(tokens []int, count int, cfg engine.SamplerConfig, callback func(token int)) ([]int, error) {
	return []int{1, 2, 3}, nil
}

func (m *mockEngine) InferWithCallbackLogits(tokens []int, count int, cfg engine.SamplerConfig, tokenCallback func(int), logitsCallback func([]float32)) ([]int, error) {
	return []int{1, 2, 3}, nil
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
	return [][]float32{{0.1, 0.9}}, nil
}

func (m *mockEngine) RollbackKV(seqID string, newPos int) error { return nil }

func (m *mockEngine) ForwardBatch(desc *engine.BatchDescriptor) ([]*device.Tensor, error) {
	return nil, nil
}

type mockTokenizer struct{}

func (m *mockTokenizer) Encode(s string) []int       { return []int{10} }
func (m *mockTokenizer) Decode(t []int) string       { return "test response" }
func (m *mockTokenizer) EncodeTokens(s string) []int { return []int{10} }
func (m *mockTokenizer) DecodeTokens(t []int) string { return "test response" }
func (m *mockTokenizer) GetVocab() []string          { return []string{"test", "vocab"} }
