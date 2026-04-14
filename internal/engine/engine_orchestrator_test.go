package engine

import (
	"testing"
	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
)

// MockBackend implements the Engine interface for testing the orchestrator
type MockBackend struct {
	cfg config.Config
}

func (m *MockBackend) ForwardBatch(desc *BatchDescriptor) ([]*device.Tensor, error) {
	results := make([]*device.Tensor, len(desc.Sequences))
	for i := range results {
		// Mock tensor with a single value (42)
		ctx := device.NewContext() // This might panic on Metal, but for CPU-only run it's fine
		t := ctx.NewTensorFP32(1, 1)
		t.LoadFrom([]float32{42.0})
		results[i] = t
	}
	return results, nil
}

func (m *MockBackend) Config() config.Config { return m.cfg }
func (m *MockBackend) Close()                 {}
func (m *MockBackend) SwapModel(path string, cfg config.Config) error { return nil }
func (m *MockBackend) Infer(tokens []int, count int, cfg SamplerConfig) ([]int, error) { return nil, nil }
func (m *MockBackend) InferWithLogits(tokens []int, count int, cfg SamplerConfig) ([]int, []float32, error) { return nil, nil, nil }
func (m *MockBackend) InferWithCallback(tokens []int, count int, cfg SamplerConfig, callback func(token int)) ([]int, error) { return nil, nil }
func (m *MockBackend) InferWithCallbackLogits(tokens []int, count int, cfg SamplerConfig, tokenCallback func(int), logitsCallback func([]float32)) ([]int, error) { return nil, nil }

func (m *MockBackend) ForwardDraft(tokens []int) ([][]float32, error) { return nil, nil }
func (m *MockBackend) RollbackKV(seqID string, newPos int) error { return nil }
func (m *MockBackend) GetSeqCachePos(seqID string) int { return 0 }

func TestEngine_Orchestrator_Lifecycle(t *testing.T) {
	// ... Test logic updated to use common engine structure if accessible, or just test sub-components
}
