package engine

import (
	"fmt"
	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
)

// MasterDistributedEngine coordinates a cluster of distributed engine workers.
// It implements the standard Engine interface but shatters requests across shards.
type MasterDistributedEngine struct {
	shards []DistributedEngine
	config config.Config
}

func NewMasterDistributedEngine(shards []DistributedEngine, cfg config.Config) *MasterDistributedEngine {
	return &MasterDistributedEngine{
		shards: shards,
		config: cfg,
	}
}

func (m *MasterDistributedEngine) Infer(tokens []int, count int, cfg SamplerConfig) ([]int, error) {
	// Simple Pipeline Parallelism strategy:
	// For now, we assume the first node is the one we talk to,
	// or we coordinate the shards sequentially.
	if len(m.shards) == 0 {
		return nil, fmt.Errorf("no distributed shards available")
	}
	
	// Delegate to primary shard (Master role)
	return m.shards[0].Infer(tokens, count, cfg)
}

func (m *MasterDistributedEngine) InferWithCallback(tokens []int, count int, cfg SamplerConfig, callback func(int)) ([]int, error) {
	if len(m.shards) == 0 {
		return nil, fmt.Errorf("no distributed shards available")
	}
	return m.shards[0].InferWithCallback(tokens, count, cfg, callback)
}

func (m *MasterDistributedEngine) InferWithLogits(tokens []int, count int, cfg SamplerConfig) ([]int, []float32, error) {
	return m.shards[0].InferWithLogits(tokens, count, cfg)
}

func (m *MasterDistributedEngine) InferWithCallbackLogits(tokens []int, count int, cfg SamplerConfig, tokenCallback func(int), logitsCallback func([]float32)) ([]int, error) {
	return m.shards[0].InferWithCallbackLogits(tokens, count, cfg, tokenCallback, logitsCallback)
}

func (m *MasterDistributedEngine) ForwardBatch(batch *BatchDescriptor) ([]*device.Tensor, error) {
	// Tensor Parallelism implementation:
	// 1. Broadcast batch info to all shards.
	// 2. Each shard computes its partial activation.
	// 3. Collect and All-Reduce results.
	
	// Stub: currently delegating to primary
	return m.shards[0].ForwardBatch(batch)
}

func (m *MasterDistributedEngine) Config() config.Config {
	return m.config
}

func (m *MasterDistributedEngine) LoadAdapter(path, id string) error {
	// Broadcast adapter load to all shards to ensure parity
	for _, s := range m.shards {
		if err := s.LoadAdapter(path, id); err != nil {
			return fmt.Errorf("shard failed to load adapter: %w", err)
		}
	}
	return nil
}

func (m *MasterDistributedEngine) RollbackKV(seqID string, newPos int) error {
	for _, s := range m.shards {
		_ = s.RollbackKV(seqID, newPos)
	}
	return nil
}

func (m *MasterDistributedEngine) ForwardDraft(tokens []int) ([][]float32, error) {
	return m.shards[0].ForwardDraft(tokens)
}

func (m *MasterDistributedEngine) Close() {
	for _, s := range m.shards {
		s.Close()
	}
}

func (m *MasterDistributedEngine) SwapModel(modelPath string, cfg config.Config) error {
	for _, s := range m.shards {
		if err := s.SwapModel(modelPath, cfg); err != nil {
			return err
		}
	}
	return nil
}

func (m *MasterDistributedEngine) GetSeqCachePos(seqID string) int {
	return m.shards[0].GetSeqCachePos(seqID)
}
