package engine

import (
	"context"
	"fmt"
	"sync"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
)

// MasterDistributedEngine coordinates a cluster of distributed engine workers.
// It implements the standard Engine interface but shatters requests across shards.
type MasterDistributedEngine struct {
	shards []DistributedEngine
	config config.Config
	ctx    *device.Context
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
	if len(m.shards) == 0 {
		return nil, fmt.Errorf("no distributed shards available")
	}

	if len(m.shards) == 1 {
		return m.shards[0].ForwardBatch(batch)
	}

	numShards := len(m.shards)

	hiddenSize := m.config.HiddenSize
	shardSize := hiddenSize / numShards
	ctx := context.Background()

	var wg sync.WaitGroup
	partialResults := make([]*device.Tensor, numShards)
	errs := make([]error, numShards)

	for i := 0; i < numShards; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			input := batch.Sequences[idx].CachedEmbedding
			if input == nil {
				errs[idx] = fmt.Errorf("no cached embedding for sequence %d", idx)
				return
			}
			partialResults[idx], errs[idx] = m.shards[idx].ForwardShardedLayer(ctx, 0, idx*shardSize, (idx+1)*shardSize, input)
		}(i)
	}
	wg.Wait()

	for _, err := range errs {
		if err != nil {
			for _, t := range partialResults {
				if t != nil {
					t.Free()
				}
			}
			return nil, fmt.Errorf("tensor parallel forward failed: %w", err)
		}
	}

	layerOutputs := make([]*device.Tensor, m.config.NumLayers)
	for layerIdx := 0; layerIdx < m.config.NumLayers; layerIdx++ {
		result := partialResults[0]
		for i := 1; i < numShards; i++ {
			if partialResults[i] != nil {
				result.AddInto(result, partialResults[i])
				partialResults[i].Free()
			}
		}
		layerOutputs[layerIdx] = result
	}

	return layerOutputs, nil
}

func (m *MasterDistributedEngine) SetContext(ctx *device.Context) {
	m.ctx = ctx
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
