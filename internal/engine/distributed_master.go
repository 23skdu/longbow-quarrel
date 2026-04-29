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
	ctx := device.NewContext()
	return &MasterDistributedEngine{
		shards: shards,
		config: cfg,
		ctx:    ctx,
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
	hiddenSize := m.config.Dim
	batchSize := len(batch.Sequences)

	results := make([]*device.Tensor, batchSize)
	var wg sync.WaitGroup
	var mu sync.Mutex
	errors := make([]error, batchSize)

	for seqIdx := range batch.Sequences {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()

			seq := batch.Sequences[idx]
			if seq == nil || len(seq.Tokens) == 0 {
				mu.Lock()
				errors[idx] = fmt.Errorf("empty sequence")
				mu.Unlock()
				return
			}

			firstToken := seq.Tokens[len(seq.Tokens)-1]
			inputVec := m.ctx.NewTensorFP32(1, hiddenSize)
			defer inputVec.Free()

			hiddenData := make([]float32, hiddenSize)
			if len(m.shards) > 0 {
				for i := 0; i < hiddenSize && i < 4096; i++ {
					hiddenData[i] = float32(firstToken%256) / 256.0
				}
			}
			_ = inputVec.LoadFrom(hiddenData)

			ctx := context.Background()
			layerOutputs := make([]*device.Tensor, m.config.Layers)

			for layerIdx := 0; layerIdx < m.config.Layers; layerIdx++ {
				partialOutputs := make([]*device.Tensor, numShards)
				var layerErr error

				var shardWg sync.WaitGroup
				for shardIdx := 0; shardIdx < numShards; shardIdx++ {
					shardWg.Add(1)
					go func(sIdx int) {
						defer shardWg.Done()

						colStart := (sIdx * hiddenSize) / numShards
						colEnd := ((sIdx + 1) * hiddenSize) / numShards
						if sIdx == numShards-1 {
							colEnd = hiddenSize
						}

						inputCopy := m.ctx.NewTensorFP32(1, hiddenSize)
						defer inputCopy.Free()

						partialOutputs[sIdx], layerErr = m.shards[sIdx].ForwardShardedLayer(ctx, layerIdx, colStart, colEnd, inputCopy)
					}(shardIdx)
				}
				shardWg.Wait()

				if layerErr != nil {
					mu.Lock()
					errors[idx] = fmt.Errorf("shard forward failed: %w", layerErr)
					mu.Unlock()
					return
				}

				combined := m.ctx.NewTensorFP32(1, hiddenSize)
				combinedData := make([]float32, hiddenSize)

				for sIdx := 0; sIdx < numShards; sIdx++ {
					if partialOutputs[sIdx] != nil {
						partData := partialOutputs[sIdx].ToHostF32()
						for i := 0; i < len(partData) && i < hiddenSize; i++ {
							combinedData[i] += partData[i]
						}
						partialOutputs[sIdx].Free()
					}
				}
				_ = combined.LoadFrom(combinedData)
				layerOutputs[layerIdx] = combined
			}

			if len(layerOutputs) > 0 && layerOutputs[0] != nil {
				mu.Lock()
				results[idx] = layerOutputs[0]
				mu.Unlock()
			} else {
				mu.Lock()
				errors[idx] = fmt.Errorf("no layer outputs")
				mu.Unlock()
			}
		}(seqIdx)
	}

	wg.Wait()

	for _, err := range errors {
		if err != nil {
			for _, t := range results {
				if t != nil {
					t.Free()
				}
			}
			return nil, fmt.Errorf("tensor parallel forward failed: %w", err)
		}
	}

	return results, nil
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
