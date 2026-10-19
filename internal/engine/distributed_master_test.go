package engine

import (
	"context"
	"errors"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
)

// stubShard is a DistributedEngine that records what it was asked to do.
type stubShard struct {
	MockEngine
	role        ShardRole
	forwardErr  error
	adapterErr  error
	swapErr     error
	forwardCall int
	shardCalls  int
}

func (s *stubShard) ShardRole() ShardRole { return s.role }

func (s *stubShard) SyncWeights(ctx context.Context) error { return nil }

func (s *stubShard) ForwardShard(ctx context.Context, input *device.Tensor) (*device.Tensor, error) {
	return nil, nil
}

func (s *stubShard) ForwardShardedLayer(ctx context.Context, layerIdx, colStart, colEnd int, input *device.Tensor) (*device.Tensor, error) {
	s.shardCalls++
	if s.forwardErr != nil {
		return nil, s.forwardErr
	}
	out := &device.Tensor{}
	_ = out
	return nil, nil
}

func (s *stubShard) LoadAdapter(path, id string) error {
	s.forwardCall++
	return s.adapterErr
}

func (s *stubShard) SwapModel(modelPath string, cfg config.Config) error {
	return s.swapErr
}

func (s *stubShard) Close() { s.forwardCall++ }

// TestMasterDistributedEngine_NoShards checks that every entry point
// handles an empty shard list.
//
// Half of them indexed the first shard without checking, so a master with
// no shards panicked.
func TestMasterDistributedEngine_NoShards(t *testing.T) {
	m := NewMasterDistributedEngine(nil, config.Default())
	defer m.Close()

	if _, err := m.Infer([]int{1}, 1, SamplerConfig{}); err == nil {
		t.Error("expected an error from Infer with no shards")
	}
	if _, err := m.InferWithCallback([]int{1}, 1, SamplerConfig{}, nil); err == nil {
		t.Error("expected an error from InferWithCallback with no shards")
	}
	if _, _, err := m.InferWithLogits([]int{1}, 1, SamplerConfig{}); err == nil {
		t.Error("expected an error from InferWithLogits with no shards")
	}
	if _, err := m.InferWithCallbackLogits([]int{1}, 1, SamplerConfig{}, nil, nil); err == nil {
		t.Error("expected an error from InferWithCallbackLogits with no shards")
	}
	if _, err := m.ForwardBatch(&BatchDescriptor{}); err == nil {
		t.Error("expected an error from ForwardBatch with no shards")
	}
	if _, err := m.ForwardDraft([]int{1}); err == nil {
		t.Error("expected an error from ForwardDraft with no shards")
	}
	if got := m.GetSeqCachePos("seq-1"); got != 0 {
		t.Errorf("the cache position is %d, want 0", got)
	}
}

// TestMasterDistributedEngine_DelegatesToThePrimaryShard checks that the
// master forwards the work to its first shard.
func TestMasterDistributedEngine_DelegatesToThePrimaryShard(t *testing.T) {
	shard := &stubShard{role: RoleMaster}
	m := NewMasterDistributedEngine([]DistributedEngine{shard}, config.Default())
	defer m.Close()

	if _, err := m.Infer([]int{1, 2}, 3, SamplerConfig{}); err != nil {
		t.Errorf("Infer failed: %v", err)
	}
	if _, err := m.InferWithCallback([]int{1, 2}, 3, SamplerConfig{}, nil); err != nil {
		t.Errorf("InferWithCallback failed: %v", err)
	}
	if _, _, err := m.InferWithLogits([]int{1, 2}, 3, SamplerConfig{}); err != nil {
		t.Errorf("InferWithLogits failed: %v", err)
	}
	if _, err := m.InferWithCallbackLogits([]int{1, 2}, 3, SamplerConfig{}, nil, nil); err != nil {
		t.Errorf("InferWithCallbackLogits failed: %v", err)
	}
	if _, err := m.ForwardDraft([]int{1, 2}); err != nil {
		t.Errorf("ForwardDraft failed: %v", err)
	}
}

// TestMasterDistributedEngine_BroadcastsToEveryShard checks that the
// operations which must reach every shard do.
func TestMasterDistributedEngine_BroadcastsToEveryShard(t *testing.T) {
	shards := []DistributedEngine{
		&stubShard{role: RoleMaster},
		&stubShard{role: RoleWorker},
		&stubShard{role: RoleWorker},
	}
	m := NewMasterDistributedEngine(shards, config.Default())
	defer m.Close()

	if err := m.LoadAdapter("/tmp/adapter.gguf", "adapter-1"); err != nil {
		t.Errorf("LoadAdapter failed: %v", err)
	}
	for i, s := range shards {
		if got := s.(*stubShard).forwardCall; got != 1 {
			t.Errorf("shard %d was asked to load the adapter %d times, want 1", i, got)
		}
	}

	if err := m.RollbackKV("seq-1", 5); err != nil {
		t.Errorf("RollbackKV failed: %v", err)
	}
}

// TestMasterDistributedEngine_AdapterErrorIsReported checks that a shard
// that refuses the adapter fails the whole operation.
func TestMasterDistributedEngine_AdapterErrorIsReported(t *testing.T) {
	shards := []DistributedEngine{
		&stubShard{role: RoleMaster},
		&stubShard{role: RoleWorker, adapterErr: errors.New("no such adapter")},
	}
	m := NewMasterDistributedEngine(shards, config.Default())
	defer m.Close()

	if err := m.LoadAdapter("/tmp/adapter.gguf", "adapter-1"); err == nil {
		t.Error("expected the adapter error to be reported")
	}
}

// TestMasterDistributedEngine_ForwardBatchRejectsEmptySequences checks the
// error path of the sharded forward.
//
// The check lives in the multi-shard path: with a single shard the master
// hands the batch straight to it.
func TestMasterDistributedEngine_ForwardBatchRejectsEmptySequences(t *testing.T) {
	shards := []DistributedEngine{
		&stubShard{role: RoleMaster},
		&stubShard{role: RoleWorker},
	}
	m := NewMasterDistributedEngine(shards, config.Default())
	defer m.Close()

	batch := &BatchDescriptor{
		Sequences: []*Sequence{{ID: 1}},
	}
	if _, err := m.ForwardBatch(batch); err == nil {
		t.Error("expected an error for a sequence with no tokens")
	}
}

// TestMasterDistributedEngine_ForwardBatchReportsShardErrors checks that a
// failing shard fails the forward.
func TestMasterDistributedEngine_ForwardBatchReportsShardErrors(t *testing.T) {
	shards := []DistributedEngine{
		&stubShard{role: RoleMaster},
		&stubShard{role: RoleWorker, forwardErr: errors.New("the worker is gone")},
	}
	cfg := config.Default()
	cfg.Layers = 2
	cfg.Dim = 8
	m := NewMasterDistributedEngine(shards, cfg)
	defer m.Close()

	batch := &BatchDescriptor{
		Sequences: []*Sequence{{ID: 1, Tokens: []int{5, 6}}},
	}
	if _, err := m.ForwardBatch(batch); err == nil {
		t.Error("expected the shard error to be reported")
	}
}
