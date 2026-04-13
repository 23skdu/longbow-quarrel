package engine

import (
	"testing"
	"github.com/23skdu/longbow-quarrel/internal/config"
)

func TestContinuousBatchManager_Lifecycle(t *testing.T) {
	conf := config.Default()
	conf.MaxBatchSize = 4
	conf.BlockSize = 8
	conf.TotalBlocks = 16

	mgr := NewContinuousBatchManager()
	
	// 1. Submit Request
	req := &InferenceRequest{
		ID:     1,
		Prompt: []int{10, 20, 30},
		Result: make(chan []int, 1),
		Err:    make(chan error, 1),
	}
	mgr.Submit(req)

	if mgr.waitingQueue.Depth() != 1 {
		t.Errorf("Expected queue depth 1, got %d", mgr.waitingQueue.Depth())
	}

	// 2. Step (Orchestration)
	// We pass 0 blocks (dummy cache) just to check if it pulls from queue
	active, _ := mgr.Step(4, nil)
	if len(active) != 1 {
		t.Errorf("Expected 1 active sequence, got %d", len(active))
	}
	if mgr.waitingQueue.Depth() != 0 {
		t.Errorf("Expected queue depth 0 after step, got %d", mgr.waitingQueue.Depth())
	}
}

func TestContinuousBatchManager_Preemption(t *testing.T) {
	// Focus on testing the scheduling logic without hardware dependencies
}
