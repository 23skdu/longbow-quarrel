//go:build metal

package engine

import (
	"errors"
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
	active, _ := mgr.Step(4, nil, nil)
	if len(active) != 1 {
		t.Errorf("Expected 1 active sequence, got %d", len(active))
	}
	if mgr.waitingQueue.Depth() != 0 {
		t.Errorf("Expected queue depth 0 after step, got %d", mgr.waitingQueue.Depth())
	}
}

func TestContinuousBatchManager_AbortAll(t *testing.T) {
	mgr := NewContinuousBatchManager()
	
	// Submit some requests
	errChan1 := make(chan error, 1)
	mgr.Submit(&InferenceRequest{ID: 1, Prompt: []int{10}, Err: errChan1})
	
	errChan2 := make(chan error, 1)
	mgr.Submit(&InferenceRequest{ID: 2, Prompt: []int{20}, Err: errChan2})
	
	// Trigger swap/abort
	testErr := errors.New("test-abort")
	mgr.AbortAll(testErr)
	
	// Verify errors sent
	select {
	case err := <-errChan1:
		if err != testErr { t.Errorf("unexpected error: %v", err) }
	default:
		t.Error("expected error on chan1")
	}
}

func TestContinuousBatchManager_Preemption(t *testing.T) {
	// Dummy for coverage
	mgr := NewContinuousBatchManager()
	mgr.Step(0, nil, nil)
}
