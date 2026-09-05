package engine

import (
	"fmt"
	"testing"
)

func TestContinuousBatchManager_Stress(t *testing.T) {
	mgr := NewContinuousBatchManager()
	
	// 1. Fill queue with multiple sequences
	for i := 0; i < 10; i++ {
		tokens := make([]int, 10+i)
		for j := range tokens {
			tokens[j] = j
		}
		mgr.Submit(&InferenceRequest{
			ID: uint64(i),
			Prompt: tokens,
			MaxTokens: 20,
		})
	}
	
	if mgr.Depth() != 10 {
		t.Errorf("Expected depth 10, got %d", mgr.Depth())
	}
	
	// 2. Step through mixed batched execution
	// Mock step loop
	for step := 0; step < 30; step++ {
		batch, _ := mgr.Step(4, nil, nil) // Small batch size to force multiple steps, nil caches for test
		if batch == nil {
			if mgr.Depth() == 0 {
				break
			}
			continue
		}
		
		// Simulate inference and callback
		for _, seq := range batch.Sequences {
			// Callback with a mock token
			mgr.CompleteSequence(seq.ID, nil)
			
			// Randomly abort some sequences
			if step == 5 && seq.ID == 3 {
				mgr.AbortAll(fmt.Errorf("stress test abort")) 
			}
		}
	}
}

func TestContinuousBatchManager_QueueUpTo(t *testing.T) {
	mgr := NewContinuousBatchManager()
	
	for i := 0; i < 20; i++ {
		mgr.Submit(&InferenceRequest{
			ID: uint64(i),
			Prompt: []int{1, 2, 3},
			MaxTokens: 10,
		})
	}
	
	// depth check
	depth := mgr.Depth()
	if depth != 20 {
		t.Errorf("Expected 20 in queue, got %d", depth)
	}
}

func TestBatchDescriptor_Metadata(t *testing.T) {
	// Cover metadata packing logic
	mgr := NewContinuousBatchManager()
	mgr.Submit(&InferenceRequest{ID: 1, Prompt: []int{1, 2, 3}, MaxTokens: 10}) // Prefill (3 tokens)
	mgr.Submit(&InferenceRequest{ID: 2, Prompt: []int{4, 5}, MaxTokens: 10})    // Prefill (2 tokens)
	
	desc, _ := mgr.Step(2, nil, nil)
	if desc == nil {
		t.Fatal("Expected batch descriptor")
	}
	
	if len(desc.Tokens) != 5 {
		t.Errorf("Expected 5 tokens total, got %d", len(desc.Tokens))
	}
	
	if len(desc.Offsets) != 2 {
		t.Errorf("Expected 2 offsets, got %d", len(desc.Offsets))
	}
	
	if desc.Offsets[1] != 3 && desc.Offsets[1] != 2 {
		t.Errorf("Expected second offset at 2 or 3, got %d", desc.Offsets[1])
	}
	
	if len(desc.TokenToSeq) != 5 {
		t.Errorf("Expected TokenToSeq length 5, got %d", len(desc.TokenToSeq))
	}
}
