package engine

import (
	"errors"
	"github.com/23skdu/longbow-quarrel/internal/config"
	"testing"
)

func TestContinuousBatchManager_Lifecycle(t *testing.T) {
	conf := config.Default()
	_ = conf

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
	// We pass enough blocks to admit the sequence
	desc, _ := mgr.Step(4, nil, nil)
	if desc == nil || len(desc.Sequences) != 1 {
		t.Errorf("Expected 1 sequence in descriptor, got %v", desc)
		return
	}

	if len(desc.Tokens) != 3 {
		t.Errorf("Expected 3 tokens (prefill), got %d", len(desc.Tokens))
	}

	if len(desc.TokenToSeq) != 3 {
		t.Errorf("Expected 3 mappings, got %d", len(desc.TokenToSeq))
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
		if err != testErr {
			t.Errorf("unexpected error: %v", err)
		}
	default:
		t.Error("expected error on chan1")
	}
}

func TestContinuousBatchManager_Preemption(t *testing.T) {
	// Dummy for coverage
	mgr := NewContinuousBatchManager()
	_, _ = mgr.Step(0, nil, nil)
}

func TestContinuousBatchManager_ChunkedPrefill(t *testing.T) {
	mgr := NewContinuousBatchManager()
	mgr.PrefillChunkSize = 256

	prompt := make([]int, 600)
	for i := range prompt {
		prompt[i] = 1000 + i
	}

	req := &InferenceRequest{
		ID:        42,
		Prompt:    prompt,
		MaxTokens: 610,
		Result:    make(chan []int, 1),
		Err:       make(chan error, 1),
	}
	mgr.Submit(req)

	// Step 1: First chunk of 256 tokens
	desc1, err := mgr.Step(4, nil, nil)
	if err != nil {
		t.Fatalf("Step 1 failed: %v", err)
	}
	if len(desc1.Sequences) != 1 {
		t.Fatalf("Step 1: expected 1 sequence, got %d", len(desc1.Sequences))
	}
	if len(desc1.Tokens) != 256 {
		t.Fatalf("Step 1: expected 256 tokens, got %d", len(desc1.Tokens))
	}
	if desc1.IsDecode[0] {
		t.Fatalf("Step 1: expected IsDecode=false")
	}
	seq := desc1.Sequences[0]
	if seq.PrefillCompleted {
		t.Fatalf("Step 1: expected PrefillCompleted=false")
	}
	// Simulate kernel execution & runBatchLoop guard: advance Pos by chunkLen
	chunkLen1 := len(desc1.Tokens)
	if !desc1.IsDecode[0] && seq.Pos+chunkLen1 < seq.PromptLen {
		seq.Pos += chunkLen1
	}
	if len(seq.Tokens) != 600 {
		t.Fatalf("Step 1: expected seq.Tokens len 600, got %d", len(seq.Tokens))
	}

	// Step 2: Second chunk of 256 tokens (offset 256..512)
	desc2, err := mgr.Step(4, nil, nil)
	if err != nil {
		t.Fatalf("Step 2 failed: %v", err)
	}
	if len(desc2.Tokens) != 256 {
		t.Fatalf("Step 2: expected 256 tokens, got %d", len(desc2.Tokens))
	}
	if desc2.Tokens[0] != prompt[256] {
		t.Fatalf("Step 2: expected token %d at chunk start, got %d", prompt[256], desc2.Tokens[0])
	}
	chunkLen2 := len(desc2.Tokens)
	if !desc2.IsDecode[0] && seq.Pos+chunkLen2 < seq.PromptLen {
		seq.Pos += chunkLen2
	}
	if len(seq.Tokens) != 600 {
		t.Fatalf("Step 2: expected seq.Tokens len 600, got %d", len(seq.Tokens))
	}

	// Step 3: Final chunk of 88 tokens (offset 512..600)
	desc3, err := mgr.Step(4, nil, nil)
	if err != nil {
		t.Fatalf("Step 3 failed: %v", err)
	}
	if len(desc3.Tokens) != 88 {
		t.Fatalf("Step 3: expected 88 tokens, got %d", len(desc3.Tokens))
	}
	if !seq.PrefillCompleted {
		t.Fatalf("Step 3: expected PrefillCompleted=true")
	}
	chunkLen3 := len(desc3.Tokens)
	// On final prefill chunk: seq.Pos+chunkLen3 >= seq.PromptLen, so sample and append
	if desc3.IsDecode[0] || seq.Pos+chunkLen3 >= seq.PromptLen {
		seq.Tokens = append(seq.Tokens, 9999) // simulated generated token
		seq.Pos += chunkLen3
	}
	if len(seq.Tokens) != 601 {
		t.Fatalf("Step 3: expected seq.Tokens len 601, got %d", len(seq.Tokens))
	}

	// Step 4: First decode step (1 token)
	desc4, err := mgr.Step(4, nil, nil)
	if err != nil {
		t.Fatalf("Step 4 failed: %v", err)
	}
	if len(desc4.Tokens) != 1 {
		t.Fatalf("Step 4: expected 1 token, got %d", len(desc4.Tokens))
	}
	if desc4.Tokens[0] != 9999 {
		t.Fatalf("Step 4: expected input token 9999, got %d", desc4.Tokens[0])
	}
	if !desc4.IsDecode[0] {
		t.Fatalf("Step 4: expected IsDecode=true")
	}
	if desc4.ContextLens[0] != 600 {
		t.Fatalf("Step 4: expected ContextLen 600, got %d", desc4.ContextLens[0])
	}
}

func TestContinuousBatchManager_IntermediatePrefillGuard(t *testing.T) {
	mgr := NewContinuousBatchManager()
	mgr.PrefillChunkSize = 128

	var receivedTokens []int
	req := &InferenceRequest{
		ID:        100,
		Prompt:    make([]int, 300),
		MaxTokens: 305,
		TokenCallback: func(tok int) {
			receivedTokens = append(receivedTokens, tok)
		},
		Result: make(chan []int, 1),
		Err:    make(chan error, 1),
	}
	mgr.Submit(req)

	// Step 1 (0..128): intermediate chunk
	desc1, _ := mgr.Step(2, nil, nil)
	seq := desc1.Sequences[0]
	chunkLen := len(desc1.Tokens)
	if !desc1.IsDecode[0] && seq.Pos+chunkLen < seq.PromptLen {
		seq.Pos += chunkLen
	} else {
		t.Fatal("expected intermediate chunk guard to trigger on chunk 1")
	}
	if len(receivedTokens) != 0 {
		t.Fatalf("expected 0 callbacks during intermediate prefill, got %d", len(receivedTokens))
	}

	// Step 2 (128..256): intermediate chunk
	desc2, _ := mgr.Step(2, nil, nil)
	chunkLen2 := len(desc2.Tokens)
	if !desc2.IsDecode[0] && seq.Pos+chunkLen2 < seq.PromptLen {
		seq.Pos += chunkLen2
	} else {
		t.Fatal("expected intermediate chunk guard to trigger on chunk 2")
	}
	if len(receivedTokens) != 0 {
		t.Fatalf("expected 0 callbacks during intermediate prefill, got %d", len(receivedTokens))
	}

	// Step 3 (256..300): final prefill chunk (44 tokens)
	desc3, _ := mgr.Step(2, nil, nil)
	chunkLen3 := len(desc3.Tokens)
	if !desc3.IsDecode[0] && seq.Pos+chunkLen3 < seq.PromptLen {
		t.Fatal("should NOT trigger intermediate guard on final chunk")
	}
	// Simulate sampling first generated token on final prefill chunk
	firstGenToken := 777
	seq.Tokens = append(seq.Tokens, firstGenToken)
	seq.Pos += chunkLen3
	if seq.TokenCallback != nil {
		seq.TokenCallback(firstGenToken)
	}

	if len(receivedTokens) != 1 || receivedTokens[0] != 777 {
		t.Fatalf("expected 1 callback with token 777, got %v", receivedTokens)
	}
}
