//go:build metal

package engine

import (
	"fmt"
	"testing"
	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
)

func TestPromptCache_Integration(t *testing.T) {
	ctx := device.NewContext()
	defer ctx.Free()

	// 1. Setup Engine with PromptCache
	eng := &metalEngine{
		ctx:          ctx,
		config:       config.Config{Layers: 2, KVHeads: 8, HeadDim: 64, SeqLen: 1024},
		BatchManager: NewContinuousBatchManager(),
		PromptCache:  NewPromptCache(),
		cache:        &PagedKVCache{},
		stopChan:     make(chan struct{}),
		doneChan:     make(chan struct{}),
	}
	_ = eng.cache.Init(ctx, eng.config)
	defer eng.cache.Free()

	// 2. Submit FIRST request
	prompt := []int{1, 2, 3, 4, 5}
	req1 := &InferenceRequest{
		ID:        1,
		Prompt:    prompt,
		MaxTokens: 10,
		Result:    make(chan []int, 1),
	}
	eng.BatchManager.Submit(req1)

	// Step 1: Prefill (First request)
	active, _ := eng.BatchManager.Step(16, eng.cache, eng.PromptCache)
	if active == nil || len(active.Sequences) != 1 {
		t.Fatalf("expected 1 active sequence, got %v", active)
	}
	
	// Simulate engine processing and marking completed
	seq := active.Sequences[0]
	seq.Pos = seq.PromptLen
	seq.PrefillCompleted = true

	// runBatchLoop logic: insert into cache
	blocks := eng.cache.GetSequenceBlocks(fmt.Sprintf("seq-%d", seq.ID))
	eng.PromptCache.Insert(seq.Tokens[:seq.PromptLen], blocks)

	// 3. Submit SECOND request with SAME prefix
	req2 := &InferenceRequest{
		ID:        2,
		Prompt:    append(prompt, 6, 7), // Shared prefix [1,2,3,4,5]
		MaxTokens: 15,
		Result:    make(chan []int, 1),
	}
	eng.BatchManager.Submit(req2)

	// Step 2: Prefill (Second request) - SHOULD MATCH CACHE
	active2, _ := eng.BatchManager.Step(16, eng.cache, eng.PromptCache)
	if active2 == nil || len(active2.Sequences) != 2 { // Request 1 (decoding) + Request 2 (prefill)
		t.Fatalf("expected 2 active sequences, got %v", active2)
	}

	for _, s := range active2.Sequences {
		if s.ID == 2 {
			if s.Pos != 5 {
				t.Errorf("expected sequence 2 to start at pos 5 (cached), got %d", s.Pos)
			}
		}
	}
}
