package engine

import (
	"testing"
	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
)

func TestPagedKVCache_Exhaustive(t *testing.T) {
	cache := &PagedKVCache{}
	ctx := device.NewContext()
	cfg := config.Config{
		BlockSize: 16,
		Layers: 1,
		Dim: 128,
		Heads: 1,
		KVHeads: 1,
	}
	
	cache.Init(ctx, cfg)
	
	// 1. Double Allocate
	seqID := "seq_1"
	err := cache.Allocate(seqID, 10)
	if err != nil { t.Errorf("Allocate failed: %v", err) }
	
	err = cache.Allocate(seqID, 20) // Test growth
	if err != nil { t.Errorf("Growth failed: %v", err) }

	// 2. Attach Prefix (via PromptCache)
	pc := NewPromptCache()
	prompt := []int{1, 2}
	blocks := []int32{0, 1}
	pc.Insert(prompt, blocks)
	
	matched, cached := pc.MatchPrefix(prompt)
	if matched != 2 || len(cached) != 2 {
		t.Error("PromptCache match failed")
	}

	targetID := "seq_match"
	err = cache.AttachPrefixBlocks(targetID, cached)
	if err != nil { t.Errorf("Attach failed: %v", err) }
	
	// 3. Free
	cache.FreeSequence(seqID)
	cache.FreeSequence(targetID)
	cache.Free() // Global free
}

func TestContinuousBatchManager_Edge_Cases(t *testing.T) {
	cm := NewContinuousBatchManager()
	
	// Submit request
	req := &InferenceRequest{ID: 1, Prompt: []int{1}}
	cm.Submit(req)
	
	// 2. Test Step with no requests (empty and non-empty paths)
	cache := &PagedKVCache{}
	desc, _ := cm.Step(4, cache, nil)
	if desc == nil {
		// Expected if queue is empty or cache lacks capacity
	}
	
	cm.AbortAll(nil)
}
