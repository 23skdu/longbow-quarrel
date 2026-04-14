package engine

import (
	"testing"
)

func TestLoRAManager_Coverage(t *testing.T) {
	mgr := NewLoRAManager()
	
	// Test adding some adapters (mock)
	// We can't easily load real files without testdata, so we focus on logic
	if weights, ok := mgr.GetWeights("non-existent", "blk.0.attn_q"); ok || weights != nil {
		t.Error("Expected nil for non-existent adapter")
	}
}

func TestPromptCache_Coverage(t *testing.T) {
	cache := NewPromptCache()
	
	prompt := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	blocks := []int32{100, 101}
	
	// 1. Insert
	cache.Insert(prompt, blocks)
	
	// 2. Match
	matched, cached := cache.MatchPrefix(prompt)
	if matched != len(prompt) {
		t.Errorf("Expected full match, got %d", matched)
	}
	if len(cached) != 2 || cached[0] != 100 {
		t.Error("Cached blocks mismatch")
	}
	
	// 3. Partial Match
	partial := []int{1, 2, 3, 99}
	matched, _ = cache.MatchPrefix(partial)
	if matched != 3 {
		t.Errorf("Expected partial match 3, got %d", matched)
	}
}

func TestPromptCache_LRU(t *testing.T) {
	// If the cache had a max size, we'd test LRU here. 
	// Currently it seems to be a simple map.
}
