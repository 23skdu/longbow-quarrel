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
	//
	// A partial match must NOT be reported as a cache hit unless the
	// matched node actually carries blocks. The blocks are stored on the
	// last node of an inserted prompt, so the intermediate nodes have none.
	// Reporting "3 tokens matched" with an empty block list would make
	// continuous_batching.go advance the sequence position with no KV state
	// attached.
	partial := []int{1, 2, 3, 99}
	matched, cached = cache.MatchPrefix(partial)
	if matched != 0 {
		t.Errorf("Expected no cache hit for a partial match with no blocks, got %d", matched)
	}
	if len(cached) != 0 {
		t.Errorf("Expected no cached blocks for a partial match, got %d", len(cached))
	}

	// 4. A shorter prompt that was inserted in its own right is a hit.
	cache.Insert([]int{1, 2, 3}, []int32{200})
	matched, cached = cache.MatchPrefix([]int{1, 2, 3})
	if matched != 3 {
		t.Errorf("Expected a 3 token match, got %d", matched)
	}
	if len(cached) != 1 || cached[0] != 200 {
		t.Errorf("Expected the cached block 200, got %v", cached)
	}
}
