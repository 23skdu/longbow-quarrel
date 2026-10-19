package engine

import (
	"testing"
)

// TestPromptCacheEvict covers the LRU eviction, which frees the physical KV
// blocks of the least recently used cached prompts back to the pool.
//
// Evict and removeLRU were at 0% coverage: the cache was only ever written
// to, so nothing exercised the path that keeps the memory bounded.
func TestPromptCacheEvict(t *testing.T) {
	pc := NewPromptCache()
	pc.maxCachedBlocks = 4

	// Nothing to evict.
	if freed := pc.Evict(nil); freed != 0 {
		t.Errorf("expected nothing to be freed from an empty cache, got %d", freed)
	}

	// Insert three prompts of two blocks each: six blocks, over the limit of
	// four.
	pc.Insert([]int{1, 2}, []int32{0, 1})
	pc.Insert([]int{3, 4}, []int32{2, 3})
	pc.Insert([]int{5, 6}, []int32{4, 5})

	if pc.lruSize != 6 {
		t.Fatalf("expected 6 cached blocks, got %d", pc.lruSize)
	}

	// Evicting must bring the cache back under the limit. The freed counter
	// only counts blocks that were returned to a KV cache, so it stays zero
	// when no cache is supplied.
	freed := pc.Evict(nil)
	if freed != 0 {
		t.Errorf("expected no blocks to be counted without a KV cache, got %d", freed)
	}
	if pc.lruSize > pc.maxCachedBlocks {
		t.Errorf("the cache is still over its limit: %d > %d", pc.lruSize, pc.maxCachedBlocks)
	}
	if pc.lruSize != 4 {
		t.Errorf("expected 4 cached blocks to remain, got %d", pc.lruSize)
	}
}

// TestPromptCacheEvict_FreesBlocks checks that the evicted blocks are
// returned to the KV cache's free pool.
func TestPromptCacheEvict_FreesBlocks(t *testing.T) {
	cache, ctx := newTestPagedCache(t, 256)
	defer cache.Free()
	defer ctx.Free()

	pc := NewPromptCache()
	pc.maxCachedBlocks = 1

	// Use two real blocks.
	blocks := []int32{0, 1}
	if err := cache.AttachPrefixBlocks("cached", blocks); err != nil {
		t.Fatalf("AttachPrefixBlocks failed: %v", err)
	}
	freeBefore := len(cache.freeBlocks)

	pc.Insert([]int{1, 2}, blocks)
	pc.Insert([]int{3, 4}, []int32{2})

	// The least recently used entry holds two blocks, so those are the ones
	// returned to the pool.
	freed := pc.Evict(cache)
	if freed != 2 {
		t.Errorf("expected 2 blocks to be freed, got %d", freed)
	}
	if len(cache.freeBlocks) != freeBefore+2 {
		t.Errorf("the free pool did not grow by 2: %d -> %d", freeBefore, len(cache.freeBlocks))
	}
	if pc.lruSize > pc.maxCachedBlocks {
		t.Errorf("the cache is still over its limit: %d > %d", pc.lruSize, pc.maxCachedBlocks)
	}
}

// TestPromptCacheReinsertDoesNotDoubleCount is the regression guard for the
// LRU accounting.
//
// Re-inserting a prompt that is already cached moves its entry to the head of
// the LRU list. The move used to add the entry's block count to lruSize a
// second time, so the cache believed it was holding twice as many blocks as
// it really was and evicted far too aggressively.
func TestPromptCacheReinsertDoesNotDoubleCount(t *testing.T) {
	pc := NewPromptCache()

	pc.Insert([]int{1, 2}, []int32{0, 1})
	if pc.lruSize != 2 {
		t.Fatalf("expected 2 cached blocks after the first insert, got %d", pc.lruSize)
	}

	// The same prompt again: the entry is moved to the head, but it is
	// already accounted for.
	pc.Insert([]int{1, 2}, []int32{0, 1})
	if pc.lruSize != 2 {
		t.Errorf("expected 2 cached blocks after re-inserting, got %d", pc.lruSize)
	}

	// A different prompt adds its own blocks.
	pc.Insert([]int{3, 4}, []int32{2, 3})
	if pc.lruSize != 4 {
		t.Errorf("expected 4 cached blocks, got %d", pc.lruSize)
	}
}

// TestPromptCacheRemoveLRU checks the LRU list bookkeeping directly.
func TestPromptCacheRemoveLRU(t *testing.T) {
	pc := NewPromptCache()

	// Removing from an empty list is a no-op.
	pc.removeLRU()
	if pc.lruHead != nil || pc.lruTail != nil || pc.lruSize != 0 {
		t.Error("removing from an empty LRU list changed its state")
	}

	pc.Insert([]int{1}, []int32{0})
	pc.Insert([]int{2}, []int32{1})
	if pc.lruSize != 2 {
		t.Fatalf("expected 2 cached blocks, got %d", pc.lruSize)
	}

	// The most recently inserted entry is at the head, the oldest at the
	// tail.
	if pc.lruHead == nil || pc.lruTail == nil {
		t.Fatal("the LRU list is empty")
	}
	if pc.lruHead == pc.lruTail {
		t.Error("the LRU list has a single entry")
	}

	pc.removeLRU()
	if pc.lruSize != 1 {
		t.Errorf("expected 1 cached block after the removal, got %d", pc.lruSize)
	}
	// One entry is left, and it is both the head and the tail.
	if pc.lruHead == nil || pc.lruTail == nil {
		t.Error("the remaining entry was dropped from the LRU list")
	}
	if pc.lruHead != pc.lruTail {
		t.Error("the remaining entry is not both the head and the tail")
	}
}
