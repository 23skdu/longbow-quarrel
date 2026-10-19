package engine

import (
	"testing"
)

// TestPromptCacheMatchPrefix_PartialMatchIsNotACacheHit is a regression
// guard for a prefix-cache correctness bug.
//
// The cache stores the physical block list on the *last* node of an inserted
// prompt, and the intermediate nodes hold no blocks. MatchPrefix used to
// return the depth of the deepest matched node together with that node's
// blocks, so a partial match reported "N tokens matched" with an *empty*
// block list.
//
// The caller in continuous_batching.go does:
//
//	if matchedCount > 0 {
//	    kvCache.AttachPrefixBlocks(seqIDStr, cachedBlocks)
//	    seq.Pos = matchedCount
//	}
//
// so the sequence would start decoding at position N with no KV blocks
// attached for those N tokens: the model would generate from a prompt whose
// KV state was never computed.
func TestPromptCacheMatchPrefix_PartialMatchIsNotACacheHit(t *testing.T) {
	pc := NewPromptCache()

	// Insert a three-token prompt holding three physical blocks.
	pc.Insert([]int{10, 20, 30}, []int32{1, 2, 3})

	t.Run("full match returns the blocks", func(t *testing.T) {
		matched, blocks := pc.MatchPrefix([]int{10, 20, 30})
		if matched != 3 {
			t.Fatalf("expected 3 matched tokens, got %d", matched)
		}
		if len(blocks) != 3 {
			t.Fatalf("expected 3 cached blocks, got %d", len(blocks))
		}
		for i, want := range []int32{1, 2, 3} {
			if blocks[i] != want {
				t.Errorf("block %d = %d, want %d", i, blocks[i], want)
			}
		}
	})

	t.Run("partial match reports no cache hit", func(t *testing.T) {
		// A prefix of the inserted prompt. The intermediate nodes hold no
		// blocks, so this must NOT be reported as a hit.
		matched, blocks := pc.MatchPrefix([]int{10, 20})
		if matched > 0 {
			t.Errorf("a partial match reported %d matched tokens; the caller "+
				"would advance seq.Pos without any KV blocks attached", matched)
		}
		if len(blocks) != 0 {
			t.Errorf("a partial match returned %d blocks, want 0", len(blocks))
		}
	})

	t.Run("single token prefix is not a hit", func(t *testing.T) {
		matched, blocks := pc.MatchPrefix([]int{10})
		if matched > 0 || len(blocks) != 0 {
			t.Errorf("a single-token prefix reported a hit (matched=%d blocks=%d)", matched, len(blocks))
		}
	})

	t.Run("no match", func(t *testing.T) {
		matched, blocks := pc.MatchPrefix([]int{99})
		if matched != 0 || len(blocks) != 0 {
			t.Errorf("expected no match, got matched=%d blocks=%d", matched, len(blocks))
		}
	})
}

// TestPromptCacheMatchPrefix_SharedPrefix checks that a second sequence
// sharing a prefix with the first gets the cached blocks.
func TestPromptCacheMatchPrefix_SharedPrefix(t *testing.T) {
	pc := NewPromptCache()

	// A long shared system prompt.
	system := []int{1, 2, 3, 4, 5, 6, 7, 8}
	pc.Insert(system, []int32{10, 11, 12, 13})

	// A request that shares the whole system prompt.
	matched, blocks := pc.MatchPrefix(system)
	if matched != len(system) {
		t.Fatalf("expected %d matched tokens, got %d", len(system), matched)
	}
	if len(blocks) != 4 {
		t.Fatalf("expected 4 cached blocks, got %d", len(blocks))
	}

	// A request that shares only part of it: the longest prefix that has
	// blocks cached is the whole system prompt, so a shorter request must
	// not be reported as a hit.
	matched, blocks = pc.MatchPrefix(system[:4])
	if matched > 0 || len(blocks) != 0 {
		t.Errorf("a partial prefix reported a hit (matched=%d blocks=%d)", matched, len(blocks))
	}
}
