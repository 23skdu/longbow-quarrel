//go:build !cuda && !metal

package engine

import (
	"testing"
	"time"
)

// TestPagedKVCacheSize checks the reported capacity.
//
// The capacity is the number of blocks times the block size, so it is
// rounded up to a whole number of blocks.
func TestPagedKVCacheSize(t *testing.T) {
	cache, ctx := newTestPagedCache(t, 256)
	defer cache.Free()
	defer ctx.Free()

	size := cache.Size()
	if size <= 0 {
		t.Fatalf("expected a positive capacity, got %d", size)
	}
	if size%cache.blockSize != 0 {
		t.Errorf("the capacity %d is not a multiple of the block size %d", size, cache.blockSize)
	}
	if size < 256 {
		t.Errorf("expected a capacity of at least the requested 256, got %d", size)
	}
}

// TestPagedKVCacheForkSequence covers forking a sequence, which is how a
// beam-search branch or a speculative branch gets its own view of the KV
// cache.
//
// The fork must share the physical blocks and take a reference on each of
// them, so that freeing one branch does not free the blocks the other still
// uses.
func TestPagedKVCacheForkSequence(t *testing.T) {
	cache, ctx := newTestPagedCache(t, 256)
	defer cache.Free()
	defer ctx.Free()

	const src = "src-seq"
	const dst = "dst-seq"

	blocks := []int32{3, 4, 5}
	if err := cache.AttachPrefixBlocks(src, blocks); err != nil {
		t.Fatalf("AttachPrefixBlocks failed: %v", err)
	}

	before := cache.blockRefs[blocks[0]]

	if err := cache.ForkSequence(src, dst); err != nil {
		t.Fatalf("ForkSequence failed: %v", err)
	}

	srcBlocks := cache.GetSequenceBlocks(src)
	dstBlocks := cache.GetSequenceBlocks(dst)
	if len(srcBlocks) != len(blocks) || len(dstBlocks) != len(blocks) {
		t.Fatalf("the fork has %d blocks, the source %d, want %d", len(dstBlocks), len(srcBlocks), len(blocks))
	}
	for i := range blocks {
		if srcBlocks[i] != blocks[i] || dstBlocks[i] != blocks[i] {
			t.Errorf("block %d: source %d, fork %d, want %d", i, srcBlocks[i], dstBlocks[i], blocks[i])
		}
	}

	// The reference count must have been incremented for every shared block.
	for _, b := range blocks {
		if got := cache.blockRefs[b]; got != before+1 {
			t.Errorf("block %d has ref count %d, want %d", b, got, before+1)
		}
	}

	// Forking onto an existing sequence must fail.
	if err := cache.ForkSequence(src, dst); err == nil {
		t.Error("expected an error when forking onto an existing sequence")
	}
	// Forking from an unknown sequence must fail.
	if err := cache.ForkSequence("does-not-exist", "another"); err == nil {
		t.Error("expected an error when forking from an unknown sequence")
	}
}

// TestPagedKVCacheForkSequence_IndependentRelease checks that the two
// sequences really are independent: removing one must not disturb the
// other.
func TestPagedKVCacheForkSequence_IndependentRelease(t *testing.T) {
	cache, ctx := newTestPagedCache(t, 256)
	defer cache.Free()
	defer ctx.Free()

	const src = "src-seq"
	const dst = "dst-seq"

	blocks := []int32{7, 8}
	if err := cache.AttachPrefixBlocks(src, blocks); err != nil {
		t.Fatalf("AttachPrefixBlocks failed: %v", err)
	}
	if err := cache.ForkSequence(src, dst); err != nil {
		t.Fatalf("ForkSequence failed: %v", err)
	}

	// Drop the source sequence.
	cache.FreeSequence(src)

	if got := cache.GetSequenceBlocks(dst); len(got) != len(blocks) {
		t.Errorf("the fork has %d blocks after the source was removed, want %d", len(got), len(blocks))
	}
}

// TestPagedKVCacheGetScaleTensors covers the per-layer scale lookup used
// by the quantized (FP8 and Q8_0) KV cache.
func TestPagedKVCacheGetScaleTensors(t *testing.T) {
	cache, ctx := newTestPagedCache(t, 256)
	defer cache.Free()
	defer ctx.Free()

	// The cache initialises one scale per pool slot, defaulting to 1.0.
	k, v := cache.GetScaleTensors(0)
	if k == nil || v == nil {
		t.Fatalf("expected scale tensors for layer 0")
	}
	if k.Rows()*k.Cols() != v.Rows()*v.Cols() {
		t.Errorf("the key and value scales differ in size: %d vs %d",
			k.Rows()*k.Cols(), v.Rows()*v.Cols())
	}

	host := k.ToHostF32()
	if len(host) == 0 {
		t.Fatal("expected the scale tensor to hold data")
	}
	for i := range host {
		if host[i] != 1.0 {
			t.Errorf("scale %d = %v, want the unit default", i, host[i])
			break
		}
	}

	// An out-of-range layer must also report none.
	k, v = cache.GetScaleTensors(9999)
	if k != nil || v != nil {
		t.Errorf("expected no scales for an out-of-range layer")
	}
	k, v = cache.GetScaleTensors(-1)
	if k != nil || v != nil {
		t.Errorf("expected no scales for a negative layer")
	}
}

// TestPagedKVCacheOffloadLRU covers the memory-pressure path: the least
// recently used sequence is offloaded to host memory so the device blocks can
// be reused.
func TestPagedKVCacheOffloadLRU(t *testing.T) {
	cache, ctx := newTestPagedCache(t, 256)
	defer cache.Free()
	defer ctx.Free()

	// With no sequences there is nothing to offload.
	seq, err := cache.OffloadLRU()
	if err != nil {
		t.Fatalf("OffloadLRU on an empty cache failed: %v", err)
	}
	if seq != "" {
		t.Errorf("expected no sequence to be offloaded from an empty cache, got %q", seq)
	}

	// With a sequence present, it must be offloaded and reported.
	const only = "only-seq"
	if err := cache.AttachPrefixBlocks(only, []int32{1, 2}); err != nil {
		t.Fatalf("AttachPrefixBlocks failed: %v", err)
	}

	// The blocks are released to the free pool by the offload itself, not
	// by the later FreeSequence.
	freeBefore := len(cache.freeBlocks)

	seq, err = cache.OffloadLRU()
	if err != nil {
		t.Fatalf("OffloadLRU failed: %v", err)
	}
	if seq != only {
		t.Errorf("OffloadLRU offloaded %q, want %q", seq, only)
	}
	if !cache.IsOffloaded(only) {
		t.Error("the offloaded sequence is not marked as offloaded")
	}
	if len(cache.freeBlocks) <= freeBefore {
		t.Errorf("the physical blocks were not returned to the free pool by the offload: %d -> %d",
			freeBefore, len(cache.freeBlocks))
	}

	// Releasing the offloaded sequence must not double-free the blocks.
	freeAfter := len(cache.freeBlocks)
	cache.FreeSequence(only)
	if len(cache.freeBlocks) != freeAfter {
		t.Errorf("FreeSequence changed the free pool from %d to %d; the blocks were already released by the offload",
			freeAfter, len(cache.freeBlocks))
	}
}

// TestPagedKVCacheOffloadLRU_PicksTheOldest checks that the least recently
// used sequence is the one that is offloaded.
func TestPagedKVCacheOffloadLRU_PicksTheOldest(t *testing.T) {
	cache, ctx := newTestPagedCache(t, 256)
	defer cache.Free()
	defer ctx.Free()

	const older = "older-seq"
	const newer = "newer-seq"

	if err := cache.AttachPrefixBlocks(older, []int32{1}); err != nil {
		t.Fatalf("AttachPrefixBlocks failed: %v", err)
	}
	// The access times are set by the cache itself; make the ordering
	// explicit.
	cache.mu.Lock()
	cache.lastAccess[older] = testTimePast()
	cache.lastAccess[newer] = testTimeNow()
	cache.mu.Unlock()

	seq, err := cache.OffloadLRU()
	if err != nil {
		t.Fatalf("OffloadLRU failed: %v", err)
	}
	if seq != older {
		t.Errorf("OffloadLRU offloaded %q, want the least recently used %q", seq, older)
	}
}

// testTimePast and testTimeNow give the LRU test explicit access times.
func testTimePast() time.Time { return time.Now().Add(-time.Hour) }
func testTimeNow() time.Time  { return time.Now() }
