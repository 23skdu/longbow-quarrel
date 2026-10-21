package engine

import (
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/device"
)

// The paged KV cache's allocation, position and rollback paths are the
// bookkeeping the attention kernels depend on, and their error branches
// had little coverage.

// TestPagedKVCacheAllocationAndPositions covers the allocation and the
// physical position lookup.
func TestPagedKVCacheAllocationAndPositions(t *testing.T) {
	cache, ctx := newTestPagedCache(t, 512)
	defer cache.Free()
	defer ctx.Free()

	// Allocating a sequence hands out physical blocks.
	if err := cache.Allocate("seq-a", 40); err != nil {
		t.Fatalf("Allocate failed: %v", err)
	}
	blocks := cache.GetSequenceBlocks("seq-a")
	if len(blocks) == 0 {
		t.Fatal("the sequence was not given any blocks")
	}

	// The physical positions of the first tokens are inside the blocks
	// the sequence was given.
	positions, err := cache.GetPhysicalPositions("seq-a", 0, 10)
	if err != nil {
		t.Fatalf("GetPhysicalPositions failed: %v", err)
	}
	if len(positions) != 10 {
		t.Fatalf("the lookup returned %d positions, want 10", len(positions))
	}
	for i, p := range positions {
		if p < 0 {
			t.Errorf("the position %d is %d, want a non-negative value", i, p)
		}
	}

	// A sequence that was never allocated has no positions.
	if _, err := cache.GetPhysicalPositions("missing", 0, 4); err == nil {
		t.Error("expected an error for a sequence that was never allocated")
	}

	// A range that runs past the end is an error.
	if _, err := cache.GetPhysicalPositions("seq-a", 0, 100000); err == nil {
		t.Error("expected an error for a range that runs past the end")
	}
}

// TestPagedKVCacheRollback covers the rollback, which is what a rejected
// draft token uses.
func TestPagedKVCacheRollback(t *testing.T) {
	cache, ctx := newTestPagedCache(t, 512)
	defer cache.Free()
	defer ctx.Free()

	if err := cache.Allocate("seq-a", 40); err != nil {
		t.Fatalf("Allocate failed: %v", err)
	}

	// The sequence was given enough blocks for 40 tokens, so rolling
	// back to position 8 prunes the blocks beyond the one that holds
	// it.
	before := len(cache.GetSequenceBlocks("seq-a"))
	if before < 2 {
		t.Fatalf("the sequence was given %d blocks, want at least 2", before)
	}

	if err := cache.RollbackKV("seq-a", 8); err != nil {
		t.Errorf("RollbackKV failed: %v", err)
	}
	after := len(cache.GetSequenceBlocks("seq-a"))
	if after >= before {
		t.Errorf("the rollback left %d blocks, want fewer than %d", after, before)
	}

	// Rolling back a sequence that was never allocated is an error.
	if err := cache.RollbackKV("missing", 4); err == nil {
		t.Error("expected an error for a sequence that was never allocated")
	}
}

// TestPagedKVCacheGetAndSize covers the per-sequence view and the size
// report.
func TestPagedKVCacheGetAndSize(t *testing.T) {
	cache, ctx := newTestPagedCache(t, 512)
	defer cache.Free()
	defer ctx.Free()

	if err := cache.Allocate("seq-a", 40); err != nil {
		t.Fatalf("Allocate failed: %v", err)
	}

	view := cache.Get("seq-a", 0)
	if view.K == nil || view.V == nil {
		t.Error("the view has no key or value pool")
	}
	if view.BlockTable == nil {
		t.Error("the view has no block table")
	}

	size := cache.Size()
	if size <= 0 {
		t.Errorf("the cache reports a size of %d", size)
	}
}

// TestPagedKVCacheMemoryGovernor covers the registration of the memory
// governor, which is what triggers the tiering.
func TestPagedKVCacheMemoryGovernor(t *testing.T) {
	cache, ctx := newTestPagedCache(t, 512)
	defer cache.Free()
	defer ctx.Free()

	governor := device.GetMemoryGovernor()
	cache.RegisterMemoryGovernor(governor)
}

// TestWeightStructsFree covers the cleanup helpers of the weight structs,
// which are the ones that hold device tensors.
func TestWeightStructsFree(t *testing.T) {
	ctx := device.NewContext()
	defer ctx.Free()

	mamba := &MambaWeights{
		A:            ctx.NewTensorWithType(2, 2, device.DataTypeF32),
		D:            ctx.NewTensorWithType(2, 2, device.DataTypeF32),
		Conv1dWeight: ctx.NewTensorWithType(2, 2, device.DataTypeF32),
		OutWeight:    ctx.NewTensorWithType(2, 2, device.DataTypeF32),
		InWeight:     ctx.NewTensorWithType(2, 2, device.DataTypeF32),
	}
	mamba.Free()
	// The cleanup releases the device memory. A second call must not
	// panic, and neither must a call on a struct whose fields were
	// never set.
	assertNoPanic(t, "the second Mamba cleanup", mamba.Free)
	assertNoPanic(t, "the cleanup of an empty Mamba", func() {
		(&MambaWeights{}).Free()
	})

	state := &MambaState{
		ConvState: ctx.NewTensorWithType(2, 2, device.DataTypeF32),
		SSMState:  ctx.NewTensorWithType(2, 2, device.DataTypeF32),
	}
	state.Free()
	assertNoPanic(t, "the second Mamba state cleanup", state.Free)
	assertNoPanic(t, "the cleanup of an empty Mamba state", func() {
		(&MambaState{}).Free()
	})

	moe := &MOESharedWeights{
		FfnGateShared: ctx.NewTensorWithType(2, 2, device.DataTypeF32),
		FfnUpShared:   ctx.NewTensorWithType(2, 2, device.DataTypeF32),
		FfnDownShared: ctx.NewTensorWithType(2, 2, device.DataTypeF32),
	}
	moe.Free()
	assertNoPanic(t, "the second shared expert cleanup", moe.Free)
	assertNoPanic(t, "the cleanup of empty shared expert weights", func() {
		(&MOESharedWeights{}).Free()
	})

	router := &MOERouterWeights{
		GateInput:      ctx.NewTensorWithType(2, 2, device.DataTypeF32),
		ExpertProbBias: ctx.NewTensorWithType(2, 2, device.DataTypeF32),
	}
	router.Free()
	assertNoPanic(t, "the second router cleanup", router.Free)
	assertNoPanic(t, "the cleanup of empty router weights", func() {
		(&MOERouterWeights{}).Free()
	})
}

// assertNoPanic fails the test when the function panics.
//
// The cleanup helpers release device memory, and a double release or a
// release of a struct whose fields were never set must not take the
// process down.
func assertNoPanic(t *testing.T, name string, fn func()) {
	t.Helper()
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("%s panicked: %v", name, r)
		}
	}()
	fn()
}
