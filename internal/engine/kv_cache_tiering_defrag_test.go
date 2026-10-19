package engine

import (
	"math"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
)

func TestPagedKVCache_OffloadAndRestoreRoundTrip(t *testing.T) {
	ctx := device.NewContext()
	defer ctx.Free()

	cfg := config.Config{
		Layers:      2,
		KVHeads:     2,
		HeadDim:     8,
		KVCacheSize: 64, // 64 tokens = 4 blocks of size 16
		SeqLen:      64,
	}

	cache := &PagedKVCache{}
	if err := cache.Init(ctx, cfg); err != nil {
		t.Fatalf("Init failed: %v", err)
	}
	defer cache.Free()

	// Allocate blocks for a sequence (32 tokens = 2 blocks)
	seqID := "test-seq-1"
	if err := cache.Allocate(seqID, 32); err != nil {
		t.Fatalf("Allocate failed: %v", err)
	}

	// Populate K and V pools with deterministic data
	kData := cache.kPools[0].ToHostF32()
	cols := cache.kPools[0].Cols()
	for i := 0; i < 32*cols; i++ {
		kData[i] = float32(i) * 0.5
	}
	_ = cache.kPools[0].LoadFrom(kData)

	origFree := cache.FreeBlocksCount()
	if origFree != 2 {
		t.Fatalf("Expected 2 free blocks, got %d", origFree)
	}

	// 1. Offload sequence
	if err := cache.OffloadSequence(seqID); err != nil {
		t.Fatalf("OffloadSequence failed: %v", err)
	}

	if !cache.IsOffloaded(seqID) {
		t.Fatalf("Expected sequence %s to be offloaded", seqID)
	}

	// Free blocks should now be back to 4
	if cache.FreeBlocksCount() != 4 {
		t.Fatalf("Expected 4 free blocks after offload, got %d", cache.FreeBlocksCount())
	}

	// 2. Restore sequence
	if err := cache.RestoreSequence(seqID); err != nil {
		t.Fatalf("RestoreSequence failed: %v", err)
	}

	if cache.IsOffloaded(seqID) {
		t.Fatalf("Expected sequence %s to not be offloaded after restore", seqID)
	}

	if cache.FreeBlocksCount() != 2 {
		t.Fatalf("Expected 2 free blocks after restore, got %d", cache.FreeBlocksCount())
	}

	// Verify restored data integrity
	restoredK := cache.kPools[0].ToHostF32()
	table := cache.blockTables[seqID]
	for bIdx, block := range table {
		dstOffset := int(block) * cache.blockSize * cols
		for row := 0; row < cache.blockSize; row++ {
			for col := 0; col < cols; col++ {
				expectedVal := float32((bIdx*cache.blockSize+row)*cols+col) * 0.5
				actualVal := restoredK[dstOffset+row*cols+col]
				if math.Abs(float64(actualVal-expectedVal)) > 1e-4 {
					t.Fatalf("Data mismatch at block %d, row %d, col %d: expected %f, got %f", block, row, col, expectedVal, actualVal)
				}
			}
		}
	}
}

func TestPagedKVCache_Defragmentation(t *testing.T) {
	ctx := device.NewContext()
	defer ctx.Free()

	cfg := config.Config{
		Layers:      1,
		KVHeads:     2,
		HeadDim:     4,
		KVCacheSize: 64, // 4 blocks: 0, 1, 2, 3
		SeqLen:      64,
	}

	cache := &PagedKVCache{}
	if err := cache.Init(ctx, cfg); err != nil {
		t.Fatalf("Init failed: %v", err)
	}
	defer cache.Free()

	// Allocate 4 sequences of 16 tokens each (1 block each: blocks 3, 2, 1, 0)
	_ = cache.Allocate("seq-A", 16)
	_ = cache.Allocate("seq-B", 16)
	_ = cache.Allocate("seq-C", 16)
	_ = cache.Allocate("seq-D", 16)

	// Free seq-A and seq-B to create holes at higher indices
	cache.FreeSequence("seq-A")
	cache.FreeSequence("seq-B")

	// Run defragmentation
	compacted, err := cache.Defragment()
	if err != nil {
		t.Fatalf("Defragment failed: %v", err)
	}

	// Verify compacting moved high index blocks to low index blocks
	if compacted < 0 {
		t.Fatalf("Expected non-negative compacted blocks, got %d", compacted)
	}

	// All remaining sequences should have valid non-overlapping blocks
	for seqID, table := range cache.blockTables {
		if len(table) == 0 {
			t.Fatalf("Empty table for %s", seqID)
		}
		for _, b := range table {
			if b < 0 || int(b) >= cache.totalBlocks {
				t.Fatalf("Invalid block %d in table for %s", b, seqID)
			}
		}
	}
}

func TestContinuousBatchManager_MemoryPressureAndTieringStress(t *testing.T) {
	ctx := device.NewContext()
	defer ctx.Free()

	cfg := config.Config{
		Layers:      1,
		KVHeads:     2,
		HeadDim:     4,
		KVCacheSize: 80, // 5 blocks total (capacity for only a few sequences)
		SeqLen:      80,
	}

	cache := &PagedKVCache{}
	if err := cache.Init(ctx, cfg); err != nil {
		t.Fatalf("Init failed: %v", err)
	}
	defer cache.Free()

	gov := device.GetMemoryGovernor()
	cache.RegisterMemoryGovernor(gov)

	mgr := NewContinuousBatchManager()
	mgr.LowWaterMark = 1 // Aggressive threshold

	// Enqueue 15 requests under very tight KV cache capacity (stress test under >95% capacity)
	completed := 0
	for i := 0; i < 15; i++ {
		req := &InferenceRequest{
			ID:        uint64(i + 1),
			Prompt:    []int{1, 2, 3, 4},
			MaxTokens: 5,
			Priority:  i % 10,
			Result:    make(chan []int, 1),
			Err:       make(chan error, 1),
		}
		mgr.Submit(req)
	}

	// Simulate batch iterations with preemption, tiering, and defragmentation
	for step := 0; step < 60; step++ {
		desc, err := mgr.Step(4, cache, nil)
		if err != nil {
			t.Fatalf("Step error: %v", err)
		}

		if desc == nil {
			if mgr.Depth() == 0 {
				break
			}
			continue
		}

		for _, seq := range desc.Sequences {
			seq.Pos++
			if seq.Pos >= seq.PromptLen+seq.MaxTokens {
				mgr.CompleteSequence(seq.ID, cache)
				completed++
			}
		}

		// Periodically trigger governor defragmentation
		if step%5 == 0 {
			gov.TriggerGovernor(0.50)
		}
	}

	if completed == 0 {
		t.Fatalf("Expected completed requests under heavy KV cache capacity, got 0")
	}
}
