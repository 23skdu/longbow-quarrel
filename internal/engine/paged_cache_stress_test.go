package engine

import (
	"fmt"
	"testing"
	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/config"
)

func TestPagedKVCache_AllocationStress(t *testing.T) {
	ctx := device.NewContext()
	cfg := config.Config{
		KVHeads: 8,
		HeadDim: 64,
		Layers: 2,
	}
	
	cache := &PagedKVCache{}
	// Small cache to force reuse/fragmentation tests
	cache.Init(ctx, cfg) 
	
	// 1. Allocate many sequences
	for i := 0; i < 20; i++ {
		seqID := fmt.Sprintf("seq-%d", i)
		err := cache.Allocate(seqID, 100) // 100 tokens
		if err != nil {
			t.Errorf("Failed to allocate for seq %d: %v", i, err)
		}
	}
	
	// 2. Free some
	for i := 0; i < 10; i++ {
		seqID := fmt.Sprintf("seq-%d", i)
		cache.FreeSequence(seqID)
	}
	
	// 3. Re-allocate larger
	for i := 10; i < 20; i++ {
		seqID := fmt.Sprintf("seq-%d", i)
		// This should grow existing tables
		err := cache.Allocate(seqID, 200)
		if err != nil {
			t.Errorf("Failed to grow seq %d: %v", i, err)
		}
	}
	
	// 4. Test mapping
	seqID := "seq-15"
	phys, err := cache.GetPhysicalPositions(seqID, 50, 10)
	if err != nil {
		t.Errorf("GetPhysicalPositions failed: %v", err)
	}
	if len(phys) != 10 {
		t.Errorf("Expected 10 positions, got %d", len(phys))
	}
	
	// 5. Test Block Table Device Sync
	view := cache.Get(seqID, 0)
	if view.BlockTable == nil {
		t.Error("Expected BlockTable device tensor")
	}
	
	// 6. Test Batch View
	batchIDs := []string{"seq-10", "seq-11", "seq-15"}
	batchView := cache.GetBatch(batchIDs, []int{0, 0, 0}, 0)
	if batchView.BlockTables == nil {
		t.Error("Expected Batched BlockTables")
	}
	if batchView.MaxBlocks == 0 {
		t.Error("MaxBlocks should not be zero")
	}
}

func TestPagedKVCache_Free(t *testing.T) {
	ctx := device.NewContext()
	cache := &PagedKVCache{}
	cache.Init(ctx, config.Config{Layers: 1, KVHeads: 1, HeadDim: 1})
	
	cache.Allocate("test", 10)
	cache.Free()
	
	if cache.initialized {
		t.Error("Cache should not be initialized after Free")
	}
}
