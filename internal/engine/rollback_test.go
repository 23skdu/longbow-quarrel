//go:build metal

package engine

import (
	"testing"
	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
)

func TestPagedKVCache_Rollback(t *testing.T) {
	ctx := device.NewContext()
	defer ctx.Free()

	conf := config.Config{
		Layers: 1, 
		KVHeads: 1, 
		HeadDim: 64, 
		SeqLen: 128,
		WindowSize: 128,
	}

	cache := &PagedKVCache{}
	err := cache.Init(ctx, conf)
	if err != nil {
		t.Fatalf("failed to init cache: %v", err)
	}
	defer cache.Free()

	seqID := "test-rollback"
	
	// 1. Fill 2 blocks (BlockSize is 16 by default)
	// Block 0: 0-15
	// Block 1: 16-31
	for p := 0; p < 24; p++ {
		k := ctx.NewTensor(1, 64)
		v := ctx.NewTensor(1, 64)
		_ = cache.Update(seqID, 0, p, k, v)
		k.Free()
		v.Free()
	}

	table := cache.blockTables[seqID]
	if len(table) != 2 {
		t.Fatalf("expected 2 blocks, got %d", len(table))
	}
	initialBlock1 := table[1]

	// 2. Rollback to position 10 (inside Block 0)
	err = cache.RollbackKV(seqID, 10)
	if err != nil {
		t.Fatalf("rollback failed: %v", err)
	}

	// 3. Verify Block 1 is freed and table pruned
	newTable := cache.blockTables[seqID]
	if len(newTable) != 1 {
		t.Errorf("expected 1 block after rollback, got %d", len(newTable))
	}
	
	if cache.blockRefs[initialBlock1] != 0 {
		t.Errorf("expected Block 1 refcount 0, got %d", cache.blockRefs[initialBlock1])
	}
}
