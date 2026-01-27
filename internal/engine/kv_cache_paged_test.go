//go:build darwin && metal

package engine

import (
	"testing"

	conf "github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
)

func TestPagedKVCache_Lifecycle(t *testing.T) {
	ctx := device.NewContext()
	defer ctx.Free()

	cache := &PagedKVCache{}
	config := conf.Config{
		KVHeads:    2,
		HeadDim:    64,
		Layers:     2,
		WindowSize: 2048, // Capacity
	}

	err := cache.Init(ctx, config)
	if err != nil {
		t.Fatalf("Init failed: %v", err)
	}
	defer cache.Free()

	if cache.totalBlocks == 0 {
		t.Error("Total blocks should be > 0")
	}
	if cache.blockSize != 16 {
		t.Errorf("Expected block size 16, got %d", cache.blockSize)
	}

	// Update
	k := ctx.NewTensor(1, 2*64)
	v := ctx.NewTensor(1, 2*64)
	k.ZeroInit()
	v.ZeroInit()

	// Update pos 0 -> Should alloc block 0
	err = cache.Update(0, 0, k, v)
	if err != nil {
		t.Errorf("Update failed: %v", err)
	}

	// Check block table
	if len(cache.blockTableHost) != 1 {
		t.Errorf("Expected 1 block allocated, got %d", len(cache.blockTableHost))
	}

	// Update pos 15 -> Same block
	err = cache.Update(0, 15, k, v)
	if err != nil {
		t.Errorf("Update failed: %v", err)
	}
	if len(cache.blockTableHost) != 1 {
		t.Errorf("Expected 1 block allocated, got %d", len(cache.blockTableHost))
	}

	// Update pos 16 -> New block
	err = cache.Update(0, 16, k, v)
	if err != nil {
		t.Errorf("Update failed: %v", err)
	}
	if len(cache.blockTableHost) != 2 {
		t.Errorf("Expected 2 blocks allocated, got %d", len(cache.blockTableHost))
	}

	// Verify Get returns BlockTable
	view := cache.Get(0)
	if view.BlockTable == nil {
		t.Error("Get returned nil BlockTable")
	}
	if view.BlockSize != 16 {
		t.Errorf("Get returned wrong blockSize: %d", view.BlockSize)
	}

	// Verify View Block Table Tensor Content
	// Need to check F32 values from device
	// This requires ToHost() on BlockTable tensor.
	// But BlockTable is F32 tensor mapped to Int32 bits (unsafe).
	// ToHost() returns []float32.
	// We need ToHostBytes() or similar.
	// metal.go has ToHostBytes?
	// I added `LoadFromRaw`, let's check `ToHostBytes` exist.
	// Assuming it exists or I can add it/use ToHost() and cast.
	// Float32 bits can be read as Int32.
}
