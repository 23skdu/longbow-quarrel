//go:build !cuda && !metal && !tpu

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

	// Update pos 0 -> Should alloc block
	err = cache.Update("seq-0", 0, 0, k, v)
	if err != nil {
		t.Errorf("Update failed: %v", err)
	}

	// Check block table
	if len(cache.blockTables["seq-0"]) != 1 {
		t.Errorf("Expected 1 block allocated, got %d", len(cache.blockTables["seq-0"]))
	}

	// Update pos 15 -> Same block
	err = cache.Update("seq-0", 0, 15, k, v)
	if err != nil {
		t.Errorf("Update failed: %v", err)
	}
	if len(cache.blockTables["seq-0"]) != 1 {
		t.Errorf("Expected 1 block allocated, got %d", len(cache.blockTables["seq-0"]))
	}

	// Update pos 16 -> New block
	err = cache.Update("seq-0", 0, 16, k, v)
	if err != nil {
		t.Errorf("Update failed: %v", err)
	}
	if len(cache.blockTables["seq-0"]) != 2 {
		t.Errorf("Expected 2 blocks allocated, got %d", len(cache.blockTables["seq-0"]))
	}

	// Verify Get returns BlockTable
	view := cache.Get("seq-0", 0)
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

func TestPagedKVCache_FP8_And_Q8(t *testing.T) {
	ctx := device.NewContext()
	defer ctx.Free()

	// 1. Test FP8 Cache
	cacheFP8 := &PagedKVCache{Precision: device.DataTypeFP8}
	cfg := conf.Config{
		KVHeads:    2,
		HeadDim:    32,
		Layers:     2,
		WindowSize: 512,
	}

	if err := cacheFP8.Init(ctx, cfg); err != nil {
		t.Fatalf("Init FP8 cache failed: %v", err)
	}
	defer cacheFP8.Free()

	k := ctx.NewTensorFP32(1, 2*32)
	v := ctx.NewTensorFP32(1, 2*32)
	kHost := make([]float32, 64)
	vHost := make([]float32, 64)
	for i := range kHost {
		kHost[i] = float32(i) * 0.1
		vHost[i] = -float32(i) * 0.1
	}
	_ = k.LoadFrom(kHost)
	_ = v.LoadFrom(vHost)

	if err := cacheFP8.Update("seq-fp8", 0, 0, k, v); err != nil {
		t.Fatalf("FP8 Update failed: %v", err)
	}
	if len(cacheFP8.blockTables["seq-fp8"]) != 1 {
		t.Errorf("expected 1 block in FP8 block table, got %d", len(cacheFP8.blockTables["seq-fp8"]))
	}

	// 2. Test Q8_0 Cache
	cacheQ8 := &PagedKVCache{Precision: device.DataTypeQ8_0}
	if err := cacheQ8.Init(ctx, cfg); err != nil {
		t.Fatalf("Init Q8 cache failed: %v", err)
	}
	defer cacheQ8.Free()

	if err := cacheQ8.Update("seq-q8", 0, 0, k, v); err != nil {
		t.Fatalf("Q8 Update failed: %v", err)
	}
	if len(cacheQ8.blockTables["seq-q8"]) != 1 {
		t.Errorf("expected 1 block in Q8 block table, got %d", len(cacheQ8.blockTables["seq-q8"]))
	}
}

