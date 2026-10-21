//go:build !cuda && !metal

package engine

import (
	"strings"
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

func TestPagedKVCache_QuantizedTypesNeedTheGPU(t *testing.T) {
	ctx := device.NewContext()
	defer ctx.Free()

	// The quantized pools store one byte per element and the quantized
	// attention kernel is a CUDA kernel. The CPU reference attention reads
	// the pool as float32, so asking for a quantized cache on the CPU has
	// to fail loudly rather than reinterpret the bytes.
	for _, kt := range []conf.KVCacheType{conf.KVCacheFP8, conf.KVCacheQ8_0} {
		cache := &PagedKVCache{}
		cfg := conf.Config{
			KVHeads:     2,
			HeadDim:     32,
			Layers:      2,
			WindowSize:  512,
			KVCacheType: kt,
		}

		err := cache.Init(ctx, cfg)
		if err == nil {
			cache.Free()
			t.Fatalf("expected the %s KV cache to be rejected on the CPU backend", kt)
		}
		if !strings.Contains(err.Error(), "CUDA") {
			t.Errorf("expected the error to name the CUDA backend, got: %v", err)
		}
	}
}

func TestPagedKVCache_KVCacheTypeReachesThePrecision(t *testing.T) {
	ctx := device.NewContext()
	defer ctx.Free()

	// The precision of the pools is not a separate setting: it is the
	// configured KV cache type, so -kv-cache-type reaches the allocation.
	cases := []struct {
		kt   conf.KVCacheType
		want device.DataType
	}{
		{conf.KVCacheF16, device.DataTypeF16},
		{conf.KVCacheTQ1_0, device.DataTypeTQ1_0},
		{conf.KVCacheTQ2_0, device.DataTypeTQ2_0},
	}

	for _, tc := range cases {
		cache := &PagedKVCache{}
		cfg := conf.Config{
			KVHeads:     2,
			HeadDim:     32,
			Layers:      1,
			WindowSize:  128,
			KVCacheType: tc.kt,
		}
		if err := cache.Init(ctx, cfg); err != nil {
			t.Fatalf("Init failed for %v: %v", tc.kt, err)
		}
		if cache.Precision != tc.want {
			t.Errorf("precision for %v = %v, want %v", tc.kt, cache.Precision, tc.want)
		}
		cache.Free()
	}
}
