//go:build darwin && metal

package engine

import (
	"testing"

	conf "github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
)

func TestSlidingWindowKVCache_Lifecycle(t *testing.T) {
	ctx := device.NewContext()
	defer ctx.Free()

	cache := &SlidingWindowKVCache{}
	config := conf.Config{
		Layers:     1,
		WindowSize: 10,
		KVHeads:    2,
		HeadDim:    4,
	}

	if err := cache.Init(ctx, config); err != nil {
		t.Fatalf("Init failed: %v", err)
	}
	defer cache.Free()

	if cache.Size() != 10 {
		t.Errorf("Expected size 10, got %d", cache.Size())
	}

	view := cache.Get(0)
	if view.K == nil || view.V == nil {
		t.Fatalf("Get(0) returned nil")
	}
	if view.K.Rows() != 10 {
		t.Errorf("Expected tensor rows to match window size 10, got %d", view.K.Rows())
	}
}

func TestSlidingWindowKVCache_UpdateWrapped(t *testing.T) {
	ctx := device.NewContext()
	defer ctx.Free()

	cache := &SlidingWindowKVCache{}
	config := conf.Config{
		Layers:     1,
		WindowSize: 4,
		KVHeads:    1,
		HeadDim:    4,
	}

	if err := cache.Init(ctx, config); err != nil {
		t.Fatalf("Init failed: %v", err)
	}
	defer cache.Free()

	kvDim := config.KVHeads * config.HeadDim
	kIn := ctx.NewTensor(1, kvDim)
	vIn := ctx.NewTensor(1, kvDim)

	// Just verify that updating past window size doesn't error
	// The standard TensorKVCache errors if pos >= size.
	// This should succeed.

	for i := 0; i < 10; i++ {
		err := cache.Update(0, i, kIn, vIn)
		if err != nil {
			t.Errorf("Update at pos %d failed (should handle wrap): %v", i, err)
		}
	}

	// Test negative pos (should still fail)
	if err := cache.Update(0, -1, kIn, vIn); err == nil {
		t.Errorf("Update at pos -1 should fail")
	}
}
