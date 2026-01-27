//go:build darwin && metal

package engine

import (
	"testing"

	conf "github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
)

func TestTensorKVCache_Lifecycle(t *testing.T) {
	ctx := device.NewContext()
	defer ctx.Free()

	cache := &TensorKVCache{}
	config := conf.Config{
		Layers:     2,
		WindowSize: 32,
		KVHeads:    4,
		HeadDim:    64,
	}

	// Test Init
	err := cache.Init(ctx, config)
	if err != nil {
		t.Fatalf("Init failed: %v", err)
	}
	defer cache.Free()

	if cache.Size() != 32 {
		t.Errorf("Expected size 32, got %d", cache.Size())
	}

	// Test Get
	view := cache.Get(0)
	if view.K == nil || view.V == nil {
		t.Errorf("Get(0) returned nil tensors")
	}

	view1 := cache.Get(1)
	if view1.K == nil || view1.V == nil {
		t.Errorf("Get(1) returned nil tensors")
	}

	// Test Get Invalid
	view2 := cache.Get(2)
	if view2.K != nil || view2.V != nil {
		t.Errorf("Get(2) should return nil for invalid layer")
	}
}

func TestTensorKVCache_Update(t *testing.T) {
	ctx := device.NewContext()
	defer ctx.Free()

	cache := &TensorKVCache{}
	config := conf.Config{
		Layers:     1,
		WindowSize: 8,
		KVHeads:    2,
		HeadDim:    4,
	}

	if err := cache.Init(ctx, config); err != nil {
		t.Fatalf("Init failed: %v", err)
	}
	defer cache.Free()

	// Create dummy input tensors (1x8)
	kvDim := config.KVHeads * config.HeadDim
	kInput := ctx.NewTensor(1, kvDim)
	vInput := ctx.NewTensor(1, kvDim)

	// Fill with data using LoadFrom (if available on dummy tensor, or we rely on zero init/random)
	// Here we just check the call succeeds
	err := cache.Update(0, 5, kInput, vInput)
	if err != nil {
		t.Errorf("Update failed for valid position: %v", err)
	}

	// Test OOB Update
	err = cache.Update(0, 10, kInput, vInput)
	if err == nil {
		t.Errorf("Expected error for OOB position 10 (size 8)")
	}

	// Test Invalid Layer
	err = cache.Update(1, 5, kInput, vInput)
	if err == nil {
		t.Errorf("Expected error for invalid layer 1 (layers 1)")
	}
}
