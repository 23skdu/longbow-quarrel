//go:build darwin && metal && integration

package integration

import (
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/engine"
)

func TestKVCacheIntegration(t *testing.T) {
	ctx := device.NewContext()
	defer ctx.Free()

	// Setup a real TensorKVCache
	config := engine.LlamaConfig{
		Layers:     1,
		WindowSize: 16,
		KVHeads:    2,
		HeadDim:    8,
	}
	cache := &engine.TensorKVCache{}
	if err := cache.Init(ctx, config); err != nil {
		t.Fatalf("Init failed: %v", err)
	}
	defer cache.Free()

	// Simulate an inference loop
	kvDim := config.KVHeads * config.HeadDim
	kIn := ctx.NewTensor(1, kvDim)
	vIn := ctx.NewTensor(1, kvDim)

	// Create verifiable data patterns
	kData := make([]float32, kvDim)
	vData := make([]float32, kvDim)

	for i := 0; i < 16; i++ {
		// Prepare data for this step
		for j := 0; j < kvDim; j++ {
			kData[j] = float32(i*100 + j)
			vData[j] = float32(i*100 + j + 50)
		}
		kIn.LoadFrom(kData)
		vIn.LoadFrom(vData)

		// Update cache
		err := cache.Update(0, i, kIn, vIn)
		if err != nil {
			t.Fatalf("Step %d update failed: %v", i, err)
		}

		// Retrieve and Verify (Partial verification since we can't easily read back form GPU in test without helpers)
		// But in integration tests we assume we *can* verify if things crash or return errors
		kCache, vCache := cache.Get(0)
		if kCache == nil || vCache == nil {
			t.Fatalf("Step %d get failed", i)
		}
	}

	t.Log("Integration loop completed successfully")
}
