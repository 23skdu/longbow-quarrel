//go:build darwin && metal

package engine

import (
	"math/rand"
	"testing"

	conf "github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
)

func FuzzKVCacheUpdate(f *testing.F) {
	// Add some seed corpus
	f.Add(0, 0)
	f.Add(1, 10)
	f.Add(0, -1) // Invalid pos
	f.Add(5, 5)  // Invalid layer

	f.Fuzz(func(t *testing.T, layer int, pos int) {
		// Limit inputs to save resources during fuzzing
		if pos > 1000 || pos < -100 {
			return
		}
		if layer > 10 || layer < -10 {
			return
		}

		ctx := device.NewContext()
		defer ctx.Free()

		config := conf.Config{Layers: 2, WindowSize: 32, KVHeads: 2, HeadDim: 8}
		cache := &TensorKVCache{}
		if err := cache.Init(ctx, config); err != nil {
			t.Skipf("Init failed (likely OOM or device issue): %v", err)
		}
		defer cache.Free()

		kvDim := config.KVHeads * config.HeadDim
		kIn := ctx.NewTensor(1, kvDim)
		vIn := ctx.NewTensor(1, kvDim)

		// Fill with random data
		data := make([]float32, kvDim)
		for i := range data {
			data[i] = rand.Float32()
		}
		kIn.LoadFrom(data)
		vIn.LoadFrom(data)

		err := cache.Update(layer, pos, kIn, vIn)

		// Basic invariant check
		if layer < 0 || layer >= config.Layers {
			if err == nil {
				t.Errorf("Expected error for invalid layer %d", layer)
			}
		} else if pos < 0 || pos >= config.WindowSize {
			if err == nil {
				t.Errorf("Expected error for invalid pos %d (size %d)", pos, config.WindowSize)
			}
		} else {
			if err != nil {
				t.Errorf("Unexpected error for valid update layer=%d pos=%d: %v", layer, pos, err)
			}
		}
	})
}
