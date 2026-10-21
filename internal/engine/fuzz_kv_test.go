//go:build darwin && metal

package engine

import (
	"math/rand"
	"testing"

	conf "github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
)

// FuzzKVCacheUpdate fuzzes the sliding window cache.
//
// The fuzz target used to fuzz the contiguous TensorKVCache, which was
// deleted when the paged cache replaced it, so the target did not compile
// and never ran.
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
		cache := &SlidingWindowKVCache{}
		if err := cache.Init(ctx, config); err != nil {
			// A failure to initialise the cache is a defect, not a
			// reason to skip: a skip inside a fuzz target discards the
			// seed, so the corpus never grows for that input.
			t.Fatalf("the cache failed to initialise: %v", err)
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
		_ = kIn.LoadFrom(data)
		_ = vIn.LoadFrom(data)

		err := cache.Update("seq-0", layer, pos, kIn, vIn)

		// The invariants: a layer outside the range is an error, a
		// negative position is an error, and everything else is stored.
		if layer < 0 || layer >= config.Layers {
			if err == nil {
				t.Errorf("Expected error for invalid layer %d", layer)
			}
		} else if pos < 0 {
			if err == nil {
				t.Errorf("Expected error for invalid pos %d", pos)
			}
		} else if err != nil {
			t.Errorf("Unexpected error for valid update layer=%d pos=%d: %v", layer, pos, err)
		}
	})
}
