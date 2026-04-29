package engine

import (
	"testing"

	conf "github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
)

func FuzzPagedKVCache_TurboQuant(f *testing.F) {
	f.Fuzz(func(t *testing.T, heads, headDim, layers, windowSize int) {
		if heads <= 0 || heads > 64 {
			t.Skip("Invalid heads")
		}
		if headDim <= 0 || headDim > 512 {
			t.Skip("Invalid headDim")
		}
		if layers <= 0 || layers > 32 {
			t.Skip("Invalid layers")
		}
		if windowSize < 16 || windowSize > 4096 {
			t.Skip("Invalid windowSize")
		}

		ctx := device.NewContext()
		defer ctx.Free()

		cache := &PagedKVCache{}
		cfg := conf.Config{
			KVHeads:     heads,
			HeadDim:    headDim,
			Layers:     layers,
			WindowSize: windowSize,
		}

		cache.Precision = device.DataTypeTQ1_0

		err := cache.Init(ctx, cfg)
		if err != nil {
			t.Fatalf("Init failed: %v", err)
		}
		defer cache.Free()

		// Verify structure
		if cache.blockSize != 16 {
			t.Errorf("Expected block size 16, got %d", cache.blockSize)
		}

		// Test allocation
		err = cache.Allocate("test-seq", 16)
		if err != nil {
			t.Errorf("Allocate failed: %v", err)
		}
	})
}