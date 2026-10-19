//go:build !metal && !tpu

package engine

import (
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
)

// newTestPagedCache builds an initialised PagedKVCache with a small capacity.
func newTestPagedCache(t *testing.T, capacity int) (*PagedKVCache, *device.Context) {
	t.Helper()

	ctx := device.NewContext()
	if ctx == nil {
		t.Fatal("failed to create a device context")
	}

	cache := &PagedKVCache{}
	cfg := config.Default()
	cfg.Layers = 2
	cfg.KVHeads = 2
	cfg.HeadDim = 8
	cfg.KVCacheSize = capacity
	cfg.WindowSize = capacity
	cfg.SeqLen = capacity

	if err := cache.Init(ctx, cfg); err != nil {
		ctx.Free()
		t.Fatalf("failed to initialise the paged cache: %v", err)
	}
	return cache, ctx
}
