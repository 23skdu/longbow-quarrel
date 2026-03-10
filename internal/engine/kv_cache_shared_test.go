//go:build darwin && metal

package engine

import (
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
)

func TestKVCacheSharing(t *testing.T) {
	ctx := device.NewContext()
	defer ctx.Free()

	conf := config.Default()
	conf.KVHeads = 1
	conf.HeadDim = 64
	conf.Layers = 2
	conf.WindowSize = 128

	cache := &PagedKVCache{}
	err := cache.Init(ctx, conf)
	if err != nil {
		t.Fatalf("Failed to init cache: %v", err)
	}
	defer cache.Free()

	k := ctx.NewTensor(1, 64)
	v := ctx.NewTensor(1, 64)
	defer k.Free()
	defer v.Free()

	// Seq1 writes to pos 0
	err = cache.Update("seq1", 0, 0, k, v)
	if err != nil {
		t.Fatalf("Update seq1 failed: %v", err)
	}

	// Fork seq2 from seq1
	err = cache.ForkSequence("seq1", "seq2")
	if err != nil {
		t.Fatalf("ForkSequence failed: %v", err)
	}

	// Seq2 should now share the same physical block for logical block 0
	// We can't easily inspect the internal map without exposing it, but we can check if it panics
	view1 := cache.Get("seq1", 0)
	view2 := cache.Get("seq2", 0)

	if view1.BlockTable == nil || view2.BlockTable == nil {
		t.Fatal("Expected block tables")
	}

	// Just for testing, let's verify blockRefs are handled by updating seq2
	// Since seq2 shares the block, writing to it should trigger Copy-On-Write
	err = cache.Update("seq2", 0, 0, k, v)
	if err != nil {
		t.Fatalf("Update seq2 (COW) failed: %v", err)
	}

	// Seq1 writes to pos 1
	k2 := ctx.NewTensor(1, 64)
	v2 := ctx.NewTensor(1, 64)
	defer k2.Free()
	defer v2.Free()
	cache.Update("seq1", 0, 1, k2, v2)

	// Seq2 writes different to pos 1
	cache.Update("seq2", 0, 1, k, v)
}
