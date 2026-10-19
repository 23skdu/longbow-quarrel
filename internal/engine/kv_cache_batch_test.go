//go:build !metal && !tpu

package engine

import (
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/device"
)

// TestPagedKVCacheUpdateBatch covers the batched KV update path, which
// writes the K and V of several sequences in a single operation.
func TestPagedKVCacheUpdateBatch(t *testing.T) {
	cache, ctx := newTestPagedCache(t, 512)
	defer cache.Free()
	defer ctx.Free()

	// An empty batch is a no-op, not an error.
	if err := cache.UpdateBatch(0, nil); err != nil {
		t.Errorf("an empty batch should be a no-op, got: %v", err)
	}

	// The batched path uses a single K and V tensor for the whole batch,
	// laid out as [batch, kvDim].
	kvHeads := cache.kvHeads
	headDim := cache.headDim
	width := kvHeads * headDim
	batchSize := 2

	k := ctx.NewTensorFP32(batchSize, width)
	v := ctx.NewTensorFP32(batchSize, width)
	if k == nil || v == nil {
		t.Fatal("failed to create the KV tensors")
	}
	defer k.Free()
	defer v.Free()

	fill := func(tensor *device.Tensor, base float32) {
		data := make([]float32, batchSize*width)
		for i := range data {
			data[i] = base + float32(i%4)
		}
		if err := tensor.LoadFrom(data); err != nil {
			t.Fatalf("failed to load the tensor: %v", err)
		}
	}
	fill(k, 1)
	fill(v, 2)

	items := []struct {
		SeqID string
		Pos   int
		K     *device.Tensor
		V     *device.Tensor
	}{
		{SeqID: "seq-a", Pos: 0, K: k, V: v},
		{SeqID: "seq-b", Pos: 0, K: k, V: v},
	}

	if err := cache.UpdateBatch(0, items); err != nil {
		t.Fatalf("UpdateBatch failed: %v", err)
	}

	// Both sequences must have been given a physical block.
	blocks := map[string][]int32{}
	for _, item := range items {
		blocks[item.SeqID] = cache.GetSequenceBlocks(item.SeqID)
		if len(blocks[item.SeqID]) == 0 {
			t.Errorf("sequence %s was not given a physical block", item.SeqID)
		}
	}

	// The K and V values must have been stored at the physical slot the
	// sequence was assigned. This is what the old implementation got
	// wrong: it read the positions from an empty buffer, so it either
	// panicked (CPU) or wrote to a garbage slot (GPU).
	for _, item := range items {
		seqBlocks := blocks[item.SeqID]
		if len(seqBlocks) == 0 {
			continue
		}
		physPos := int(seqBlocks[0])*cache.blockSize + item.Pos
		off := physPos * width

		// poolValue reads the pool the way the backend stores it: float32
		// on the CPU, half precision on CUDA and Metal.
		pool := cache.kPools[0]
		src := k.ToHostF32()
		for d := 0; d < width; d++ {
			got := poolValue(pool, off+d)
			want := src[d] // the first sequence's K row
			if item.SeqID == "seq-b" {
				want = src[width+d] // the second sequence's K row
			}
			if got != want {
				t.Fatalf("sequence %s: K[%d] = %v, want %v (the value was not stored at the right slot)",
					item.SeqID, d, got, want)
			}
		}
	}
}

// TestPagedKVCacheUpdateBatch_Uninitialised checks the guard.
func TestPagedKVCacheUpdateBatch_Uninitialised(t *testing.T) {
	cache := &PagedKVCache{}
	if err := cache.UpdateBatch(0, nil); err == nil {
		t.Error("expected an error when the cache has not been initialised")
	}
}
