//go:build !metal && !tpu

package engine

import (
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/device"
)

// TestPagedKVCachePoolElementType checks that the paged KV pool's declared
// element type matches what the backend actually stores in it.
//
// The GPU backends store the K and V projections as packed half-precision
// values, while the CPU stores float32. The pool used to be declared
// float32 on every backend, so on a GPU it was allocated at twice the size
// the cache needed and a host-side read had to decode the halves by hand.
//
// The check is a round trip through a value that half precision cannot
// represent exactly. The read-back has to come back half-rounded on the
// GPU and exact on the CPU; a pool whose declared type disagreed with the
// stored type would return something else entirely.
func TestPagedKVCachePoolElementType(t *testing.T) {
	cache, ctx := newTestPagedCache(t, 512)
	defer cache.Free()
	defer ctx.Free()

	kvHeads := cache.kvHeads
	headDim := cache.headDim
	width := kvHeads * headDim

	// 0.1 is not exactly representable in half precision.
	const value = float32(0.1)

	k := ctx.NewTensorFP32(1, width)
	v := ctx.NewTensorFP32(1, width)
	if k == nil || v == nil {
		t.Fatal("failed to create the KV tensors")
	}
	defer k.Free()
	defer v.Free()

	data := make([]float32, width)
	for i := range data {
		data[i] = value
	}
	if err := k.LoadFrom(data); err != nil {
		t.Fatalf("failed to load the K tensor: %v", err)
	}
	if err := v.LoadFrom(data); err != nil {
		t.Fatalf("failed to load the V tensor: %v", err)
	}

	items := []struct {
		SeqID string
		Pos   int
		K     *device.Tensor
		V     *device.Tensor
	}{
		{SeqID: "seq-a", Pos: 0, K: k, V: v},
	}
	if err := cache.UpdateBatch(0, items); err != nil {
		t.Fatalf("UpdateBatch failed: %v", err)
	}

	pool := cache.kPools[0]
	if pool == nil {
		t.Fatal("the K pool was not allocated")
	}

	got := poolValue(pool, 0)
	switch pool.DataType() {
	case device.DataTypeF16:
		want := device.Float16ToFloat32(device.Float32ToFloat16(value))
		if got != want {
			t.Errorf("the pool holds %v, want the half-precision value %v", got, want)
		}
	case device.DataTypeF32:
		if got != value {
			t.Errorf("the pool holds %v, want the exact value %v", got, value)
		}
	default:
		t.Fatalf("the pool has the unexpected element type %v", pool.DataType())
	}

	// The pool must not be allocated at more than the space the stored
	// values need. The GPU backends used to declare it float32 while
	// storing half-precision values, which doubled the memory the cache
	// occupied.
	elemBytes := 4
	if pool.DataType() == device.DataTypeF16 {
		elemBytes = 2
	}
	capacity := cache.totalBlocks * cache.blockSize
	wantBytes := capacity * width * elemBytes
	if pool.SizeBytes() != wantBytes {
		t.Errorf("the pool occupies %d bytes, want %d", pool.SizeBytes(), wantBytes)
	}
}
