//go:build linux && amd64 && cuda && cgo

package device

import (
	"math"
	"testing"
)

// TestCUDAPagedStoreUsesThePhysicalPositions checks that the paged store
// puts every token in the slot its physical position names.
//
// The positions are carried as float32: the store kernel reads them as
// float32 and casts them to an integer. They used to be carried as
// int32, so the kernel read the bits of the integer as a float -- a
// denormal -- and every token landed in the first slot.
//
// The parity test in internal/engine did not catch it: the attention then
// read an unwritten slot, which happened to be zero, and the difference
// was small enough to fall inside the tolerance.
func TestCUDAPagedStoreUsesThePhysicalPositions(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	const (
		kvDim     = 4
		numTokens = 3
		capacity  = 64
	)

	pool := ctx.NewTensorWithType(capacity, kvDim, DataTypeF16)
	if pool == nil {
		t.Fatal("failed to create the pool")
	}
	defer pool.Free()

	vPool := ctx.NewTensorWithType(capacity, kvDim, DataTypeF16)
	if vPool == nil {
		t.Fatal("failed to create the value pool")
	}
	defer vPool.Free()

	// Fill the pool with a sentinel, so that the slots that nothing
	// names can be told apart from the ones the store wrote.
	sentinel := make([]float32, capacity*kvDim)
	for i := range sentinel {
		sentinel[i] = 42
	}
	if err := pool.LoadFrom(sentinel); err != nil {
		t.Fatalf("failed to fill the pool: %v", err)
	}

	// Every token has a distinct value.
	kData := make([]float32, numTokens*kvDim)
	for i := range kData {
		kData[i] = float32(i/kvDim) + 1
	}
	kTensor := ctx.NewTensorFP32(numTokens, kvDim)
	if kTensor == nil {
		t.Fatal("failed to create the key tensor")
	}
	defer kTensor.Free()
	if err := kTensor.LoadFrom(kData); err != nil {
		t.Fatalf("failed to load the keys: %v", err)
	}

	vTensor := ctx.NewTensorFP32(numTokens, kvDim)
	if vTensor == nil {
		t.Fatal("failed to create the value tensor")
	}
	defer vTensor.Free()
	if err := vTensor.LoadFrom(kData); err != nil {
		t.Fatalf("failed to load the values: %v", err)
	}

	// The tokens go to three different, non-sequential slots.
	positions := []float32{5, 9, 2}
	physTensor := ctx.NewTensorFP32(1, numTokens)
	if physTensor == nil {
		t.Fatal("failed to create the position tensor")
	}
	defer physTensor.Free()
	if err := physTensor.LoadFrom(positions); err != nil {
		t.Fatalf("failed to load the positions: %v", err)
	}

	ctx.StoreKVPagedBatch(kTensor, vTensor, pool, vPool, physTensor, kvDim, numTokens)
	ctx.Synchronize()

	host := pool.ToHostF32()
	if len(host) < capacity*kvDim {
		t.Fatalf("the pool holds %d values, want %d", len(host), capacity*kvDim)
	}

	for token, slot := range positions {
		want := float32(token) + 1
		for d := 0; d < kvDim; d++ {
			got := host[int(slot)*kvDim+d]
			if math.Abs(float64(got-want)) > 0.001 {
				t.Fatalf("the token %d is in the slot %d: the value %d is %v, want %v",
					token, int(slot), d, got, want)
			}
		}
	}

	// The slots that were not named must still hold the sentinel.
	untouched := []int{0, 1, 3, 4, 6, 7, 8}
	for _, slot := range untouched {
		for d := 0; d < kvDim; d++ {
			if math.Abs(float64(host[slot*kvDim+d]-42)) > 0.001 {
				t.Fatalf("the slot %d was written, but nothing named it", slot)
			}
		}
	}
}
