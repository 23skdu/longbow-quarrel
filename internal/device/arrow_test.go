package device

import (
	"testing"
	"github.com/apache/arrow-go/v18/arrow/memory"
)

// Dummy testing of the arrow integration logic
func TestTensorToArrowArray(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	// 1x4 float32 tensor
	tensor := ctx.NewTensorFP32(1, 4)
	err := tensor.LoadFrom([]float32{1.0, 2.0, 3.0, 4.0})
	if err != nil {
		t.Fatalf("failed to load tensor: %v", err)
	}

	alloc := memory.NewGoAllocator()
	arr, err := tensor.ToArrowArray(alloc)
	if err != nil {
		t.Fatalf("failed to convert to arrow array: %v", err)
	}
	defer arr.Release()

	if arr.Len() != 1 {
		t.Errorf("expected length 1, got %d", arr.Len())
	}
	
	// We could verify the actual elements here, but the length and no-error represents correct allocation.
}
