package device

import (
	"testing"
	"unsafe"

	"github.com/apache/arrow-go/v18/arrow"
)

func TestToArrowArray(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	// 1. Test F32 Zero-Copy
	rows, cols := 2, 4
	tensor := ctx.NewTensorWithType(rows, cols, DataTypeF32)
	defer tensor.Free()

	// Fill with test data
	data := make([]float32, rows*cols)
	for i := range data {
		data[i] = float32(i)
	}
	_ = tensor.LoadFrom(data)

	// Convert to Arrow
	arr, err := tensor.ToArrowArray(nil)
	if err != nil {
		t.Fatalf("ToArrowArray failed: %v", err)
	}
	defer arr.Release()

	// Verify dimensions
	if arr.Len() != rows {
		t.Errorf("expected length %d, got %d", rows, arr.Len())
	}

	// Verify zero-copy: pointer equivalence on unified/host memory
	if !tensor.IsDevice() {
		rawData := tensor.RawData()
		arrowChild := arr.ListValues()
		arrowRaw := arrowChild.Data().Buffers()[1].Bytes()

		if unsafe.Pointer(&rawData[0]) != unsafe.Pointer(&arrowRaw[0]) {
			t.Errorf("True Zero-Copy FAILED: pointer mismatch")
		}
	}

	// 2. Test F16 Zero-Copy (if supported by backend)
	tensorF16 := ctx.NewTensorWithType(rows, cols, DataTypeF16)
	if tensorF16 != nil {
		defer tensorF16.Free()
		
		arr16, err := tensorF16.ToArrowArray(nil)
		if err != nil {
			t.Fatalf("ToArrowArray F16 failed: %v", err)
		}
		defer arr16.Release()

		if arr16.DataType().(*arrow.FixedSizeListType).Elem().ID() != arrow.FLOAT16 {
			t.Errorf("expected Float16 element type, got %v", arr16.DataType())
		}
		
		// Verify zero-copy on host/unified memory
		if !tensorF16.IsDevice() {
			rawData16 := tensorF16.RawData()
			arrowRaw16 := arr16.ListValues().Data().Buffers()[1].Bytes()
			if unsafe.Pointer(&rawData16[0]) != unsafe.Pointer(&arrowRaw16[0]) {
				t.Errorf("True Zero-Copy F16 FAILED: pointer mismatch")
			}
		}
	}
}
