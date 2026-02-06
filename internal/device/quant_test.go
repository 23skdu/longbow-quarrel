//go:build darwin && metal

package device

import (
	"math"
	"testing"
)

func TestIQ4_NL_Allocation(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	rows, cols := 128, 512
	tensor := ctx.NewTensorWithType(rows, cols, DataTypeIQ4_NL)
	if tensor == nil {
		t.Fatal("Failed to create IQ4_NL tensor")
	}
	defer tensor.Free()

	expectedSize := (rows * cols / 32) * 18
	if tensor.sizeBytes != expectedSize {
		t.Errorf("Expected size %d, got %d", expectedSize, tensor.sizeBytes)
	}
	if tensor.dataType != DataTypeIQ4_NL {
		t.Errorf("Expected datatype IQ4_NL, got %d", tensor.dataType)
	}
}

func TestMXFP4_Allocation(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	rows, cols := 128, 512
	tensor := ctx.NewTensorWithType(rows, cols, DataTypeMXFP4)
	if tensor == nil {
		t.Fatal("Failed to create MXFP4 tensor")
	}
	defer tensor.Free()

	expectedSize := (rows * cols / 32) * 18
	if tensor.sizeBytes != expectedSize {
		t.Errorf("Expected size %d, got %d", expectedSize, tensor.sizeBytes)
	}
}

func TestIQ4_NL_Linear_Stability(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	rows, cols := 64, 256
	input := ctx.NewTensor(1, cols)
	weight := ctx.NewTensorWithType(rows, cols, DataTypeIQ4_NL)
	out := ctx.NewTensor(1, rows)

	defer input.Free()
	defer weight.Free()
	defer out.Free()

	// Fill weight with dummy data (all 0s is safe)
	weightData := make([]byte, weight.sizeBytes)
	// Set scale 'd' to 1.0 (FP16 0x3C00) for all blocks
	for i := 0; i < weight.sizeBytes; i += 18 {
		weightData[i] = 0x00
		weightData[i+1] = 0x3C
	}
	weight.LoadFromRaw(weightData)

	// Fill input
	inData := make([]float32, cols)
	for i := range inData {
		inData[i] = 0.5
	}
	input.LoadFrom(inData)

	err := input.LinearInto(weight, out, 1.0)
	if err != nil {
		t.Fatalf("LinearInto failed: %v", err)
	}
	ctx.Synchronize()

	res := out.ToHost()
	for i, v := range res {
		if math.IsNaN(float64(v)) {
			t.Errorf("NaN detected at index %d", i)
		}
	}
}

func TestMXFP4_Linear_Stability(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	rows, cols := 64, 256
	input := ctx.NewTensor(1, cols)
	weight := ctx.NewTensorWithType(rows, cols, DataTypeMXFP4)
	out := ctx.NewTensor(1, rows)

	defer input.Free()
	defer weight.Free()
	defer out.Free()

	// Fill weight with dummy data
	weightData := make([]byte, weight.sizeBytes)
	// Set scale to 1.0 (pow(2, 127-127))
	for i := 0; i < weight.sizeBytes; i += 18 {
		weightData[i] = 127
	}
	weight.LoadFromRaw(weightData)

	// Fill input
	inData := make([]float32, cols)
	for i := range inData {
		inData[i] = 0.5
	}
	input.LoadFrom(inData)

	err := input.LinearInto(weight, out, 1.0)
	if err != nil {
		t.Fatalf("LinearInto failed: %v", err)
	}
	ctx.Synchronize()

	res := out.ToHost()
	for i, v := range res {
		if math.IsNaN(float64(v)) {
			t.Errorf("NaN detected at index %d", i)
		}
	}
}
