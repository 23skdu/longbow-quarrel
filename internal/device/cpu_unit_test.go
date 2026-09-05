//go:build !metal && !cuda && !tpu
package device

import (
	"testing"
)

func TestCPU_TensorLifecycle(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	// 1. NewTensor (F16)
	ten := ctx.NewTensor(10, 10)
	if ten.Rows() != 10 || ten.Cols() != 10 {
		t.Errorf("Expected 10x10 tensor, got %dx%d", ten.Rows(), ten.Cols())
	}
	if ten.DataType() != DataType(3) { // 3 is F16 in this codebase
		t.Errorf("Expected F16 (3), got %v", ten.DataType())
	}

	// 2. NewTensorFP32
	ten32 := ctx.NewTensorFP32(5, 5)
	if ten32.DataType() != DataTypeF32 {
		t.Errorf("Expected F32, got %v", ten32.DataType())
	}

	// 3. LoadFromF32 and ToHostF32
	data := make([]float32, 25)
	for i := range data { data[i] = float32(i) }
	ten32.LoadFromF32(data)
	
	back := ten32.ToHostF32()
	for i := range data {
		if back[i] != data[i] {
			t.Errorf("Data mismatch at %d: %f != %f", i, back[i], data[i])
		}
	}

	// 4. ZeroInit
	ten32.ZeroInit()
	back2 := ten32.ToHostF32()
	for i := range back2 {
		if back2[i] != 0 {
			t.Errorf("ZeroInit failed at %d: %f", i, back2[i])
		}
	}

	// 5. CopyToF16 (F32 -> F16 stub for CPU)
	ten32.LoadFromF32(data)
	f16Target := ten32.CopyToF16()
	
	back16 := f16Target.ToHostF32()
	for i := range data {
		// FP16 precision check (approx)
		if math_abs(float64(back16[i] - data[i])) > 1e-2 {
			t.Errorf("CopyToF16 deviation at %d: %f -> %f", i, data[i], back16[i])
		}
	}
}

func math_abs(x float64) float64 {
	if x < 0 { return -x }
	return x
}

func TestCPU_ContextOps(t *testing.T) {
	ctx := NewContext()
	ctx.SetNumThreads(4)
	if ctx.NumThreads() != 4 {
		t.Errorf("SetNumThreads failed")
	}
	ctx.Synchronize() // Should do nothing/not panic
}

func TestCPU_AllocatedBytes(t *testing.T) {
	// For CPU, memory tracking might be disabled or pooled. 
	// We'll just verify the call doesn't panic.
	_ = CPUAllocatedBytes()
}
