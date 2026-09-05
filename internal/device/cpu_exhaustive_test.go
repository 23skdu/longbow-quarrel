//go:build !metal && !cuda && !tpu
package device

import (
	"math"
	"testing"
)

func TestCPU_ReferenceKernels(t *testing.T) {
	// 1. RMSNorm
	weights := []float32{1.0, 1.0, 1.0}
	input := []float32{1.0, 2.0, 3.0}
	output := CPURMSNorm(input, weights, 1e-5)
	
	sumSquare := float32(0)
	for _, v := range output { sumSquare += v * v }
	// mean square should be approx 1.0
	if math.Abs(float64(sumSquare/3.0 - 1.0)) > 1e-3 {
		t.Errorf("CPURMSNorm output not normalized: mean square = %f", sumSquare/3.0)
	}

	// 2. RoPE
	q := []float32{1.0, 0.0, 1.0, 0.0}
	output_rope := CPURoPE(q, 0, 2, 2, 10000.0)
	if len(output_rope) != 4 {
		t.Error("CPURoPE output length mismatch")
	}

	// 3. SwiGLU
	gate := []float32{1.0}
	up := []float32{2.0}
	output_swiglu := CPUSwiGLU(gate, up)
	// silu(1.0) * 2.0 = 0.731 * 2.0 = 1.462
	if math_abs(float64(output_swiglu[0] - 1.4621)) > 1e-3 {
		t.Errorf("CPUSwiGLU failed: got %f", output_swiglu[0])
	}
}

func TestCPU_MatMul(t *testing.T) {
	// 2x2 matmul (A * B^T)
	a := []float32{1, 2, 3, 4}
	b := []float32{5, 6, 7, 8} // B^T = [[5, 6], [7, 8]]^T = [[5, 7], [6, 8]]
	// C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[0][1] = 1*5 + 2*6 = 17
	c := CPUMatMul(a, b, 2, 2, 2)
	if c[0] != 17 {
		t.Errorf("CPUMatMul failed: expected 17, got %f", c[0])
	}
}

func TestCPU_Validation(t *testing.T) {
	data := []float32{1.0, float32(math.NaN()), 3.0}
	if !HasAnyNaN(data) {
		t.Error("HasAnyNaN failed to detect NaN")
	}
	if IsValid(data) {
		t.Error("IsValid failed to detect NaN")
	}
	
	infData := []float32{1.0, float32(math.Inf(1)), 3.0}
	if !HasAnyInf(infData) {
		t.Error("HasAnyInf failed to detect Inf")
	}
	
	if Float32Max([]float32{1.0, 10.0, 5.0}) != 10.0 {
		t.Error("Float32Max failed")
	}
}

func TestCPU_Conversion(t *testing.T) {
	val := float32(3.14)
	f16 := Float32ToFloat16(val)
	back := Float16ToFloat32(f16)
	if math_abs(float64(back - val)) > 1e-2 {
		t.Errorf("F16 conversion loss too high: %f -> %f", val, back)
	}
}
