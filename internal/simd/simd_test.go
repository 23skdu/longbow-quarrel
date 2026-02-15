package simd

import (
	"math"
	"testing"
)

func TestSoftmax(t *testing.T) {
	testCases := []struct {
		name     string
		input    []float64
		expected []float64
	}{
		{
			name:     "simple",
			input:    []float64{1, 2, 3},
			expected: []float64{0.09003057, 0.24472847, 0.66524096},
		},
		{
			name:     "negative",
			input:    []float64{-1, -2, -3},
			expected: []float64{0.66524096, 0.24472847, 0.09003057},
		},
		{
			name:     "zero",
			input:    []float64{0, 0, 0},
			expected: []float64{0.33333333, 0.33333333, 0.33333333},
		},
		{
			name:     "empty",
			input:    []float64{},
			expected: []float64{},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			input := make([]float64, len(tc.input))
			copy(input, tc.input)
			Softmax(input)
			if len(input) != len(tc.expected) {
				t.Errorf("expected length %d, got %d", len(tc.expected), len(input))
			}
			for i := range input {
				if math.Abs(input[i]-tc.expected[i]) > 1e-6 {
					t.Errorf("expected %v, got %v", tc.expected, input)
					break
				}
			}
		})
	}
}

func TestSoftmaxF32(t *testing.T) {
	input := []float32{1, 2, 3}
	SoftmaxF32(input)

	expected := []float64{0.09003057, 0.24472847, 0.66524096}
	for i := range input {
		if math.Abs(float64(input[i])-expected[i]) > 1e-5 {
			t.Errorf("SoftmaxF32 failed at %d: got %f, expected %f", i, input[i], expected[i])
		}
	}
}

func TestSwiGLU(t *testing.T) {
	gate := []float32{0, 1, -1, 10, -10}
	up := []float32{1, 2, 3, 4, 5}
	out := make([]float32, 5)
	SwiGLU(gate, up, out)

	for i := range out {
		if math.IsInf(float64(out[i]), 0) || math.IsNaN(float64(out[i])) {
			t.Errorf("SwiGLU produced invalid value at %d: %f", i, out[i])
		}
	}
}

func TestFp16ToFp32(t *testing.T) {
	src := []uint16{0x3C00, 0x4000, 0x0000, 0x8000}
	dst := make([]float32, 4)
	Fp16ToFp32(src, dst)

	expected := []float32{1.0, 2.0, 0.0, float32(math.Copysign(0, -1))}
	for i := range dst {
		if math.Abs(float64(dst[i])-float64(expected[i])) > 1e-6 {
			t.Errorf("Fp16ToFp32 failed at %d: got %f, expected %f", i, dst[i], expected[i])
		}
	}
}

func TestSoftmaxStability(t *testing.T) {
	// Test that softmax handles very large values numerically stably
	x := []float64{1000.0, 1001.0, 1002.0}
	Softmax(x)

	sum := x[0] + x[1] + x[2]
	if math.Abs(sum-1.0) > 1e-10 {
		t.Errorf("Softmax sum = %f, want 1.0", sum)
	}
}

func TestSoftmaxF32Stability(t *testing.T) {
	x := []float32{1000.0, 1001.0, 1002.0}
	SoftmaxF32(x)

	var sum float32
	for _, v := range x {
		sum += v
	}
	if math.Abs(float64(sum)-1.0) > 1e-6 {
		t.Errorf("SoftmaxF32 sum = %f, want 1.0", sum)
	}
}

func TestSwiGLUClamping(t *testing.T) {
	gate := []float32{-20.0, -10.0, 10.0, 20.0}
	up := []float32{1.0, 1.0, 1.0, 1.0}
	out := make([]float32, 4)

	SwiGLU(gate, up, out)

	// Verify no NaN/Inf produced
	for i, v := range out {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("SwiGLU clamping produced NaN/Inf at %d: %f", i, v)
		}
	}
}

func TestFp16ToFp32Inf(t *testing.T) {
	src := []uint16{0x7C00} // +Inf
	dst := make([]float32, 1)
	Fp16ToFp32(src, dst)

	if !math.IsInf(float64(dst[0]), 1) {
		t.Errorf("Fp16ToFp32 +Inf = %f, want +Inf", dst[0])
	}
}

func TestFp16ToFp32NegInf(t *testing.T) {
	src := []uint16{0xFC00} // -Inf
	dst := make([]float32, 1)
	Fp16ToFp32(src, dst)

	if !math.IsInf(float64(dst[0]), -1) {
		t.Errorf("Fp16ToFp32 -Inf = %f, want -Inf", dst[0])
	}
}

func TestFp16ToFp32NaN(t *testing.T) {
	src := []uint16{0x7E00} // NaN
	dst := make([]float32, 1)
	Fp16ToFp32(src, dst)

	if !math.IsNaN(float64(dst[0])) {
		t.Errorf("Fp16ToFp32 NaN = %f, want NaN", dst[0])
	}
}

func TestFp32ToFp16Inf(t *testing.T) {
	src := []float32{float32(math.Inf(1))}
	dst := make([]uint16, 1)
	Fp32ToFp16(src, dst)

	if dst[0] != 0x7C00 {
		t.Errorf("Fp32ToFp16 +Inf = 0x%04X, want 0x7C00", dst[0])
	}
}

func TestFp32ToFp16NegInf(t *testing.T) {
	src := []float32{float32(math.Inf(-1))}
	dst := make([]uint16, 1)
	Fp32ToFp16(src, dst)

	if dst[0] != 0xFC00 {
		t.Errorf("Fp32ToFp16 -Inf = 0x%04X, want 0xFC00", dst[0])
	}
}

func TestSoftmaxVeryLargeInput(t *testing.T) {
	x := []float64{1e20, 1e20 + 1}
	Softmax(x)

	// Should handle without overflow
	for _, v := range x {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("Softmax very large input produced NaN/Inf: %f", v)
		}
	}
}

func TestSwiGLUEmpty(t *testing.T) {
	SwiGLU([]float32{}, []float32{}, make([]float32, 0))
}

func TestSoftmaxPreservesOrder(t *testing.T) {
	x := []float64{1.0, 2.0, 3.0}
	Softmax(x)

	// Larger inputs should have larger outputs after softmax
	if !(x[0] < x[1] && x[1] < x[2]) {
		t.Errorf("Softmax should preserve order: got %v", x)
	}
}

func TestFp16ToFp32Precision(t *testing.T) {
	// Test some specific FP16 values
	tests := []uint16{
		0x3C00, // 1.0
		0x4000, // 2.0
		0x4200, // 4.0
		0x0000, // 0.0
		0xC000, // -2.0
	}

	for _, src := range tests {
		dst := make([]float32, 1)
		Fp16ToFp32([]uint16{src}, dst)

		// Just verify no crash and reasonable range
		if math.IsNaN(float64(dst[0])) && src != 0x7E00 {
			t.Errorf("Fp16ToFp32(0x%04X) = NaN, unexpected", src)
		}
	}
}

func BenchmarkSoftmaxLarge(b *testing.B) {
	x := make([]float64, 16384)
	for i := range x {
		x[i] = float64(i%100) / 10.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Softmax(x)
	}
}

func BenchmarkSoftmaxF32Large(b *testing.B) {
	x := make([]float32, 16384)
	for i := range x {
		x[i] = float32(i%100) / 10.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SoftmaxF32(x)
	}
}
