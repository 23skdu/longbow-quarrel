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

	expected := []float32{1.0, 2.0, 0.0, -0.0}
	for i := range dst {
		if math.Abs(float64(dst[i])-float64(expected[i])) > 1e-6 {
			t.Errorf("Fp16ToFp32 failed at %d: got %f, expected %f", i, dst[i], expected[i])
		}
	}
}
