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
				if math.Abs(input[i]-tc.expected[i]) > 1e-8 {
					t.Errorf("expected %v, got %v", tc.expected, input)
					break
				}
			}
		})
	}
}
