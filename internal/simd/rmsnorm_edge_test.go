package simd

import (
	"math"
	"testing"
)

// The epsilon of the RMSNorm comes from the model's metadata, so a
// corrupt or hostile model file can make it negative. A negative epsilon
// turns the normalisation into a division by zero, and the output fills
// with NaN. The kernel clamps it.
//
// The fuzz target found this: its seed corpus held a case with a negative
// epsilon, and the target had never run because `make test-simd-fuzz`
// passed the flags in a form that fuzzed nothing.

// TestRMSNorm_NegativeEpsilon checks that a negative epsilon does not
// produce NaN.
func TestRMSNorm_NegativeEpsilon(t *testing.T) {
	input := []float32{1, 2, 3, 4}
	weight := []float32{1, 1, 1, 1}
	output := make([]float32, 4)

	RMSNorm(input, weight, output, 1, 4, -62)

	for i, v := range output {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("output[%d] is not finite with a negative epsilon: %v", i, v)
		}
	}
}

// TestRMSNorm_ZeroEpsilonZeroRow checks that an all-zero row with no
// epsilon produces zeros rather than NaN. Zero divided by zero is NaN,
// which is what the kernel used to produce.
func TestRMSNorm_ZeroEpsilonZeroRow(t *testing.T) {
	input := []float32{0, 0, 0, 0}
	weight := []float32{1, 1, 1, 1}
	output := make([]float32, 4)

	RMSNorm(input, weight, output, 1, 4, 0)

	for i, v := range output {
		if math.IsNaN(float64(v)) {
			t.Fatalf("output[%d] is NaN for an all-zero row: %v", i, v)
		}
		if v != 0 {
			t.Errorf("output[%d] = %v, want 0 for an all-zero row", i, v)
		}
	}
}

// TestRMSNorm_NormalisesTheRow checks the arithmetic itself: the output
// must have the same norm as the input, scaled by the weights.
func TestRMSNorm_NormalisesTheRow(t *testing.T) {
	input := []float32{3, 4, 0, 0}
	weight := []float32{1, 1, 1, 1}
	output := make([]float32, 4)

	RMSNorm(input, weight, output, 1, 4, 0)

	// The sum of squares of the input is 25, over 4 columns, so the
	// normalised value has a sum of squares of 4.
	sum := float32(0)
	for _, v := range output {
		sum += v * v
	}
	if math.Abs(float64(sum-4)) > 1e-4 {
		t.Errorf("the sum of squares of the output is %v, want 4", sum)
	}
}
