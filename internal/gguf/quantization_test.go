package gguf

import (
	"testing"
)

// maxGroupStep returns the largest step of the K-quantizers over the
// values: every group of 32 values is represented with fifteen steps
// between its lowest and its highest value.
func maxGroupStep(values []float32) float64 {
	const (
		groupSize = 32
		steps     = 15
	)
	maxStep := 0.0
	for start := 0; start+groupSize <= len(values); start += groupSize {
		group := values[start : start+groupSize]
		low, high := group[0], group[0]
		for _, v := range group {
			if v < low {
				low = v
			}
			if v > high {
				high = v
			}
		}
		if step := float64(high-low) / steps; step > maxStep {
			maxStep = step
		}
	}
	return maxStep
}

// TestQuantizationDequantization validates that quantization and dequantization are reversible within acceptable error bounds
func TestQuantizationDequantization(t *testing.T) {
	t.Run("Q4K_Reversibility", func(t *testing.T) {
		// Create test data. The K-quants work on blocks of 256
		// values, so the pattern is repeated to fill one.
		original := make([]float32, 256)
		pattern := []float32{
			1.0, -1.0, 0.5, -0.25, 0.0, 0.25, 0.5, 1.0, // Pattern for testing
			2.0, -2.0, 1.5, -0.75, 1.0, 0.75, -1.5, -1.0,
			0.1, -0.1, 0.05, -0.05, 0.025, -0.025, 0.0125, -0.0125,
			3.14159, -3.14159, 1.5708, -1.5708, 0.7854, -0.7854,
			0.3927, -0.3927, 0.19635, -0.19635, 0.098175, -0.098175,
		}
		for i := range original {
			original[i] = pattern[i%len(pattern)]
		}

		// Quantize to Q4_K
		q4kData, err := QuantizeWeightsToQ4K(original, len(original))
		if err != nil {
			t.Fatalf("the Q4_K quantizer failed: %v", err)
		}

		// Dequantize back
		dequantized := DequantizeQ4K(q4kData, len(original))
		if len(dequantized) != len(original) {
			t.Fatalf("Length mismatch after dequantization: got %d, want %d", len(dequantized), len(original))
		}

		// Check error bounds
		maxAbsError := float32(0.0)
		maxRelError := float32(0.0)
		threshold := float32(0.1) // 10% threshold for Q4_K

		for i, orig := range original {
			deq := dequantized[i]
			absError := float32(deq - orig)
			if absError < 0 {
				absError = -absError
			}

			relError := absError / (float32(orig) + 1e-6)
			if relError > maxRelError {
				maxRelError = relError
			}
			if absError > maxAbsError {
				maxAbsError = absError
			}

			// Log significant errors
			if relError > threshold {
				t.Logf("Index %d: orig=%f, deq=%f, rel_err=%.6f", i, orig, deq, relError)
			}
		}

		t.Logf("Q4_K quantization: max_abs_error=%.6f, max_rel_error=%.6f", maxAbsError, maxRelError)

		// Verify the error bounds. The K-quants give every group of 32
		// values a scale, so the step of the quantizer is the range of
		// the group over fifteen. The error of a value is bounded by
		// that step and not by its own magnitude: a small value next to
		// a large one has a large relative error and a small absolute
		// one.
		if float64(maxAbsError) > maxGroupStep(original) {
			t.Errorf("the absolute error is %f, want less than the step of the quantizer (%f)",
				maxAbsError, maxGroupStep(original))
		}
		if maxRelError > threshold {
			t.Logf("the relative error is %f, which is the price of a four-bit quantizer", maxRelError)
		}
	})

}
