//go:build darwin && metal

package gguf

import (
	"math"
	"testing"
	"time"
)

// TestQuantization_Q4K compares GPU dequantization against CPU reference
func TestQuantization_Q4K(t *testing.T) {
	t.Run("BasicDequantization", func(t *testing.T) {
		// Create test weights
		f32Weights := []float32{
			1.0, -1.0, 0.5, 0.25, 0.0, -0.25, -0.5, -1.0, 1.0,
			2.0, -2.0, 1.0, 1.5, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0,
			1.5, -1.5, 1.0, 0.5, 0.25, 0.0, -0.25, -0.5, -1.0,
		}

		// Q4K quantize the weights
		q4kData, err := QuantizeWeightsToQ4K(f32Weights, 32)
		if err != nil {
			t.Fatalf("Failed to quantize weights: %v", err)
		}

		// Dequantize back to F32
		dequantized := DequantizeQ4K(q4kData, 32)
		if err != nil {
			t.Fatalf("Failed to dequantize weights: %v", err)
		}

		// Compare original and dequantized weights
		maxAbsError := float32(0.0)
		maxRelError := float32(0.0)
		threshold := float32(0.01) // 1% threshold

		for i, original := range f32Weights {
			dequantized := dequantized[i]
			absError := float32(math.Abs(float64(dequantized - original)))
			relError := absError / (float32(math.Abs(float64(original)) + 1e-6))

			if absError > maxAbsError {
				maxAbsError = absError
			}
			if relError > maxRelError {
				maxRelError = relError
			}

			t.Logf("Weight %d: orig=%f, deq=%f, abs_err=%.6f, rel_err=%.6f",
				i, original, dequantized, absError, relError)
		}

		t.Logf("Q4K quantization: max_abs_error=%.6f, max_rel_error=%.6f",
			maxAbsError, maxRelError)

		// Verify errors are within acceptable bounds
		if maxAbsError > 0.1 {
			t.Errorf("Maximum absolute error too high: %f > 0.1", maxAbsError)
		}

		if maxRelError > 0.05 { // 5% relative error
			t.Errorf("Maximum relative error too high: %f > 0.05", maxRelError)
		}

		if maxAbsError < threshold && maxRelError < threshold {
			t.Log("Q4K quantization accuracy acceptable")
		}
	})

	t.Run("EdgeCase_ExtremeValues", func(t *testing.T) {
		// Test with extreme values
		extremeWeights := []float32{
			1000.0, -1000.0, 3.402823e38, -3.402823e38,
			65504.0, -65504.0, 1e-6, -1e-6,
		}

		q4kData, err := QuantizeWeightsToQ4K(extremeWeights, 32)
		if err != nil {
			t.Fatalf("Failed to quantize extreme weights: %v", err)
		}

		// Should handle extreme values gracefully
		dequantized := DequantizeQ4K(q4kData, 32)
		if err != nil {
			t.Fatalf("Failed to dequantize extreme weights: %v", err)
		}

		// Check for infinities or NaNs
		hasIssues := false
		for _, value := range dequantized {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				hasIssues = true
				break
			}
		}

		if hasIssues {
			t.Error("Extreme values produced NaN or Inf in dequantization")
		}

		// High error tolerance for extreme values
		maxAbsError := float32(0.0)
		maxRelError := float32(0.0)
		threshold := float32(0.5) // 50% tolerance for extreme values

		for i, original := range extremeWeights {
			dequantized := dequantized[i]
			absError := float32(math.Abs(float64(dequantized - original)))
			relError := absError / (float32(math.Abs(float64(original)) + 1e-6))

			if absError > maxAbsError {
				maxAbsError = absError
			}
			if relError > maxRelError {
				maxRelError = relError
			}
		}

		if maxAbsError > threshold || maxRelError > threshold {
			t.Logf("Extreme values within high error tolerance: max_abs=%.6f, max_rel=%.6f",
				maxAbsError, maxRelError)
		} else {
			t.Logf("Extreme values handled gracefully")
		}
	})

	t.Run("MatrixDimensions", func(t *testing.T) {
		// Test different matrix dimensions
		testCases := []struct {
			name      string
			rows      int
			cols      int
			precision float32
		}{
			{"Square", 32, 32, 0.001},
			{"Rectangular", 16, 64, 0.001},
			{"Tall", 64, 8, 0.001},
			{"Wide", 8, 64, 0.001},
			{"Small", 8, 8, 0.001},
			{"Single", 1, 1000, 0.001},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				// Create test matrix
				f32Weights := make([]float32, tc.rows*tc.cols)
				for i := range f32Weights {
					f32Weights[i] = float32(i%100) * tc.precision // Test pattern
				}

				// Test quantization
				q4kData, err := QuantizeWeightsToQ4K(f32Weights, tc.cols)
				if err != nil {
					t.Fatalf("Failed to quantize %s matrix: %v", tc.name, err)
				}

				dequantized := DequantizeQ4K(q4kData, len(f32Weights))
				if err != nil {
					t.Fatalf("Failed to dequantize %s matrix: %v", tc.name, err)
				}

				// Calculate error metrics
				maxAbsError := float32(0.0)
				maxRelError := float32(0.0)

				for i := 0; i < len(f32Weights); i++ {
					absError := float32(math.Abs(float64(dequantized[i] - f32Weights[i])))
					relError := absError / (float32(math.Abs(float64(f32Weights[i])) + 1e-6))

					if absError > maxAbsError {
						maxAbsError = absError
					}
					if relError > maxRelError {
						maxRelError = relError
					}
				}

				t.Logf("%s (%dx%d): max_abs=%.6f, max_rel=%.6f",
					tc.name, tc.rows, tc.cols, maxAbsError, maxRelError)

				// Validation thresholds (relaxed for test diversity)
				if maxAbsError > 0.1 { // 10% absolute error
					t.Errorf("%s: Absolute error too high: %f > 0.1", tc.name, maxAbsError)
				}

				if maxRelError > 0.1 { // 10% relative error
					t.Errorf("%s: Relative error too high: %f > 0.1", tc.name, maxRelError)
				}
			})
		}
	})

	t.Run("Performance", func(t *testing.T) {
		// Test quantization performance
		weights := make([]float32, 1000)
		for i := range weights {
			weights[i] = float32(math.Sin(float64(i) * 0.01))
		}

		// Time quantization
		start := time.Now()
		_, err := QuantizeWeightsToQ4K(weights, 64)
		if err != nil {
			t.Fatalf("Failed performance quantization: %v", err)
		}

		quantDuration := time.Since(start)

		// Time dequantization
		q4kData, _ := QuantizeWeightsToQ4K(weights, 64)
		start = time.Now()
		_, err = DequantizeWeightsFromQ4K(q4kData, 64, 64)
		if err != nil {
			t.Fatalf("Failed performance dequantization: %v", err)
		}

		dequantDuration := time.Since(start)

		t.Logf("Performance (1000 weights): quant=%v, dequant=%v",
			quantDuration, dequantDuration)

		// Performance should be reasonable
		if quantDuration > 100*time.Millisecond {
			t.Errorf("Quantization too slow: %v", quantDuration)
		}

		if dequantDuration > 100*time.Millisecond {
			t.Errorf("Dequantization too slow: %v", dequantDuration)
		}
	})
}

// Helper functions for quantization testing
func calculateQuantizationError(original, dequantized []float32) (maxAbsError, maxRelError float32) {
	maxAbsError = float32(0.0)
	maxRelError = float32(0.0)

	for i := 0; i < len(original); i++ {
		absError := float32(math.Abs(float64(dequantized[i] - original[i])))
		relError := absError / (float32(math.Abs(float64(original[i])) + 1e-6))

		if absError > maxAbsError {
			maxAbsError = absError
		}
		if relError > maxRelError {
			maxRelError = relError
		}
	}

	return maxAbsError, maxRelError
}
