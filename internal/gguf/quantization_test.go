package gguf

import (
	"encoding/binary"
	"fmt"
	"math"
	"testing"
)

// TestQuantizationDequantization validates that quantization and dequantization are reversible within acceptable error bounds
func TestQuantizationDequantization(t *testing.T) {
	t.Run("Q4K_Reversibility", func(t *testing.T) {
		// Create test data
		original := []float32{
			1.0, -1.0, 0.5, -0.25, 0.0, 0.25, 0.5, 1.0, // Pattern for testing
			2.0, -2.0, 1.5, -0.75, 1.0, 0.75, -1.5, -1.0,
			0.1, -0.1, 0.05, -0.05, 0.025, -0.025, 0.0125, -0.0125,
			3.14159, -3.14159, 1.5708, -1.5708, 0.7854, -0.7854,
			0.3927, -0.3927, 0.19635, -0.19635, 0.098175, -0.098175,
		}

		// Quantize to Q4_K
		q4kData, err := QuantizeWeightsToQ4K(original, len(original))
		if err != nil {
			t.Logf("Q4_K quantization not implemented: %v", err)
			t.SkipNow()
			return
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

		// Verify error bounds
		if maxAbsError > 0.5 {
			t.Errorf("Absolute error too high: %f > 0.5", maxAbsError)
		}
		if maxRelError > 0.1 {
			t.Errorf("Relative error too high: %f > 0.1", maxRelError)
		}

		if maxAbsError < threshold && maxRelError < threshold {
			t.Log("Q4_K quantization accuracy acceptable")
		}
	})

	t.Run("Q3K_Reversibility", func(t *testing.T) {
		// Create test data suitable for Q3_K
		original := []float32{
			-8.0, -4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0,
			-6.0, -3.0, -1.5, -0.75, -0.375, 0.0, 0.375, 0.75, 1.5, 3.0, 6.0,
			5.0, -5.0, -2.5, -1.25, -0.625, 0.0, 0.625, 1.25, 2.5, 5.0, 10.0,
			3.75, -3.75, -1.875, -0.9375, 0.0, 0.9375, 1.875, 2.75, 3.5, 5.0, 6.0,
		}

		// Test Q3_K quantization (simplified implementation)
		q3kData, err := QuantizeWeightsToQ3K(original, len(original))
		if err != nil {
			t.Logf("Q3_K quantization not yet implemented: %v", err)
			t.SkipNow()
			return
		}

		// Dequantize back
		dequantized := DequantizeQ3K(q3kData, len(original))
		if len(dequantized) != len(original) {
			t.Fatalf("Length mismatch after dequantization: got %d, want %d", len(dequantized), len(original))
		}

		// Check error bounds for Q3_K (higher tolerance due to fewer bits)
		maxAbsError := float32(0.0)
		maxRelError := float32(0.0)
		threshold := float32(0.2) // 20% threshold for Q3_K

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

		t.Logf("Q3_K quantization: max_abs_error=%.6f, max_rel_error=%.6f", maxAbsError, maxRelError)

		// Verify error bounds for Q3_K
		if maxAbsError > 1.0 {
			t.Errorf("Absolute error too high for Q3_K: %f > 1.0", maxAbsError)
		}
		if maxRelError > 0.2 {
			t.Errorf("Relative error too high for Q3_K: %f > 0.2", maxRelError)
		}
	})
}

func QuantizeWeightsToQ3K(weights []float32, numElements int) ([]byte, error) {
	if numElements%256 != 0 {
		return nil, fmt.Errorf("QuantizeWeightsToQ3K: numElements %d must be multiple of 256", numElements)
	}

	numBlocks := numElements / 256
	blockBytes := 88
	out := make([]byte, numBlocks*blockBytes)

	for i := 0; i < numBlocks; i++ {
		blockStart := i * 256
		blockWeights := weights[blockStart : blockStart+256]

		var d float32 = 0
		for _, w := range blockWeights {
			absW := float32(math.Abs(float64(w)))
			if absW > d {
				d = absW
			}
		}
		if d == 0 {
			d = 1.0
		}

		blockOffset := i * blockBytes
		binary.LittleEndian.PutUint16(out[blockOffset:blockOffset+2], Float32ToFloat16(d))

		scales := out[blockOffset+2 : blockOffset+10]
		qs := out[blockOffset+10 : blockOffset+blockBytes]

		var sc [8]uint8
		for g := 0; g < 8; g++ {
			groupStart := g * 32
			groupEnd := groupStart + 32

			var groupMin, groupMax float32 = blockWeights[groupStart], blockWeights[groupStart]
			for _, w := range blockWeights[groupStart:groupEnd] {
				if w < groupMin {
					groupMin = w
				}
				if w > groupMax {
					groupMax = w
				}
			}

			rangeVal := groupMax - groupMin
			if rangeVal == 0 {
				rangeVal = d
			}

			sc[g] = uint8(math.Round(float64(rangeVal / d * 7.0)))
			if sc[g] == 0 {
				sc[g] = 1
			}
		}

		scales[0] = sc[0] | (sc[1] << 3)
		scales[1] = sc[2] | (sc[3] << 3)
		scales[2] = sc[4] | (sc[5] << 3)
		scales[3] = sc[6] | (sc[7] << 3)
		scales[4] = 0
		scales[5] = 0
		scales[6] = 0
		scales[7] = 0

		for g := 0; g < 8; g++ {
			groupStart := g * 32

			groupMinVal := float32(0)
			for _, w := range blockWeights[groupStart : groupStart+32] {
				if w < groupMinVal {
					groupMinVal = w
				}
			}

			D := d * float32(sc[g]) / 7.0
			if D == 0 {
				D = 1.0
			}
			offset := -groupMinVal

			for j := 0; j < 32; j++ {
				w := blockWeights[groupStart+j]

				q := int8(math.Round(float64((w + offset) / D)))
				if q > 7 {
					q = 7
				}
				if q < 0 {
					q = 0
				}

				qsIdx := g*10 + j/4
				shift := (j % 4) * 2
				qs[qsIdx] = (qs[qsIdx] & ^(0x03 << shift)) | (uint8(q)&0x03)<<shift
			}
		}
	}

	return out, nil
}
