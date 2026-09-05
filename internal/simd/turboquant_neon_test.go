//go:build arm64

package simd

import (
	"math"
	"math/rand"
	"testing"
)

func TestNEONPolarQuant(t *testing.T) {
	testSizes := []int{16, 32, 64, 128, 256, 67, 125} // includes non-multiple of 4 for tail handling
	bits := 4

	for _, n := range testSizes {
		t.Run("Size_"+string(rune(n)), func(t *testing.T) {
			input := make([]float32, n)
			rotation := make([]float32, n*n)
			for i := 0; i < n; i++ {
				input[i] = float32(i+1) / 10.0
				rotation[i*n+i] = 1.0 // Identity matrix
			}

			q, scale, res := PolarQuantSIMD(input, rotation, n, bits)

			if len(q) != n {
				t.Fatalf("expected len(q)=%d, got %d", n, len(q))
			}
			if len(res) != n {
				t.Fatalf("expected len(res)=%d, got %d", n, len(res))
			}
			if scale <= 0 {
				t.Fatalf("expected positive scale, got %v", scale)
			}

			// For identity matrix, rotated == input.
			// Verify residual is within expected quantization error bounds (<= scale / 2)
			maxAllowedErr := scale*0.6 + 1e-4
			for i := 0; i < n; i++ {
				if math.Abs(float64(res[i])) > float64(maxAllowedErr) {
					t.Errorf("residual[%d] = %v exceeds max error %v (scale %v)", i, res[i], maxAllowedErr, scale)
				}
			}

			// Verify reconstruction
			rec := DequantizeTurboQuant(q, scale, rotation, n)
			for i := 0; i < n; i++ {
				diff := math.Abs(float64(rec[i] - input[i]))
				if diff > float64(maxAllowedErr) {
					t.Errorf("reconstructed[%d]=%v vs input[%d]=%v diff %v > %v", i, rec[i], i, input[i], diff, maxAllowedErr)
				}
			}
		})
	}
}

func TestNEONQJLTransform(t *testing.T) {
	testCases := []struct {
		rows int
		cols int
	}{
		{rows: 32, cols: 64},
		{rows: 64, cols: 128},
		{rows: 16, cols: 33}, // Odd tail for NEON (cols not multiple of 4)
	}

	for _, tc := range testCases {
		residual := make([]float32, tc.cols)
		signMatrix := make([]float32, tc.rows*tc.cols)

		for i := range residual {
			residual[i] = float32(i%7 - 3)
		}
		for i := range signMatrix {
			if i%2 == 0 {
				signMatrix[i] = 1.0
			} else {
				signMatrix[i] = -1.0
			}
		}

		quantized, scale := QJLTransformSIMD(residual, signMatrix, tc.rows, tc.cols)

		if len(quantized) != tc.rows {
			t.Fatalf("expected quantized len %d, got %d", tc.rows, len(quantized))
		}
		if scale < 0 {
			t.Fatalf("negative scale: %v", scale)
		}
		for i, q := range quantized {
			if q != 1 && q != -1 {
				t.Errorf("quantized[%d] must be +1 or -1, got %d", i, q)
			}
		}
	}
}

func BenchmarkNEONPolarQuant_64(b *testing.B) {
	benchmarkNEONPolarQuant(b, 64)
}

func BenchmarkNEONPolarQuant_128(b *testing.B) {
	benchmarkNEONPolarQuant(b, 128)
}

func BenchmarkNEONPolarQuant_256(b *testing.B) {
	benchmarkNEONPolarQuant(b, 256)
}

func benchmarkNEONPolarQuant(b *testing.B, n int) {
	input := make([]float32, n)
	rot := make([]float32, n*n)
	for i := range input {
		input[i] = rand.Float32()
	}
	for i := 0; i < n; i++ {
		rot[i*n+i] = 1.0
	}

	b.ResetTimer()
	for b.Loop() {
		_, _, _ = PolarQuantSIMD(input, rot, n, 4)
	}
}

func BenchmarkNEONQJLTransform_64x128(b *testing.B) {
	rows, cols := 64, 128
	res := make([]float32, cols)
	sign := make([]float32, rows*cols)
	for i := range res {
		res[i] = rand.Float32()
	}
	for i := range sign {
		sign[i] = 1.0
	}

	b.ResetTimer()
	for b.Loop() {
		_, _ = QJLTransformSIMD(res, sign, rows, cols)
	}
}
