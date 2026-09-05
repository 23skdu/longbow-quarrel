package simd

import (
	"math"
	"testing"
)

func FuzzPolarQuant(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, uint8(4))
	f.Add([]byte{255, 128, 64, 32, 16, 8, 4, 2, 1, 0, 10, 20, 30, 40, 50, 60}, uint8(2))
	f.Add([]byte{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, uint8(8))

	f.Fuzz(func(t *testing.T, data []byte, bitsByte uint8) {
		if len(data) < 16 {
			return
		}
		bits := int(bitsByte%7) + 2 // 2 to 8 bits
		n := 16
		input := make([]float32, n)
		rot := make([]float32, n*n)
		for i := 0; i < n; i++ {
			input[i] = float32(int8(data[i])) / 10.0
			rot[i*n+i] = 1.0 // Identity matrix
		}

		q, scale, res := PolarQuantSIMD(input, rot, n, bits)

		if len(q) != n || len(res) != n {
			t.Fatalf("length mismatch")
		}
		if math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
			t.Fatalf("scale is NaN or Inf")
		}
		if scale < 0 {
			t.Fatalf("scale must be non-negative, got %v", scale)
		}

		maxVal := int8((1 << (bits - 1)) - 1)
		for i := 0; i < n; i++ {
			if q[i] > maxVal || q[i] < -maxVal {
				t.Fatalf("quantized[%d]=%d out of bounds [-%d, %d]", i, q[i], maxVal, maxVal)
			}
			if math.IsNaN(float64(res[i])) {
				t.Fatalf("residual[%d] is NaN", i)
			}
		}
	})
}

func FuzzQJLTransform(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4, 5, 6, 7, 8}, []byte{1, 2, 3, 4, 5, 6, 7, 8})

	f.Fuzz(func(t *testing.T, resBytes []byte, signBytes []byte) {
		cols := 8
		rows := 4
		if len(resBytes) < cols || len(signBytes) < rows*cols {
			return
		}

		res := make([]float32, cols)
		for i := 0; i < cols; i++ {
			res[i] = float32(int8(resBytes[i])) / 5.0
		}

		sign := make([]float32, rows*cols)
		for i := 0; i < rows*cols; i++ {
			if signBytes[i]%2 == 0 {
				sign[i] = 1.0
			} else {
				sign[i] = -1.0
			}
		}

		quantized, scale := QJLTransformSIMD(res, sign, rows, cols)

		if len(quantized) != rows {
			t.Fatalf("len(quantized) = %d != %d", len(quantized), rows)
		}
		if math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) || scale < 0 {
			t.Fatalf("invalid scale %v", scale)
		}
		for i, v := range quantized {
			if v != 1 && v != -1 {
				t.Fatalf("quantized[%d] = %d != +-1", i, v)
			}
		}
	})
}
