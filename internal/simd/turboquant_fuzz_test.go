package simd

import (
	"math"
	"testing"
)

func FuzzPolarQuant(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, uint8(4))
	f.Add([]byte{255, 128, 64, 32, 16, 8, 4, 2, 1, 0, 10, 20, 30, 40, 50, 60}, uint8(2))
	f.Add([]byte{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, uint8(8))
	f.Add([]byte{7, 249, 7, 253, 3, 253, 11, 245, 1, 255, 9, 247, 5, 251, 13, 243}, uint8(8))

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
		}
		// A mixed rotation matrix, not the identity: with R = I the inverse
		// rotation is a no-op, so a residual-space mismatch between the
		// accelerated kernels and the reference would go unnoticed. The
		// entries are derived from the fuzz input so the fuzzer can explore
		// different matrices.
		//
		// Each column is normalised to unit L1 norm so the inverse rotation
		// cannot amplify a one-level quantization difference without bound;
		// that keeps the comparison below meaningful for an arbitrary
		// (non-orthonormal) matrix.
		for i := range rot {
			rot[i] = float32(int8(data[i%len(data)])) / 32.0
		}
		for c := 0; c < n; c++ {
			var l1 float64
			for r := 0; r < n; r++ {
				l1 += math.Abs(float64(rot[r*n+c]))
			}
			if l1 == 0 {
				rot[c*n+c] = 1
				continue
			}
			for r := 0; r < n; r++ {
				rot[r*n+c] = float32(float64(rot[r*n+c]) / l1)
			}
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

		// The accelerated kernels must agree with the canonical reference.
		// Without this, a convention mismatch (residual in rotated vs
		// original space) would pass every other assertion here.
		//
		// The comparison is done on the *decoded* vector rather than on the
		// internals: the accelerated kernels use fused multiply-add, so the
		// pre-rounding value can differ by an ULP, which can flip a
		// quantized value by one level where it lands on a .5 boundary. A
		// convention mismatch moves the decoded vector by O(1) relative,
		// far beyond the bound used here.
		wantQ, wantScale, _ := polarQuantReference(input, rot, n, bits)
		if math.Abs(float64(scale-wantScale)) > 1e-4*math.Max(1, math.Abs(float64(wantScale))) {
			t.Fatalf("scale = %v, reference = %v", scale, wantScale)
		}
		for i := range wantQ {
			if d := int(q[i]) - int(wantQ[i]); d > 1 || d < -1 {
				t.Fatalf("quantized[%d] = %d, reference = %d (delta %d)", i, q[i], wantQ[i], d)
			}
		}

		gotDecoded := DequantizeTurboQuant(q, scale, rot, n)
		wantDecoded := DequantizeTurboQuant(wantQ, wantScale, rot, n)
		// A one-level difference in a quantized value propagates through the
		// inverse rotation, so the bound for element i is the scale times
		// the weighted L1 norm of column i, using the actual per-element
		// differences.
		for i := range gotDecoded {
			var bound float64
			for j := 0; j < n; j++ {
				d := math.Abs(float64(q[j]) - float64(wantQ[j]))
				bound += float64(wantScale) * math.Abs(float64(rot[j*n+i])) * d
			}
			bound += 1e-4 * math.Max(1, math.Abs(float64(wantDecoded[i])))
			if math.Abs(float64(gotDecoded[i]-wantDecoded[i])) > bound {
				t.Fatalf("decoded[%d] = %v, reference = %v (bound %v)",
					i, gotDecoded[i], wantDecoded[i], bound)
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
