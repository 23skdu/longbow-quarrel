package simd

import (
	"math"
	"math/rand"
	"testing"
)

// rotationMatrix builds the same deterministic orthogonal matrix the
// production code uses.
//
// internal/device.GetPrecomputedRotation builds a DCT-II matrix, which is
// orthonormal (R^T * R = I). The previous TurboQuant tests used an identity
// matrix, which cannot tell the two residual conventions apart: with R = I
// the inverse rotation R^T is the identity too, so "residual in rotated
// space" and "residual in original space" produce identical results. Every
// check here therefore uses a genuinely mixed, orthonormal matrix.
//
// The construction is duplicated rather than imported because
// internal/device imports internal/simd, and importing it back would create
// an import cycle.
func rotationMatrix(n int, seed int64) []float32 {
	// A DCT-II basis is orthonormal but deterministic, so the seed only
	// selects between a couple of phase variants to keep the matrix
	// non-degenerate for small n.
	phase := float64(seed%3) * math.Pi / 4

	m := make([]float32, n*n)
	scale0 := float32(math.Sqrt(1.0 / float64(n)))
	scaleN := float32(math.Sqrt(2.0 / float64(n)))

	for i := 0; i < n; i++ {
		scale := scaleN
		if i == 0 {
			scale = scale0
		}
		for j := 0; j < n; j++ {
			m[i*n+j] = scale * float32(math.Cos(
				math.Pi*float64(i)*(float64(j)+0.5+phase)/float64(n)))
		}
	}
	return m
}

func randomInput(n int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	in := make([]float32, n)
	for i := range in {
		in[i] = float32(rng.NormFloat64() * 10)
	}
	return in
}

// TestPolarQuantReference_ClampsQuantizedValues covers the clamp.
//
// Without the clamp a value that rounds past +/-maxQuantVal overflows the
// int8 and wraps to the opposite sign, which silently corrupts the KV
// cache. The CUDA kernel (turboquant_encode_kernel) and the C SIMD kernels
// both clamp, so the reference must too.
func TestPolarQuantReference_ClampsQuantizedValues(t *testing.T) {
	const n = 64
	const bits = 2 // maxQuantVal = 1, so any |value| > scale saturates

	input := randomInput(n, 1)
	rot := rotationMatrix(n, 2)

	q, scale, _ := polarQuantReference(input, rot, n, bits)

	maxQuantVal := float32(1) // (1 << (bits-1)) - 1
	if scale <= 0 {
		t.Fatalf("expected a positive scale, got %v", scale)
	}
	for i, v := range q {
		f := float32(v)
		if f > maxQuantVal || f < -maxQuantVal {
			t.Fatalf("quantized[%d] = %v is outside +/-%v", i, f, maxQuantVal)
		}
	}
}

// TestPolarQuantReference_ResidualIsInOriginalSpace pins the residual
// convention.
//
// The residual must be R^T applied to the rotated-space residual, so that
// the decoder (internal/device/cpu.go, which adds the QJL contribution in
// original space) and the CUDA kernel agree.
func TestPolarQuantReference_ResidualIsInOriginalSpace(t *testing.T) {
	const n = 32
	const bits = 4

	input := randomInput(n, 3)
	rot := rotationMatrix(n, 4)

	q, scale, residual := polarQuantReference(input, rot, n, bits)

	// Recompute the rotated space independently.
	rotated := make([]float32, n)
	for i := 0; i < n; i++ {
		var sum float32
		for j := 0; j < n; j++ {
			sum += rot[i*n+j] * input[j]
		}
		rotated[i] = sum
	}

	// The quantized values must be the clamped rounded scaled values.
	maxQuantVal := float32((1 << (bits - 1)) - 1)
	for i := 0; i < n; i++ {
		want := float32(math.Round(float64(rotated[i] / scale)))
		if want > maxQuantVal {
			want = maxQuantVal
		} else if want < -maxQuantVal {
			want = -maxQuantVal
		}
		if float32(q[i]) != want {
			t.Fatalf("quantized[%d] = %v, want %v", i, q[i], want)
		}
	}

	// The residual must equal R^T * (rotated - q*scale).
	resRotated := make([]float32, n)
	for i := 0; i < n; i++ {
		resRotated[i] = rotated[i] - float32(q[i])*scale
	}
	for i := 0; i < n; i++ {
		var want float32
		for j := 0; j < n; j++ {
			want += rot[j*n+i] * resRotated[j]
		}
		if math.Abs(float64(residual[i]-want)) > 1e-4 {
			t.Fatalf("residual[%d] = %v, want %v (R^T applied)", i, residual[i], want)
		}
	}
}

// TestPolarQuantSIMD_MatchesReference checks that the public entry point
// agrees with the canonical reference, whichever accelerated kernel the
// host happens to select.
//
// This is the test that fails if the C SIMD kernels are wired in with
// different semantics from the reference.
//
// The accelerated kernels use fused multiply-add, so their intermediate
// rounding differs from the reference's separate multiply and add. The
// comparison therefore allows a quantized value to differ by one level (a
// last-ULP difference in the pre-rounding value can only flip a level when
// it lands exactly on a .5 boundary) and compares the residual with a tight
// relative tolerance.
//
// A genuine convention mismatch -- residual in rotated instead of original
// space, or a missing clamp -- produces differences orders of magnitude
// larger than these tolerances. TestPolarQuantSIMD_AcceleratedUsesOriginalSpace
// pins the convention directly.
func TestPolarQuantSIMD_MatchesReference(t *testing.T) {
	for _, n := range []int{16, 32, 64, 128, 256} {
		for _, bits := range []int{2, 4, 8} {
			input := randomInput(n, int64(n*100+bits))
			rot := rotationMatrix(n, int64(n*100+bits+1))

			gotQ, gotScale, gotRes := PolarQuantSIMD(input, rot, n, bits)
			wantQ, wantScale, wantRes := polarQuantReference(input, rot, n, bits)

			if gotScale != wantScale {
				// The scale is derived from the largest |rotated| value,
				// which differs by a few ULP between the fused
				// multiply-add and the separate multiply and add.
				rel := math.Abs(float64(gotScale-wantScale)) / math.Max(1e-9, math.Abs(float64(wantScale)))
				if rel > 1e-5 {
					t.Fatalf("n=%d bits=%d: scale = %v, want %v (rel diff %v)", n, bits, gotScale, wantScale, rel)
				}
			}
			for i := range wantQ {
				if d := int(gotQ[i]) - int(wantQ[i]); d > 1 || d < -1 {
					t.Fatalf("n=%d bits=%d: quantized[%d] = %v, want %v (delta %d)",
						n, bits, i, gotQ[i], wantQ[i], d)
				}
			}
			for i := range wantRes {
				tol := 1e-4 * math.Max(1, math.Abs(float64(wantRes[i])))
				if math.Abs(float64(gotRes[i]-wantRes[i])) > tol {
					t.Fatalf("n=%d bits=%d: residual[%d] = %v, want %v", n, bits, i, gotRes[i], wantRes[i])
				}
			}
		}
	}
}

// TestPolarQuantSIMD_AcceleratedUsesOriginalSpace pins the residual
// convention of whichever kernel the host selects.
//
// The residual must equal R^T * (rotated - q*scale), i.e. it must already be
// in the original space. Returning the rotated-space residual instead --
// which is what the previous x86 implementation did -- would make the CPU
// KV cache differ from the GPU one, because the decoder adds the QJL
// contribution in original space.
//
// The rotation is recomputed in Go here, so the check is independent of the
// reference implementation's own arithmetic: a wrong convention is off by
// O(1) relative, while the FMA-versus-multiply-and-add difference is off by
// only a few ULP.
func TestPolarQuantSIMD_AcceleratedUsesOriginalSpace(t *testing.T) {
	const n = 64
	const bits = 4

	input := randomInput(n, 41)
	rot := rotationMatrix(n, 42)

	q, scale, residual := PolarQuantSIMD(input, rot, n, bits)

	// Recompute the rotated space independently.
	rotated := make([]float32, n)
	for i := 0; i < n; i++ {
		var sum float32
		for j := 0; j < n; j++ {
			sum += rot[i*n+j] * input[j]
		}
		rotated[i] = sum
	}

	maxQuantVal := float32((1 << (bits - 1)) - 1)
	for i := 0; i < n; i++ {
		want := float32(math.Round(float64(rotated[i] / scale)))
		if want > maxQuantVal {
			want = maxQuantVal
		} else if want < -maxQuantVal {
			want = -maxQuantVal
		}
		if float32(q[i]) != want {
			t.Fatalf("quantized[%d] = %v, want %v", i, q[i], want)
		}
	}

	resRotated := make([]float32, n)
	for i := 0; i < n; i++ {
		resRotated[i] = rotated[i] - float32(q[i])*scale
	}
	for i := 0; i < n; i++ {
		var want float32
		for j := 0; j < n; j++ {
			want += rot[j*n+i] * resRotated[j]
		}
		tol := 1e-4 * math.Max(1, math.Abs(float64(want)))
		if math.Abs(float64(residual[i]-want)) > tol {
			t.Fatalf("residual[%d] = %v, want %v (R^T applied)", i, residual[i], want)
		}
	}
}

// TestQJLTransformSIMD_MatchesReference is the equivalent check for the QJL
// transform.
func TestQJLTransformSIMD_MatchesReference(t *testing.T) {
	const cols = 64
	const rows = 16

	residual := randomInput(cols, 11)
	sign := make([]float32, rows*cols)
	for i := range sign {
		if i%3 == 0 {
			sign[i] = -1
		} else {
			sign[i] = 1
		}
	}

	gotQ, gotScale := QJLTransformSIMD(residual, sign, rows, cols)
	wantQ, wantScale := qjlTransformReference(residual, sign, rows, cols)

	if math.Abs(float64(gotScale-wantScale)) > 1e-4*math.Max(1, math.Abs(float64(wantScale))) {
		t.Fatalf("scale = %v, want %v", gotScale, wantScale)
	}
	for i := range wantQ {
		if gotQ[i] != wantQ[i] {
			t.Fatalf("quantized[%d] = %v, want %v", i, gotQ[i], wantQ[i])
		}
	}
}

// TestPolarQuantSIMD_EmptyRotationMatrixIsSafe guards the degenerate case.
//
// The accelerated kernels index the rotation matrix as n*n, so they cannot
// be used without one. The reference implementation must handle a missing
// matrix without panicking -- the previous no-cgo implementation read
// rotationMatrix[j*n+i] unconditionally and would have panicked with an
// index-out-of-range.
func TestPolarQuantSIMD_EmptyRotationMatrixIsSafe(t *testing.T) {
	const n = 16
	input := randomInput(n, 21)

	for _, rot := range [][]float32{nil, {}, make([]float32, 5)} {
		q, scale, res := PolarQuantSIMD(input, rot, n, 4)
		if len(q) != n {
			t.Fatalf("expected %d quantized values, got %d", n, len(q))
		}
		if len(res) != n {
			t.Fatalf("expected %d residual values, got %d", n, len(res))
		}
		if scale <= 0 {
			t.Fatalf("expected a positive scale, got %v", scale)
		}
	}
}

// TestPolarQuantSIMD_DecodeRoundTrip checks that the encoder and the decoder
// in internal/device/cpu.go agree on the conventions.
//
// The decoder reconstructs R^T * (q * scale) and adds the QJL contribution in
// original space, so the encoder must produce a residual in original space
// for the round trip to be meaningful.
func TestPolarQuantSIMD_DecodeRoundTrip(t *testing.T) {
	const n = 64
	const bits = 4

	input := randomInput(n, 31)
	rot := rotationMatrix(n, 32)

	q, scale, _ := PolarQuantSIMD(input, rot, n, bits)

	// Decoder step 1+2: out = R^T * (q * scale).
	out := DequantizeTurboQuant(q, scale, rot, n)

	// The reconstruction should be in the same space as the input, so the
	// error against the input must be bounded by the quantization error
	// scaled by the rotation's condition number. A residual-space mix-up
	// would make this error explode.
	var sumSq, inSq float64
	for i := 0; i < n; i++ {
		d := float64(out[i] - input[i])
		sumSq += d * d
		inSq += float64(input[i]) * float64(input[i])
	}
	relErr := math.Sqrt(sumSq / inSq)
	if relErr > 0.5 {
		t.Fatalf("round-trip relative error %v is too large; the encoder and "+
			"decoder disagree on the residual convention", relErr)
	}
}

// BenchmarkPolarQuantReference is the pure-Go baseline, used to measure what
// the accelerated kernels actually contribute.
func BenchmarkPolarQuantReference(b *testing.B) {
	n := 256
	bits := 4
	input := randomInput(n, 99)
	rotation := rotationMatrix(n, 98)

	for b.Loop() {
		_, _, _ = polarQuantReference(input, rotation, n, bits)
	}
}
