package simd

import (
	"math"
	"testing"
)

// The TurboQuant encoder rounds half away from zero, so that it agrees
// with the CUDA kernel, with C's roundf() and with Go's math.Round. The
// AVX2 and AVX-512 kernels implement that with an explicit correction
// over vrndnq, which rounds half to even.
//
// The NEON kernel had the same correction, but its absolute-value mask
// was built as vdupq_n_f32(0x7FFFFFFF) - an integer literal converted to
// a float, which is 2147483648.0f and has the bit pattern 0x4F000000,
// not 0x7FFFFFFF. The mask therefore kept the sign bit and mangled the
// exponent, the tie was never detected, and the kernel silently rounded
// half to even. It did not compile at all, so nothing noticed.
//
// The test below pins the behaviour at the level the caller can see: a
// value that lands exactly on a half must round away from zero.

// TestPolarQuantSIMD_RoundsHalfAwayFromZero checks the tie-breaking on
// the value the encoder quantises.
func TestPolarQuantSIMD_RoundsHalfAwayFromZero(t *testing.T) {
	// The identity rotation, so the rotated value is the input. It has
	// to be at least four elements wide: the vector kernels process
	// four at a time and fall back to a scalar tail below that, and the
	// scalar tail was always correct.
	identity := []float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}

	cases := []struct {
		name  string
		input []float32
		want  int8
	}{
		// The maximum absolute value is 1, so with two bits the scale is
		// 1 and 0.5 is exactly half a quantum.
		{"positive tie", []float32{1, 0.5, 0.25, 0.125}, 1},
		{"negative tie", []float32{-1, -0.5, -0.25, -0.125}, -1},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			quantized, _, _ := PolarQuantSIMD(tc.input, identity, len(tc.input), 2)
			if quantized[1] != tc.want {
				t.Errorf("the quantised value is %d, want %d: a tie must round away from zero",
					quantized[1], tc.want)
			}
		})
	}
}

// TestPolarQuantSIMD_TieMatchesTheReference checks the tie-breaking
// against the pure-Go reference, which uses math.Round.
func TestPolarQuantSIMD_TieMatchesTheReference(t *testing.T) {
	identity := []float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
	input := []float32{1, 0.5, 0.25, 0.125}

	got, _, _ := PolarQuantSIMD(input, identity, len(input), 2)
	want, _, _ := polarQuantReference(input, identity, len(input), 2)

	if got[1] != want[1] {
		t.Errorf("the SIMD path rounded the tie to %d, the reference to %d",
			got[1], want[1])
	}
	if want[1] != 1 {
		t.Errorf("the reference rounded the tie to %d, want 1: a tie must round away from zero",
			want[1])
	}
}

// TestPolarQuantSIMD_ScaleIsExactOnAPowerOfTwo checks the scale the
// encoder derives, which the tie case depends on.
func TestPolarQuantSIMD_ScaleIsExactOnAPowerOfTwo(t *testing.T) {
	identity := []float32{1, 0, 0, 1}
	_, scale, _ := PolarQuantSIMD([]float32{1, 0.5}, identity, 2, 2)
	if math.Abs(float64(scale-1)) > 1e-6 {
		t.Errorf("the scale is %v, want 1", scale)
	}
}
