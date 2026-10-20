//go:build !amd64 || !cgo

package simd

import "github.com/23skdu/longbow-quarrel/internal/mathx"

// The scalar reference implementations of the kernels.
//
// The x86 builds have their own copies - avx2_scalar.go for the AVX2
// build and avx512.go for the AVX-512 one - and the tests refer to these
// three by name. Without them the test suite did not compile at all on
// arm64 or without cgo, so the NEON path and the pure-Go fallback could
// never be tested.
//
// The implementations are the same as the x86 ones: they are the
// reference the vector kernels are compared against.

// softmaxScalar computes the softmax of x in place.
func softmaxScalar(x []float32) {
	if len(x) == 0 {
		return
	}

	max := x[0]
	for _, v := range x {
		if v > max {
			max = v
		}
	}

	sum := float32(0.0)
	for i := range x {
		x[i] = fastExp(x[i] - max)
		sum += x[i]
	}

	if sum > 0 {
		invSum := float32(1.0) / sum
		for i := range x {
			x[i] *= invSum
		}
	}
}

// swigluScalar computes the gated linear unit activation.
func swigluScalar(gate, up, out []float32) {
	for i := 0; i < len(gate); i++ {
		g := gate[i]
		if g > 10.0 {
			g = 10.0
		}
		if g < -10.0 {
			g = -10.0
		}
		sigmoid := float32(1.0) / (float32(1.0) + fastExp(-g))
		out[i] = up[i] * g * sigmoid
	}
}

// fp32ToFp16Scalar converts a slice of float32 to a slice of float16.
func fp32ToFp16Scalar(src []float32, dst []uint16) {
	for i := 0; i < len(src); i++ {
		dst[i] = fp32ToFp16(src[i])
	}
}

// fastExp is the fast approximate exponential used by the scalar
// fallbacks.
func fastExp(x float32) float32 {
	return float32(mathx.Exp(float64(x)))
}
