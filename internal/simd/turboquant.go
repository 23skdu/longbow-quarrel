package simd

import (
	"github.com/23skdu/longbow-quarrel/internal/mathx"
)

// TurboQuantType represents the bit depth for TurboQuant encoding
type TurboQuantType int

const (
	TurboQuant2 TurboQuantType = 2 // 2-bit + 1-bit QJL residual
	TurboQuant4 TurboQuantType = 4 // 4-bit + 1-bit QJL residual
	TurboQuant8 TurboQuantType = 8 // 8-bit + 1-bit QJL residual
)

func (t TurboQuantType) String() string {
	switch t {
	case TurboQuant2:
		return "TurboQuant2"
	case TurboQuant4:
		return "TurboQuant4"
	case TurboQuant8:
		return "TurboQuant8"
	default:
		return "TurboQuant(unknown)"
	}
}

// PolarQuantSIMD performs the PolarQuant rotation + quantization step of
// TurboQuant encoding.
//
// It returns the quantized values, the scale, and the residual.
//
// # Semantics
//
// The residual is returned in the **original (un-rotated) space**, i.e. it is
// R^T applied to the rotated-space residual. This matches:
//
//   - the CUDA kernel `turboquant_encode_kernel` in internal/device/cuda_kernels.cu
//   - the C SIMD kernels polar_quant_avx512/avx2/neon
//   - the decoder in internal/device/cpu.go, which adds the QJL contribution
//     in original space
//
// An earlier revision of this file returned the residual in rotated space and
// did not clamp the quantized value. That made the x86 CPU path produce a
// different KV cache from the GPU path for the same input, because the two
// disagreed on both the clamp and the residual space. The three copies of
// this function (amd64+cgo, arm64+cgo, and the no-cgo fallback) had drifted
// apart from each other in exactly that way; they are now consolidated here
// so there is a single definition.
//
// The accelerated implementations live in the build-tagged files and must
// reproduce these semantics exactly; internal/simd/turboquant_parity_test.go
// enforces that.
func PolarQuantSIMD(input []float32, rotationMatrix []float32, n int, bits int) ([]int8, float32, []float32) {
	if n <= 0 || len(input) < n {
		return nil, 0, nil
	}
	if bits < 2 {
		bits = 2
	}
	// The accelerated kernels index the rotation matrix as n*n; they cannot
	// be used with a short or empty matrix.
	if len(rotationMatrix) >= n*n {
		if q, s, res, ok := polarQuantAccelerated(input, rotationMatrix, n, bits); ok {
			return q, s, res
		}
	}
	return polarQuantReference(input, rotationMatrix, n, bits)
}

// polarQuantReference is the canonical pure-Go PolarQuant implementation.
//
// It is the definition of the algorithm; the accelerated kernels must match
// it bit for bit on the quantized output.
func polarQuantReference(input []float32, rotationMatrix []float32, n int, bits int) ([]int8, float32, []float32) {
	rotated := make([]float32, n)
	maxAbs := float32(0.0)

	hasMatrix := len(rotationMatrix) >= n*n

	if !hasMatrix {
		// No usable rotation matrix: the input is already in "rotated" space.
		copy(rotated, input[:n])
		for _, x := range rotated {
			if a := float32(mathx.Abs(float64(x))); a > maxAbs {
				maxAbs = a
			}
		}
	} else {
		for i := 0; i < n; i++ {
			var sum float32
			for j := 0; j < n; j++ {
				sum += rotationMatrix[i*n+j] * input[j]
			}
			rotated[i] = sum
			if a := float32(mathx.Abs(float64(sum))); a > maxAbs {
				maxAbs = a
			}
		}
	}

	shiftAmount := uint(bits - 1)
	maxQuantInt := (1 << shiftAmount) - 1
	maxQuantVal := float32(maxQuantInt)
	scale := maxAbs / maxQuantVal
	if scale == 0 {
		scale = 1.0
	}

	inverseScale := 1.0 / scale
	quantized := make([]int8, n)
	resRotated := make([]float32, n)

	for i := 0; i < n; i++ {
		q := float32(mathx.Round(float64(rotated[i] * inverseScale)))
		// Clamp to the representable range, matching the CUDA kernel and the
		// C SIMD kernels. Without this, a value that rounds past the limit
		// would overflow the int8 and wrap to the opposite sign.
		if q > maxQuantVal {
			q = maxQuantVal
		} else if q < -maxQuantVal {
			q = -maxQuantVal
		}
		quantized[i] = int8(q)
		resRotated[i] = rotated[i] - (q * scale)
	}

	// Inverse rotation: bring the residual back to the original space.
	residual := make([]float32, n)
	if hasMatrix {
		for i := 0; i < n; i++ {
			var sum float32
			for j := 0; j < n; j++ {
				sum += rotationMatrix[j*n+i] * resRotated[j]
			}
			residual[i] = sum
		}
	} else {
		copy(residual, resRotated)
	}

	return quantized, scale, residual
}

// QJLTransformSIMD projects the residual onto a sign matrix and quantizes the
// projection to a sign, returning the RMS scale of the projection.
//
// The projection and the scale match the CUDA kernel
// `turboquant_qjl_transform_kernel` and the C SIMD kernels.
func QJLTransformSIMD(residual []float32, signMatrix []float32, rows, cols int) ([]int8, float32) {
	if rows <= 0 || cols <= 0 {
		return nil, 0
	}
	if len(signMatrix) >= rows*cols && len(residual) >= cols {
		if q, s, ok := qjlTransformAccelerated(residual, signMatrix, rows, cols); ok {
			return q, s
		}
	}
	return qjlTransformReference(residual, signMatrix, rows, cols)
}

// qjlTransformReference is the canonical pure-Go QJL transform.
func qjlTransformReference(residual []float32, signMatrix []float32, rows, cols int) ([]int8, float32) {
	projected := make([]float32, rows)
	normSq := float32(0.0)

	for i := 0; i < rows; i++ {
		var sum float32
		for j := 0; j < cols; j++ {
			sum += signMatrix[i*cols+j] * residual[j]
		}
		projected[i] = sum
		normSq += sum * sum
	}

	scale := float32(mathx.Sqrt(float64(normSq / float32(rows))))
	quantized := make([]int8, rows)
	for i := 0; i < rows; i++ {
		if projected[i] >= 0 {
			quantized[i] = 1
		} else {
			quantized[i] = -1
		}
	}

	return quantized, scale
}

// PolarQuantVariant performs TurboQuant with specified bit depth
func PolarQuantVariant(input []float32, rotationMatrix []float32, n int, tqType TurboQuantType) ([]int8, float32, []float32) {
	return PolarQuantSIMD(input, rotationMatrix, n, int(tqType))
}

// DequantizeTurboQuant reconstructs float32 from TurboQuant-encoded data
func DequantizeTurboQuant(quantized []int8, scale float32, rotationMatrix []float32, n int) []float32 {
	result := make([]float32, n)

	// First, reconstruct from quantized
	for i := 0; i < n; i++ {
		result[i] = float32(quantized[i]) * scale
	}

	// Apply inverse rotation
	rotated := make([]float32, n)
	for i := 0; i < n; i++ {
		var sum float32
		for j := 0; j < n; j++ {
			sum += rotationMatrix[j*n+i] * result[j]
		}
		rotated[i] = sum
	}

	return rotated
}
