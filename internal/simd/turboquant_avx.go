//go:build amd64 && cgo

package simd

import (
	"math"
)

// TurboQuantType represents the bit depth for TurboQuant encoding
type TurboQuantType int

const (
	TurboQuant2 TurboQuantType = 2 // 2-bit + 1-bit QJL residual
	TurboQuant4 TurboQuantType = 4 // 4-bit + 1-bit QJL residual
	TurboQuant8 TurboQuantType = 8 // 8-bit + 1-bit QJL residual
)

func PolarQuantSIMD(input []float32, rotationMatrix []float32, n int, bits int) ([]int8, float32, []float32) {
	// Fallback to generic when CGO functions unavailable
	rotated := make([]float32, n)
	maxAbs := float32(0.0)

	for i := 0; i < n; i++ {
		var sum float32
		for j := 0; j < n; j++ {
			sum += rotationMatrix[i*n+j] * input[j]
		}
		rotated[i] = sum
		if a := float32(math.Abs(float64(sum))); a > maxAbs {
			maxAbs = a
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
		q := float32(math.Round(float64(rotated[i] * inverseScale)))
		if q > maxQuantVal {
			q = maxQuantVal
		} else if q < -maxQuantVal {
			q = -maxQuantVal
		}
		quantized[i] = int8(q)
		resRotated[i] = rotated[i] - (q * scale)
	}

	return quantized, scale, resRotated
}

func QJLTransformSIMD(residual []float32, signMatrix []float32, rows, cols int) ([]int8, float32) {
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

	scale := float32(math.Sqrt(float64(normSq / float32(rows))))
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

	for i := 0; i < n; i++ {
		result[i] = float32(quantized[i]) * scale
	}

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