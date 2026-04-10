//go:build !(arm64 && cgo) && !(amd64 && cgo)

package simd

import (
	"math"
)

func PolarQuantSIMD(input []float32, rotationMatrix []float32, n int, bits int) ([]int8, float32, []float32) {
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

	finalResidual := make([]float32, n)
	for i := 0; i < n; i++ {
		var sum float32
		for j := 0; j < n; j++ {
			sum += rotationMatrix[j*n+i] * resRotated[j]
		}
		finalResidual[i] = sum
	}

	return quantized, scale, finalResidual
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
