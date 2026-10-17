package device

import (
	"github.com/23skdu/longbow-quarrel/internal/mathx"
)

// GetPrecomputedRotation returns a deterministic orthogonal matrix for the given dimension.
func GetPrecomputedRotation(dim int) []float32 {
	// Standard DCT-II orthogonal matrix
	data := make([]float32, dim*dim)
	scale0 := float32(mathx.Sqrt(1.0 / float64(dim)))
	scaleN := float32(mathx.Sqrt(2.0 / float64(dim)))

	for i := 0; i < dim; i++ {
		scale := scaleN
		if i == 0 {
			scale = scale0
		}
		for j := 0; j < dim; j++ {
			data[i*dim+j] = scale * float32(mathx.Cos(mathx.Pi*float64(i)*(float64(j)+0.5)/float64(dim)))
		}
	}
	return data
}

// GetPrecomputedQJLSigns returns deterministic random signs for QJL transformation.
func GetPrecomputedQJLSigns(n int) []float32 {
	data := make([]float32, n)
	for i := 0; i < n; i++ {
		// Deterministic "random" signs using a simple oscillation
		if mathx.Sin(float64(i)*0.1)*mathx.Cos(float64(i)*0.7) > 0 {
			data[i] = 1.0
		} else {
			data[i] = -1.0
		}
	}
	return data
}
