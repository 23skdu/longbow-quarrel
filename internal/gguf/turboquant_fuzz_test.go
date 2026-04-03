package gguf

import (
	"testing"
)

func FuzzPolarQuant(f *testing.F) {
	f.Add(float32(1.0), float32(-1.0), float32(0.5), float32(2.5))
	f.Add(float32(0.0), float32(0.0), float32(0.0), float32(0.0))
	f.Add(float32(100.0), float32(-100.0), float32(50.0), float32(-50.0))

	f.Fuzz(func(t *testing.T, a, b, c, d float32) {
		n := 4
		bits := 4
		input := []float32{a, b, c, d}
		rotation := GenerateRandomOrthogonalMatrix(n)

		quantized, scale, residual, err := PolarQuant(input, rotation, n, bits)
		if err != nil {
			t.Errorf("PolarQuant error: %v", err)
		}
		
		if len(quantized) != n || len(residual) != n {
			t.Errorf("Output length mismatch")
		}

		// Ensure no panics for NaN or Inf (math.IsNaN(a) handles its own control flow)
		_ = scale
	})
}
