package simd

import (
	"math"
	"testing"
)

func TestTurboQuantSIMD(t *testing.T) {
	n := 256
	bits := 4
	input := make([]float32, n)
	rotation := make([]float32, n*n)
	for i := 0; i < n; i++ {
		input[i] = float32(i) / 100.0
		rotation[i*n+i] = 1.0 // Identity matrix
	}

	// 1. PolarQuant
	q, s, res := PolarQuantSIMD(input, rotation, n, bits)
	
	// Expectations for identity matrix: rotated == input
	shiftAmount := uint(bits - 1)
	maxQuantVal := float32((int(1) << shiftAmount) - 1)
	expectedScale := float32(math.Abs(float64(n-1)/100.0)) / maxQuantVal
	
	if math.Abs(float64(s-expectedScale)) > 1e-5 {
		t.Errorf("PolarQuantSIMD: scale got %v, want %v", s, expectedScale)
	}

	if len(q) != n || len(res) != n {
		t.Errorf("PolarQuantSIMD: size mismatch")
	}

	// 2. QJLTransform
	rows := 32
	qjlMatrix := make([]float32, rows*n)
	for i := range qjlMatrix {
		if i%2 == 0 {
			qjlMatrix[i] = 1.0
		} else {
			qjlMatrix[i] = -1.0
		}
	}

	qj, sj := QJLTransformSIMD(res, qjlMatrix, rows, n)
	if len(qj) != rows {
		t.Errorf("QJLTransformSIMD: size mismatch")
	}
	if sj < 0 {
		t.Errorf("QJLTransformSIMD: negative scale %v", sj)
	}
}

func TestTurboQuantVariants(t *testing.T) {
	n := 128
	input := make([]float32, n)
	rotation := make([]float32, n*n)
	for i := 0; i < n; i++ {
		input[i] = float32(i+1) * 0.1
		rotation[i*n+i] = 1.0
	}

	variants := []TurboQuantType{TurboQuant2, TurboQuant4, TurboQuant8}
	names := []string{"TurboQuant2", "TurboQuant4", "TurboQuant8"}

	for i, tt := range variants {
		t.Run(names[i], func(t *testing.T) {
			q, s, res := PolarQuantVariant(input, rotation, n, tt)
			if len(q) != n {
				t.Errorf("PolarQuantVariant(%s): size mismatch", names[i])
			}
			if s <= 0 {
				t.Errorf("PolarQuantVariant(%s): invalid scale %v", names[i], s)
			}
			if len(res) != n {
				t.Errorf("PolarQuantVariant(%s): residual size mismatch", names[i])
			}

			// Verify dequantization works
			reconstructed := DequantizeTurboQuant(q, s, rotation, n)
			if len(reconstructed) != n {
				t.Errorf("DequantizeTurboQuant(%s): size mismatch", names[i])
			}
		})
	}
}

func TestDequantizeTurboQuant(t *testing.T) {
	n := 64
	rotation := make([]float32, n*n)
	for i := 0; i < n; i++ {
		rotation[i*n+i] = 1.0
	}

	quantized := make([]int8, n)
	for i := 0; i < n; i++ {
		quantized[i] = int8(i % 16 - 8)
	}
	scale := float32(0.1)

	reconstructed := DequantizeTurboQuant(quantized, scale, rotation, n)
	if len(reconstructed) != n {
		t.Error("DequantizeTurboQuant: size mismatch")
	}

	// Verify values reconstruct correctly
	for i := 0; i < 5; i++ {
		expected := float32(quantized[i]) * scale
		if math.Abs(float64(reconstructed[i]-expected)) > 0.01 {
			t.Errorf("DequantizeTurboQuant[%d]: got %v, want %v", i, reconstructed[i], expected)
		}
	}
}

func BenchmarkPolarQuantSIMD(b *testing.B) {
	n := 256
	bits := 4
	input := make([]float32, n)
	rotation := make([]float32, n*n)
	for i := range rotation {
		rotation[i] = 0.1
	}

	for b.Loop() {
		_, _, _ = PolarQuantSIMD(input, rotation, n, bits)
	}
}

func BenchmarkTurboQuantVariants(b *testing.B) {
	n := 256
	input := make([]float32, n)
	rotation := make([]float32, n*n)
	for i := range rotation {
		rotation[i] = 0.1
	}

	variants := []TurboQuantType{TurboQuant2, TurboQuant4, TurboQuant8}
	names := []string{"TurboQuant2", "TurboQuant4", "TurboQuant8"}
	for i, tt := range variants {
		tt := tt
		b.Run(names[i], func(b *testing.B) {
			for b.Loop() {
				_, _, _ = PolarQuantVariant(input, rotation, n, tt)
			}
		})
	}
}
