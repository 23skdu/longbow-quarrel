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

func BenchmarkPolarQuantSIMD(b *testing.B) {
	n := 256
	bits := 4
	input := make([]float32, n)
	rotation := make([]float32, n*n)
	for i := range rotation {
		rotation[i] = 0.1
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _ = PolarQuantSIMD(input, rotation, n, bits)
	}
}
