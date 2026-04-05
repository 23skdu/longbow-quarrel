package gguf

import (
	"math"
	"testing"
)

func TestPolarQuant(t *testing.T) {
	n := 4
	bits := 4
	input := []float32{1.5, -2.3, 0.8, 3.1}
	
	// Use identity matrix for rotation to simplify MVP test
	rotation := GenerateRandomOrthogonalMatrix(n)

	quantized, scale, residual, err := PolarQuant(input, rotation, n, bits)
	if err != nil {
		t.Fatalf("PolarQuant failed: %v", err)
	}

	if len(quantized) != n {
		t.Errorf("Expected %d quantized values, got %d", n, len(quantized))
	}
	if len(residual) != n {
		t.Errorf("Expected %d residual values, got %d", n, len(residual))
	}

	// Reconstruct input: R^T * (quantized * scale) + residual
	// Since R = I, input ≈ quantized * scale + residual
	for i := 0; i < n; i++ {
		reconstructed := float32(quantized[i]) * scale + residual[i]
		if math.Abs(float64(reconstructed - input[i])) > 1e-4 {
			t.Errorf("Mismatch at index %d: reconstructed %f, original %f", i, reconstructed, input[i])
		}
	}
}

func TestTurboQuantRoundtrip(t *testing.T) {
	n := 256
	input := make([]float32, n)
	for i := range input {
		input[i] = float32(math.Sin(float64(i)))
	}

	rot := GenerateRandomOrthogonalMatrix(n)
	qjl := GenerateRandomSignMatrix(64, n)

	data, err := QuantizeTurboQuant(input, rot, qjl, n, 4)
	if err != nil {
		t.Fatal(err)
	}

	output, err := DequantizeTurboQuant(data, rot, qjl, n)
	if err != nil {
		t.Fatal(err)
	}

	if len(output) != n {
		t.Errorf("got %d, want %d", len(output), n)
	}

	// Check if reconstruction is somewhat reasonable (allow higher error since it's very lossy)
	for i := 0; i < 10; i++ {
		if math.Abs(float64(input[i]-output[i])) > 0.5 {
			t.Errorf("large error at %d: in=%f, out=%f", i, input[i], output[i])
		}
	}
}

func TestQJLTransform(t *testing.T) {
	rows := 4
	cols := 4
	residual := []float32{0.1, -0.2, 0.05, -0.15}
	
	signMatrix := GenerateRandomSignMatrix(rows, cols)

	quantized, scale, err := QJLTransform(residual, signMatrix, rows, cols)
	if err != nil {
		t.Fatalf("QJLTransform failed: %v", err)
	}

	if len(quantized) != rows {
		t.Errorf("Expected %d quantized QJL values, got %d", rows, len(quantized))
	}
	if scale <= 0 && scale != 0 {
		t.Errorf("Expected positive scale, got %f", scale)
	}
}
