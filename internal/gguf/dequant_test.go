package gguf

import (
	"math"
	"testing"
)

func TestDequantizeBlock(t *testing.T) {
	tests := []struct {
		name     string
		dataType GGMLType
		size     int
		input    []byte
	}{
		{
			name:     "F32",
			dataType: GGMLTypeF32,
			size:     4,
			input:    []byte{0, 0, 0, 0}, // 0.0
		},
		{
			name:     "F16",
			dataType: GGMLTypeF16,
			size:     2,
			input:    []byte{0, 0}, // 0.0 in F16
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, 1)
			DequantizeBlock(tt.input, dst, tt.dataType)
			if math.IsNaN(float64(dst[0])) {
				t.Errorf("DequantizeBlock(%s) produced NaN", tt.name)
			}
		})
	}
}

func TestDequantizeQ4_K(t *testing.T) {
	// 256 weights in Q4_K
	// Super-block: 8 blocks of 32
	// Each block has 16 bytes (32 weights @ 4bit)
	// Plus scales/mins
	input := make([]byte, 176) // Correct size for Q4_K block
	dst := make([]float32, 256)
	
	// Fill with some pattern
	for i := range input {
		input[i] = byte(i)
	}

	DequantizeBlock(input, dst, GGMLTypeQ4_K)
	
	// Just verify no panic and non-zero-ish values (since input is non-zero)
	foundNonZero := false
	for _, v := range dst {
		if v != 0 {
			foundNonZero = true
			break
		}
	}
	if !foundNonZero {
		t.Error("DequantizeBlock(Q4_K) produced all zeros for non-zero input")
	}
}

func TestDequantizeQ6_K(t *testing.T) {
	input := make([]byte, 210) // Approx size for Q6_K
	dst := make([]float32, 256)
	
	DequantizeBlock(input, dst, GGMLTypeQ6_K)
	// Verify non-panic
}

func TestDequantizeQ8_0(t *testing.T) {
	input := make([]byte, 34) // 32 weights @ 8bit + 2 byte scale
	dst := make([]float32, 32)
	
	DequantizeBlock(input, dst, GGMLTypeQ8_0)
}

func TestMatVecMulQ8_0(t *testing.T) {
	// Create 2 rows, 64 cols (2 blocks per row, 68 bytes per row = 136 bytes)
	rows := 2
	cols := 64
	data := make([]byte, rows*(cols/32)*34)
	
	// Set scale = 1.0 (float16: 0x3C00) for all blocks
	for b := 0; b < rows*(cols/32); b++ {
		data[b*34] = 0x00
		data[b*34+1] = 0x3C
		// Set values to 1, 2, 3...
		for j := 0; j < 32; j++ {
			data[b*34+2+j] = byte(j + 1)
		}
	}

	vector := make([]float32, cols)
	for i := range vector {
		vector[i] = 1.0
	}

	res := MatVecMulQ8_0(data, vector, rows, cols)
	if len(res) != rows {
		t.Fatalf("expected %d rows, got %d", rows, len(res))
	}

	dequant := DequantizeQ8_0(data, rows*cols)
	for r := 0; r < rows; r++ {
		var expected float32
		for c := 0; c < cols; c++ {
			expected += dequant[r*cols+c] * vector[c]
		}
		diff := math.Abs(float64(res[r] - expected))
		if diff > 1e-4 {
			t.Errorf("row %d: got %f, want %f (diff %f)", r, res[r], expected, diff)
		}
	}
}
