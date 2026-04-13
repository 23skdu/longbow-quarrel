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
