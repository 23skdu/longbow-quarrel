package gguf

import (
	"encoding/binary"
	"math"
	"testing"
)

// TestQ4KDequantization tests the Q4K dequantization matches expected behavior
func TestQ4KDequantization(t *testing.T) {
	// Create a simple Q4K block with known values
	// Block size: 144 bytes for 256 weights
	block := make([]byte, 144)
	
	// Set d (scale) to 0.001 (as FP16)
	d := float32ToFloat16(0.001)
	binary.LittleEndian.PutUint16(block[0:2], d)
	
	// Set dmin to 0.0001 (as FP16)
	dmin := float32ToFloat16(0.0001)
	binary.LittleEndian.PutUint16(block[2:4], dmin)
	
	// Set scales: simple pattern for first 4 sub-blocks
	// scales[0-3] = 10, 20, 30, 40 (6-bit values)
	block[4] = 10  // sc[0] lower 6 bits
	block[5] = 20  // sc[1] lower 6 bits
	block[6] = 30  // sc[2] lower 6 bits
	block[7] = 40  // sc[3] lower 6 bits
	
	// scales[4-7] = 5, 15, 25, 35
	block[8] = 5   // m[0] lower 6 bits
	block[9] = 15  // m[1] lower 6 bits
	block[10] = 25 // m[2] lower 6 bits
	block[11] = 30 // m[3] lower 6 bits
	
	// Set quantized values (qs): all zeros for simplicity
	// qs starts at offset 16, 128 bytes
	for i := 16; i < 144; i++ {
		block[i] = 0x00 // Each byte has two 4-bit values (both 0)
	}
	
	// Dequantize
	result := DequantizeQ4K(block, 256)
	
	// Check result length
	if len(result) != 256 {
		t.Fatalf("Expected 256 weights, got %d", len(result))
	}
	
	// For sub-block 0 (first 32 weights):
	// d_val = d * sc[0] = 0.001 * 10 = 0.01
	// m_val = dmin * m[0] = 0.0001 * 5 = 0.0005
	// weight = d_val * q - m_val = 0.01 * 0 - 0.0005 = -0.0005
	expected := float32(-0.0005)
	
	for i := 0; i < 32; i++ {
		if math.Abs(float64(result[i]-expected)) > 0.0001 {
			t.Errorf("Weight %d: expected %.6f, got %.6f", i, expected, result[i])
		}
	}
	
	t.Logf("Q4K dequantization test passed: first 32 weights = %.6f", result[0])
}

// Helper to convert float32 to float16
func float32ToFloat16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := (bits >> 31) & 0x1
	exp := (bits >> 23) & 0xff
	mant := bits & 0x7fffff

	if exp == 0 {
		return uint16(sign << 15)
	} else if exp == 0xff {
		return uint16((sign << 15) | 0x7c00 | (mant >> 13))
	}

	newExp := int(exp) - 127 + 15
	if newExp < 0 {
		return uint16(sign << 15)
	} else if newExp >= 31 {
		return uint16((sign << 15) | 0x7c00)
	}

	return uint16((sign << 15) | (uint32(newExp) << 10) | (mant >> 13))
}
