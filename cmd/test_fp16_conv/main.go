//go:build darwin && metal

package main

import (
	"encoding/binary"
	"fmt"
	"math"
)

// Manual FP16->FP32 conversion
// Manual FP16->FP32 conversion
func Float16ToFloat32Manual(bits uint16) float32 {
	sign := uint32(bits&0x8000) << 16
	expRaw := uint32(bits&0x7C00) >> 10
	mant := uint32(bits & 0x03FF)

	var result uint32
	if expRaw == 0 {
		// Subnormal or zero
		if mant == 0 {
			result = sign
		} else {
			// Subnormal: normalize it
			shift := uint32(0)
			testMant := mant
			for (testMant & 0x0400) == 0 {
				testMant <<= 1
				shift++
			}
			finalMant := (testMant & 0x03FF) << 13
			exp := (113 - shift) << 23
			result = sign | exp | finalMant
		}
	} else {
		// Normal
		exp := (expRaw + (127 - 15)) << 23
		result = sign | exp | (mant << 13)
	}

	return math.Float32frombits(result)
}

func main() {
	// Test with the actual Mistral values
	d_bits := uint16(0x01a0)    // From extraction: 0.000024795532
	dmin_bits := uint16(0x0b59) // From extraction: 0.000224232674

	// Using standard library
	// d_std := math.Float32frombits(uint32(binary.LittleEndian.Uint16([]byte{byte(d_bits & 0xFF), byte(d_bits >> 8)}))<<16 | uint32(binary.LittleEndian.Uint16([]byte{byte(d_bits & 0xFF), byte(d_bits >> 8)})))

	// Simpler: just use encoding/binary approach
	buf := make([]byte, 2)
	binary.LittleEndian.PutUint16(buf, d_bits)

	// Actually, let me use a proper conversion
	d_manual := Float16ToFloat32Manual(d_bits)
	dmin_manual := Float16ToFloat32Manual(dmin_bits)

	fmt.Printf("d_bits: 0x%04x\n", d_bits)
	fmt.Printf("d manual: %.12f\n", d_manual)
	fmt.Printf("Expected: %.12f\n", 0.000024795532)
	fmt.Printf("\n")
	fmt.Printf("dmin_bits: 0x%04x\n", dmin_bits)
	fmt.Printf("dmin manual: %.12f\n", dmin_manual)
	fmt.Printf("Expected: %.12f\n", 0.000224232674)
}
