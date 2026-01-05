//go:build darwin && metal

package main

import (
	"fmt"
	"math"
)

func Float32ToFloat16(f float32) uint16 {
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
        return uint16(sign << 15) // Flush to zero
    } else if newExp >= 31 {
        return uint16((sign << 15) | 0x7c00) // Inf
    }

    return uint16((sign << 15) | (uint32(newExp) << 10) | (mant >> 13))
}

func main() {
	v := float32(0.00415)
	fmt.Printf("Input: %f -> %x\n", v, Float32ToFloat16(v))
	
	v2 := float32(0.037124)
	fmt.Printf("Input: %f -> %x\n", v2, Float32ToFloat16(v2))

	v3 := float32(1e-6)
	fmt.Printf("Input: %f -> %x\n", v3, Float32ToFloat16(v3))
}
