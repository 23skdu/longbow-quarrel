//go:build amd64 && !avx512

package simd

import (
	"math"
	"unsafe"
)

// Scalar fallback implementations
func softmaxScalar(x []float32) {
	if len(x) == 0 {
		return
	}

	max := x[0]
	for _, v := range x {
		if v > max {
			max = v
		}
	}

	sum := float32(0.0)
	for i := range x {
		x[i] = fastExp(x[i] - max)
		sum += x[i]
	}

	if sum > 0 {
		invSum := float32(1.0) / sum
		for i := range x {
			x[i] *= invSum
		}
	}
}

func swigluScalar(gate, up, out []float32) {
	for i := 0; i < len(gate); i++ {
		g := gate[i]
		if g > 10.0 {
			g = 10.0
		}
		if g < -10.0 {
			g = -10.0
		}
		sigmoid := float32(1.0) / (float32(1.0) + fastExp(-g))
		out[i] = up[i] * g * sigmoid
	}
}

func fp16ToFp32Scalar(src []uint16, dst []float32) {
	for i := 0; i < len(src); i++ {
		dst[i] = fp16ToFp32(src[i])
	}
}

func fp32ToFp16Scalar(src []float32, dst []uint16) {
	for i := 0; i < len(src); i++ {
		dst[i] = fp32ToFp16(src[i])
	}
}

// Fast approximate exp for scalar fallback
func fastExp(x float32) float32 {
	return float32(math.Exp(float64(x)))
}

// FP16 to FP32 conversion
func fp16ToFp32(h uint16) float32 {
	sign := uint32(h>>15) & 0x1
	exp := uint32(h>>10) & 0x1F
	mant := uint32(h) & 0x3FF

	var f32 uint32
	switch exp {
	case 0:
		if mant == 0 {
			f32 = sign << 31
		} else {
			shift := uint32(0)
			m := mant
			for m < 0x400 {
				m <<= 1
				shift++
			}
			m = (m & 0x3FF) << 13
			e := uint32(127 - 14 - shift)
			f32 = (sign << 31) | (e << 23) | m
		}
	case 31:
		if mant == 0 {
			f32 = (sign << 31) | 0x7F800000
		} else {
			f32 = (sign << 31) | 0x7F800000 | (mant << 13)
		}
	default:
		newExp := exp - 15 + 127
		f32 = (sign << 31) | (newExp << 23) | (mant << 13)
	}

	return *(*float32)(unsafe.Pointer(&f32)) // #nosec G103
}

// FP32 to FP16 conversion
func fp32ToFp16(f float32) uint16 {
	bits := *(*uint32)(unsafe.Pointer(&f)) // #nosec G103
	sign := bits >> 31
	exp := (bits >> 23) & 0xFF
	mant := bits & 0x7FFFFF

	switch exp {
	case 0:
		return uint16(sign << 15) // #nosec G115
	case 255:
		return uint16((sign << 15) | 0x7C00 | (mant >> 9)) // #nosec G115
	}

	newExp := int(exp) - 127 + 15
	if newExp >= 31 {
		return uint16((sign << 15) | 0x7C00) // #nosec G115
	} else if newExp <= 0 {
		shift := uint32(1 - newExp) // #nosec G115
		m := mant | 0x800000
		return uint16((sign << 15) | (m >> (9 + shift))) // #nosec G115
	}

	return uint16((sign << 15) | (uint32(newExp) << 10) | (mant >> 13)) // #nosec G115
}
