package device

// utils.go provides platform-agnostic data types and utility functions.
// This file is compiled into all builds and must NOT contain platform-specific
// build tags, CGO references, or dependencies on Metal/CUDA-specific headers.

import (
	"math"
	"unsafe"
)

type DataType int

const (
	DataTypeF16  DataType = 0
	DataTypeQ4K  DataType = 1
	DataTypeQ4_0 DataType = 2
	DataTypeF32  DataType = 3
	DataTypeQ6K  DataType = 4
	DataTypeQ3K  DataType = 5
	DataTypeQ8_0 DataType = 6

	DataTypeQ4_K   = DataTypeQ4K
	DataTypeQ6_K   = DataTypeQ6K
	DataTypeIQ4_NL DataType = 7
	DataTypeMXFP4  DataType = 8
	DataTypeTQ1_0  DataType = 9
	DataTypeTQ2_0  DataType = 10
	DataTypeFP8    DataType = 11
	DataTypeINT8   DataType = 12

	DataTypeQ5_K DataType = 13
	DataTypeQ2_K DataType = 14
	DataTypeQ1_K DataType = 15
	DataTypeI32   DataType = 16
)

// Reference implementation of Float32 <-> Float16
// Real implementation should handle exponents properly or use a library,
// but for a rough draft this truncation works for small values.
// Ideally use `github.com/x448/float16` if allowed or copy proper bit magic.

// Fast approximation for prototype
func Float32ToFloat16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := (bits >> 31) & 0x1
	exp := (bits >> 23) & 0xff
	mant := bits & 0x7fffff

	switch exp {
	case 0:
		return uint16(sign << 15)
	case 0xff:
		return uint16((sign << 15) | 0x7c00 | (mant >> 13)) // #nosec G115 -- safe bit manipulation
	}

	newExp := int(exp) - 127 + 15
	if newExp < 0 {
		return uint16(sign << 15) // Flush to zero
	} else if newExp >= 31 {
		return uint16((sign << 15) | 0x7c00) // #nosec G115 -- safe bit manipulation
	}

	return uint16((sign << 15) | (uint32(newExp) << 10) | (mant >> 13)) // #nosec G115 -- safe bit manipulation
}

func Float16ToFloat32(f uint16) float32 {
	sign := (uint32(f) >> 15) & 0x1
	exp := (uint32(f) >> 10) & 0x1f
	mant := uint32(f) & 0x3ff

	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign << 31)
		}
		// Subnormal: value = (-1)^sign * 2^-14 * (mant / 1024)
		// In FP32: need to normalize it
		// Find leading bit position
		shift := uint32(0)
		temp := mant
		for temp < 0x400 {
			temp <<= 1
			shift++
		}
		// Now temp has implicit 1 in bit 10
		mant = (temp & 0x3ff) << 13 // Keep only fractional part, shift to FP32 position
		exp = 127 - 14 - shift      // FP32 bias - FP16 subnormal exp - normalization shift
		return math.Float32frombits((sign << 31) | (exp << 23) | mant)
	case 31:
		if mant == 0 {
			return math.Float32frombits((sign << 31) | 0x7f800000)
		}
		return math.Float32frombits((sign << 31) | 0x7f800000 | (mant << 13))
	}

	newExp := exp - 15 + 127
	return math.Float32frombits((sign << 31) | (newExp << 23) | (mant << 13))
}

func Float32SliceToBytes(s []float32) []byte {
	if len(s) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&s[0])), len(s)*4) // #nosec G103 G115 -- intentional unsafe for zero-copy conversion
}

// Float32ToFP8E4M3 converts a float32 into an 8-bit float (1 sign, 4 exponent, 3 mantissa).
func Float32ToFP8E4M3(f float32) uint8 {
	bits := math.Float32bits(f)
	sign := uint8((bits >> 31) & 0x1)
	exp := int((bits>>23)&0xff) - 127 + 7 // bias 7
	mant := (bits >> 20) & 0x7            // 3 mantissa bits

	if exp <= 0 {
		return sign << 7 // Subnormal / zero flush
	}
	if exp >= 15 {
		// Clamp to max representable normal value
		return (sign << 7) | 0x7e
	}
	return (sign << 7) | (uint8(exp) << 3) | uint8(mant) // #nosec G115
}

// FP8E4M3ToFloat32 converts an 8-bit float (E4M3) back into float32.
func FP8E4M3ToFloat32(b uint8) float32 {
	sign := uint32(b>>7) & 0x1
	exp := uint32(b>>3) & 0xf
	mant := uint32(b) & 0x7

	if exp == 0 {
		if mant == 0 {
			return math.Float32frombits(sign << 31)
		}
		// Subnormal
		return math.Float32frombits((sign << 31) | ((127 - 7) << 23) | (mant << 20))
	}
	if exp == 15 && mant == 0x7 {
		return math.Float32frombits((sign << 31) | 0x7fc00000) // NaN
	}

	newExp := exp - 7 + 127
	return math.Float32frombits((sign << 31) | (newExp << 23) | (mant << 20))
}

// Float32ToFP8E5M2 converts a float32 into an 8-bit float (1 sign, 5 exponent, 2 mantissa).
func Float32ToFP8E5M2(f float32) uint8 {
	bits := math.Float32bits(f)
	sign := uint8((bits >> 31) & 0x1)
	exp := int((bits>>23)&0xff) - 127 + 15 // bias 15
	mant := (bits >> 21) & 0x3             // 2 mantissa bits

	if exp <= 0 {
		return sign << 7
	}
	if exp >= 31 {
		return (sign << 7) | 0x7c // Infinity
	}
	return (sign << 7) | (uint8(exp) << 2) | uint8(mant) // #nosec G115
}

// FP8E5M2ToFloat32 converts an 8-bit float (E5M2) back into float32.
func FP8E5M2ToFloat32(b uint8) float32 {
	sign := uint32(b>>7) & 0x1
	exp := uint32(b>>2) & 0x1f
	mant := uint32(b) & 0x3

	if exp == 0 {
		if mant == 0 {
			return math.Float32frombits(sign << 31)
		}
		// Subnormal
		return math.Float32frombits((sign << 31) | ((127 - 15) << 23) | (mant << 21))
	}
	if exp == 31 {
		return math.Float32frombits((sign << 31) | 0x7f800000 | (mant << 21))
	}

	newExp := exp - 15 + 127
	return math.Float32frombits((sign << 31) | (newExp << 23) | (mant << 21))
}

// QuantizeBlockQ8_0 quantizes a slice of float32 into signed 8-bit integers with scale.
func QuantizeBlockQ8_0(src []float32, dst []int8) float32 {
	if len(src) == 0 || len(dst) < len(src) {
		return 0
	}
	var amax float32
	for _, v := range src {
		abs := float32(math.Abs(float64(v)))
		if abs > amax {
			amax = abs
		}
	}
	if amax == 0 {
		for i := range src {
			dst[i] = 0
		}
		return 0
	}

	scale := amax / 127.0
	invScale := 127.0 / amax
	for i, v := range src {
		q := int(math.Round(float64(v * invScale)))
		if q > 127 {
			q = 127
		} else if q < -127 {
			q = -127
		}
		dst[i] = int8(q) // #nosec G115
	}
	return scale
}

// DequantizeBlockQ8_0 dequantizes signed 8-bit integers back to float32 using scale.
func DequantizeBlockQ8_0(src []int8, scale float32, dst []float32) {
	n := len(src)
	if len(dst) < n {
		n = len(dst)
	}
	for i := 0; i < n; i++ {
		dst[i] = float32(src[i]) * scale
	}
}

