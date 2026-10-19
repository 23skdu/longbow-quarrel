package gguf

import (
	"github.com/23skdu/longbow-quarrel/internal/mathx"
)

// The FP8 conversions live here, in the lowest-level package that both the
// GGUF reader and the device layer can use.
//
// They used to exist twice: once in this package and once in
// internal/device/utils.go, and the two did not agree. The copy here used a
// different exponent bias and a different mantissa convention, so it did
// not even round-trip through its own decoder: a value of 16 came back as
// 4, and a value of -16 came back as -131072. The device copy was the
// correct one, and the only test of the pair checked the length of the
// result rather than its values.

// Float32ToFP8E4M3 converts a float32 into an 8-bit float (1 sign, 4
// exponent, 3 mantissa).
//
// The exponent bias is 7. The largest finite value is 448 (the exponent
// field 15 with the mantissa 6); the exponent field 15 with the mantissa 7
// is the NaN encoding. Values below the smallest normal are encoded as
// subnormals, where the mantissa field is the fraction of 2^-6.
func Float32ToFP8E4M3(f float32) uint8 {
	bits := mathx.Float32bits(f)
	sign := uint8((bits >> 31) & 0x1)
	exp := int((bits>>23)&0xff) - 127 + 7 // bias 7
	mant := (bits >> 20) & 0x7            // 3 mantissa bits

	if mathx.IsNaN(float64(f)) || mathx.IsInf(float64(f), 0) {
		// The format has no infinity: it is encoded as the NaN pattern.
		return (sign << 7) | 0x7f
	}
	if exp > 15 || (exp == 15 && mant > 6) {
		// Above the largest finite value, which is 448.
		return (sign << 7) | 0x7e
	}
	if exp <= 0 {
		if exp == 0 && mant == 0 {
			return sign << 7 // Zero.
		}
		// Subnormal: the value is 2^-6 * (m/8), so m is the value
		// scaled by 2^9.
		m := int(mathx.Round(mathx.Abs(float64(f)) * 512.0))
		if m > 7 {
			m = 7
		}
		return (sign << 7) | uint8(m) // #nosec G115
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
			return mathx.Float32frombits(sign << 31)
		}
		// Subnormal.
		return mathx.Float32frombits((sign << 31) | ((127 - 7) << 23) | (mant << 20))
	}
	if exp == 15 && mant == 0x7 {
		return mathx.Float32frombits((sign << 31) | 0x7fc00000) // NaN
	}

	newExp := exp - 7 + 127
	return mathx.Float32frombits((sign << 31) | (newExp << 23) | (mant << 20))
}

// Float32ToFP8E5M2 converts a float32 into an 8-bit float (1 sign, 5
// exponent, 2 mantissa).
//
// The exponent bias is 15. The largest finite value is 57344; the
// exponent field 31 is the infinity and NaN encoding.
func Float32ToFP8E5M2(f float32) uint8 {
	bits := mathx.Float32bits(f)
	sign := uint8((bits >> 31) & 0x1)
	exp := int((bits>>23)&0xff) - 127 + 15 // bias 15
	mant := (bits >> 21) & 0x3             // 2 mantissa bits

	if mathx.IsNaN(float64(f)) {
		return (sign << 7) | 0x7f
	}
	if mathx.IsInf(float64(f), 0) {
		return (sign << 7) | 0x7c
	}
	if exp >= 31 {
		return (sign << 7) | 0x7c // Infinity
	}
	if exp <= 0 {
		if exp == 0 && mant == 0 {
			return sign << 7 // Zero.
		}
		// Subnormal: the value is 2^-14 * (m/4), so m is the value
		// scaled by 2^16.
		m := int(mathx.Round(mathx.Abs(float64(f)) * 65536.0))
		if m > 3 {
			m = 3
		}
		return (sign << 7) | uint8(m) // #nosec G115
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
			return mathx.Float32frombits(sign << 31)
		}
		// Subnormal.
		return mathx.Float32frombits((sign << 31) | ((127 - 15) << 23) | (mant << 21))
	}
	if exp == 31 {
		return mathx.Float32frombits((sign << 31) | 0x7f800000 | (mant << 21))
	}

	newExp := exp - 15 + 127
	return mathx.Float32frombits((sign << 31) | (newExp << 23) | (mant << 21))
}
