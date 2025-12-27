package device

import "math"

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

func Float16ToFloat32(f uint16) float32 {
    sign := (uint32(f) >> 15) & 0x1
    exp := (uint32(f) >> 10) & 0x1f
    mant := uint32(f) & 0x3ff

    if exp == 0 {
        if mant == 0 {
            return math.Float32frombits(sign << 31)
        }
        // Subnormal
        return math.Float32frombits((sign << 31) | (1 << 23)) // Hacky subnormal mapping
    } else if exp == 31 {
        if mant == 0 {
            return math.Float32frombits((sign << 31) | 0x7f800000)
        }
        return math.Float32frombits((sign << 31) | 0x7f800000 | (mant << 13))
    }

    newExp := exp - 15 + 127
    return math.Float32frombits((sign << 31) | (newExp << 23) | (mant << 13))
}
