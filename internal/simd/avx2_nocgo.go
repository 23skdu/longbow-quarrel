//go:build (amd64 || arm64) && !cgo

package simd

import "math"

// SoftmaxAVX2 computes softmax (scalar fallback when CGO disabled)
func SoftmaxAVX2(x []float32) {
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
		x[i] = float32(math.Exp(float64(x[i] - max)))
		sum += x[i]
	}

	if sum > 0 {
		invSum := float32(1.0) / sum
		for i := range x {
			x[i] *= invSum
		}
	}
}

// SwiGLUAVX2 computes SwiGLU activation (scalar fallback when CGO disabled)
func SwiGLUAVX2(gate, up, out []float32) {
	n := len(gate)
	if n != len(up) || n != len(out) {
		return
	}

	for i := 0; i < n; i++ {
		g := gate[i]
		if g > 10.0 {
			g = 10.0
		}
		if g < -10.0 {
			g = -10.0
		}
		sigmoid := float32(1.0) / (float32(1.0) + float32(math.Exp(float64(-g))))
		out[i] = up[i] * g * sigmoid
	}
}

// Fp16ToFp32AVX2 converts FP16 to FP32 (scalar fallback when CGO disabled)
func Fp16ToFp32AVX2(src []uint16, dst []float32) {
	n := len(src)
	if n != len(dst) {
		return
	}

	for i := 0; i < n; i++ {
		dst[i] = fp16ToFp32Scalar(src[i])
	}
}

// Fp32ToFp16AVX2 converts FP32 to FP16 (scalar fallback when CGO disabled)
func Fp32ToFp16AVX2(src []float32, dst []uint16) {
	n := len(src)
	if n != len(dst) {
		return
	}

	for i := 0; i < n; i++ {
		dst[i] = fp32ToFp16Scalar(src[i])
	}
}

func fp16ToFp32Scalar(h uint16) float32 {
	sign := uint32(h>>15) & 0x1
	exp := uint32(h>>10) & 0x1F
	mant := uint32(h) & 0x3FF

	var f32 uint32
	if exp == 0 {
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
	} else if exp == 31 {
		if mant == 0 {
			f32 = (sign << 31) | 0x7F800000
		} else {
			f32 = (sign << 31) | 0x7F800000 | (mant << 13)
		}
	} else {
		newExp := exp - 15 + 127
		f32 = (sign << 31) | (newExp << 23) | (mant << 13)
	}
	return math.Float32frombits(f32)
}

func fp32ToFp16Scalar(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := bits >> 31
	exp := (bits >> 23) & 0xFF
	mant := bits & 0x7FFFFF

	var h uint16
	if exp == 0 {
		h = 0
	} else if exp == 255 {
		h = uint16(sign<<15) | 0x7C00 | uint16(mant>>9)
	} else {
		newExp := int(exp) - 127 + 15
		if newExp >= 31 {
			h = uint16(sign<<15) | 0x7C00
		} else if newExp <= 0 {
			shift := uint32(1 - newExp)
			m := mant | 0x800000
			h = uint16(sign<<15) | uint16(m>>(9+shift))
		} else {
			h = uint16(sign<<15) | uint16(newExp<<10) | uint16(mant>>13)
		}
	}
	return h
}
