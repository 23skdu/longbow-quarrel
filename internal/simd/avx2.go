//go:build (amd64 || arm64) && cgo

package simd

/*
#cgo CFLAGS: -mavx2 -mf16c
#include <stdint.h>

void softmax_avx2(float* x, int n);
void swiglu_avx2(const float* gate, const float* up, float* out, int n);
void fp16_to_fp32_avx2(const uint16_t* src, float* dst, int n);
void fp32_to_fp16_avx2(const float* src, uint16_t* dst, int n);
*/
import "C"
import (
	"math"
	"unsafe"
)

// SoftmaxAVX2 computes softmax using AVX2 intrinsics
func SoftmaxAVX2(x []float32) {
	if len(x) == 0 {
		return
	}

	// Only use AVX2 for larger arrays where overhead is worth it
	if len(x) >= 16 {
		C.softmax_avx2((*C.float)(unsafe.Pointer(&x[0])), C.int(len(x)))
	} else {
		softmaxScalar(x)
	}
}

// SwiGLUAVX2 computes SwiGLU activation using AVX2 intrinsics
func SwiGLUAVX2(gate, up, out []float32) {
	n := len(gate)
	if n == 0 || n != len(up) || n != len(out) {
		return
	}

	if n >= 16 {
		C.swiglu_avx2(
			(*C.float)(unsafe.Pointer(&gate[0])),
			(*C.float)(unsafe.Pointer(&up[0])),
			(*C.float)(unsafe.Pointer(&out[0])),
			C.int(n),
		)
	} else {
		swigluScalar(gate, up, out)
	}
}

// Fp16ToFp32AVX2 converts FP16 to FP32 using AVX2 intrinsics
func Fp16ToFp32AVX2(src []uint16, dst []float32) {
	n := len(src)
	if n == 0 || n != len(dst) {
		return
	}

	if n >= 16 {
		C.fp16_to_fp32_avx2(
			(*C.uint16_t)(unsafe.Pointer(&src[0])),
			(*C.float)(unsafe.Pointer(&dst[0])),
			C.int(n),
		)
	} else {
		fp16ToFp32Scalar(src, dst)
	}
}

// Fp32ToFp16AVX2 converts FP32 to FP16 using AVX2 intrinsics
func Fp32ToFp16AVX2(src []float32, dst []uint16) {
	n := len(src)
	if n == 0 || n != len(dst) {
		return
	}

	if n >= 16 {
		C.fp32_to_fp16_avx2(
			(*C.float)(unsafe.Pointer(&src[0])),
			(*C.uint16_t)(unsafe.Pointer(&dst[0])),
			C.int(n),
		)
	} else {
		fp32ToFp16Scalar(src, dst)
	}
}

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
	if x < -10.0 {
		return 0.0
	}
	if x > 10.0 {
		return 22026.465
	}
	x64 := float64(x)
	result := 1.0 + x64 + x64*x64/2.0 + x64*x64*x64/6.0 + x64*x64*x64*x64/24.0
	return float32(result)
}

// FP16 to FP32 conversion
func fp16ToFp32(h uint16) float32 {
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

	return *(*float32)(unsafe.Pointer(&f32))
}

// FP32 to FP16 conversion
func fp32ToFp16(f float32) uint16 {
	bits := *(*uint32)(unsafe.Pointer(&f))
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
