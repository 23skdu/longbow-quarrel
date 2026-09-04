//go:build amd64 && cgo && !avx512

package simd

/*
#cgo CFLAGS: -mavx2
#include <stdint.h>

void softmax_avx2(float* x, long n);
void swiglu_avx2(const float* gate, const float* up, float* out, long n);
void fp16_to_fp32_avx2(const uint16_t* src, float* dst, long n);
void fp32_to_fp16_avx2(const float* src, uint16_t* dst, long n);
*/
import "C"
import (
	"unsafe"
)

func useAVX2() bool {
	if !cpuInitDone {
		detectCPU()
	}
	return cpuInitDone && hasAVX2
}

// SoftmaxAVX2 computes softmax using AVX2 intrinsics
func SoftmaxAVX2(x []float32) {
	if len(x) == 0 {
		return
	}

	if useAVX2() && len(x) >= 16 {
		C.softmax_avx2((*C.float)(unsafe.Pointer(&x[0])), C.long(len(x)))
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

	if useAVX2() && n >= 16 {
		C.swiglu_avx2(
			(*C.float)(unsafe.Pointer(&gate[0])),
			(*C.float)(unsafe.Pointer(&up[0])),
			(*C.float)(unsafe.Pointer(&out[0])),
			C.long(n),
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

	if useAVX2() && n >= 16 {
		C.fp16_to_fp32_avx2(
			(*C.uint16_t)(unsafe.Pointer(&src[0])),
			(*C.float)(unsafe.Pointer(&dst[0])),
			C.long(n),
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

	if useAVX2() && n >= 16 {
		C.fp32_to_fp16_avx2(
			(*C.float)(unsafe.Pointer(&src[0])),
			(*C.uint16_t)(unsafe.Pointer(&dst[0])),
			C.long(n),
		)
	} else {
		fp32ToFp16Scalar(src, dst)
	}
}
