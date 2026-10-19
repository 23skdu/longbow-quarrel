//go:build amd64 && cgo

package simd

/*
#include <stdint.h>

void polar_quant_avx512(const float* input, const float* rotation_matrix, int8_t* quantized, float* scale_out, float* residual, int n, int bits);
void qjl_transform_avx512(const float* residual, const float* sign_matrix, int8_t* quantized, float* scale_out, int rows, int cols);
void polar_quant_avx2(const float* input, const float* rotation_matrix, int8_t* quantized, float* scale_out, float* residual, int n, int bits);
void qjl_transform_avx2(const float* residual, const float* sign_matrix, int8_t* quantized, float* scale_out, int rows, int cols);
*/
import "C"

import (
	"unsafe"
)

// The C kernels were written but never referenced from Go, so every build
// used the pure-Go fallback while the documentation advertised vectorized
// TurboQuant. They are dispatched here, guarded by runtime CPU feature
// detection: polar_quant_avx512.c is compiled with
// `#pragma GCC target("avx512f,avx512bw,fma")`, so calling it on a CPU
// without AVX-512 would raise SIGILL.
//
// The kernels reproduce the canonical semantics documented on
// PolarQuantSIMD: the quantized value is clamped to +/-maxQuantVal and the
// residual is returned in the original (un-rotated) space.

// polarQuantAccelerated dispatches to the best available x86 kernel.
func polarQuantAccelerated(input []float32, rotationMatrix []float32, n, bits int) ([]int8, float32, []float32, bool) {
	if n <= 0 || len(input) < n || len(rotationMatrix) < n*n {
		return nil, 0, nil, false
	}

	quantized := make([]int8, n)
	residual := make([]float32, n)
	var scale C.float

	switch {
	case HasAVX512():
		C.polar_quant_avx512(
			(*C.float)(unsafe.Pointer(&input[0])),
			(*C.float)(unsafe.Pointer(&rotationMatrix[0])),
			(*C.int8_t)(unsafe.Pointer(&quantized[0])),
			&scale,
			(*C.float)(unsafe.Pointer(&residual[0])),
			C.int(n),
			C.int(bits),
		)
	case HasAVX2():
		C.polar_quant_avx2(
			(*C.float)(unsafe.Pointer(&input[0])),
			(*C.float)(unsafe.Pointer(&rotationMatrix[0])),
			(*C.int8_t)(unsafe.Pointer(&quantized[0])),
			&scale,
			(*C.float)(unsafe.Pointer(&residual[0])),
			C.int(n),
			C.int(bits),
		)
	default:
		return nil, 0, nil, false
	}

	return quantized, float32(scale), residual, true
}

// qjlTransformAccelerated dispatches to the best available x86 kernel.
func qjlTransformAccelerated(residual []float32, signMatrix []float32, rows, cols int) ([]int8, float32, bool) {
	if rows <= 0 || cols <= 0 || len(residual) < cols || len(signMatrix) < rows*cols {
		return nil, 0, false
	}

	quantized := make([]int8, rows)
	var scale C.float

	switch {
	case HasAVX512():
		C.qjl_transform_avx512(
			(*C.float)(unsafe.Pointer(&residual[0])),
			(*C.float)(unsafe.Pointer(&signMatrix[0])),
			(*C.int8_t)(unsafe.Pointer(&quantized[0])),
			&scale,
			C.int(rows),
			C.int(cols),
		)
	case HasAVX2():
		C.qjl_transform_avx2(
			(*C.float)(unsafe.Pointer(&residual[0])),
			(*C.float)(unsafe.Pointer(&signMatrix[0])),
			(*C.int8_t)(unsafe.Pointer(&quantized[0])),
			&scale,
			C.int(rows),
			C.int(cols),
		)
	default:
		return nil, 0, false
	}

	return quantized, float32(scale), true
}
