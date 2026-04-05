//go:build amd64 && cgo
package simd

/*
#cgo CFLAGS: -mavx2
#include <stdint.h>
void polar_quant_avx2(const float* input, const float* rotation_matrix, int8_t* quantized, float* scale_out, float* residual, int n, int bits);
void qjl_transform_avx2(const float* residual, const float* sign_matrix, int8_t* quantized, float* scale_out, int rows, int cols);

#ifdef AVX512
#cgo CFLAGS: -mavx512f -mavx512bw
void polar_quant_avx512(const float* input, const float* rotation_matrix, int8_t* quantized, float* scale_out, float* residual, int n, int bits);
void qjl_transform_avx512(const float* residual, const float* sign_matrix, int8_t* quantized, float* scale_out, int rows, int cols);
#endif
*/
import "C"
import "unsafe"

func PolarQuantSIMD(input []float32, rotationMatrix []float32, n int, bits int) ([]int8, float32, []float32) {
	quantized := make([]int8, n)
	residual := make([]float32, n)
	var scale C.float

	// TODO: Runtime CPU detection for AVX512
	// For now, use AVX2 as default for x86-64
	C.polar_quant_avx2(
		(*C.float)(unsafe.Pointer(&input[0])),
		(*C.float)(unsafe.Pointer(&rotationMatrix[0])),
		(*C.int8_t)(unsafe.Pointer(&quantized[0])),
		&scale,
		(*C.float)(unsafe.Pointer(&residual[0])),
		C.int(n),
		C.int(bits),
	)

	return quantized, float32(scale), residual
}

func QJLTransformSIMD(residual []float32, signMatrix []float32, rows, cols int) ([]int8, float32) {
	quantized := make([]int8, rows)
	var scale C.float

	C.qjl_transform_avx2(
		(*C.float)(unsafe.Pointer(&residual[0])),
		(*C.float)(unsafe.Pointer(&signMatrix[0])),
		(*C.int8_t)(unsafe.Pointer(&quantized[0])),
		&scale,
		C.int(rows),
		C.int(cols),
	)

	return quantized, float32(scale)
}
