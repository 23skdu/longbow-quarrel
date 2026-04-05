//go:build arm64 && cgo
package simd

/*
#include <stdint.h>
void polar_quant_neon(const float* input, const float* rotation_matrix, int8_t* quantized, float* scale_out, float* residual, int n, int bits);
void qjl_transform_neon(const float* residual, const float* sign_matrix, int8_t* quantized, float* scale_out, int rows, int cols);
*/
import "C"
import "unsafe"

func PolarQuantSIMD(input []float32, rotationMatrix []float32, n int, bits int) ([]int8, float32, []float32) {
	quantized := make([]int8, n)
	residual := make([]float32, n)
	var scale C.float

	C.polar_quant_neon(
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

	C.qjl_transform_neon(
		(*C.float)(unsafe.Pointer(&residual[0])),
		(*C.float)(unsafe.Pointer(&signMatrix[0])),
		(*C.int8_t)(unsafe.Pointer(&quantized[0])),
		&scale,
		C.int(rows),
		C.int(cols),
	)

	return quantized, float32(scale)
}
