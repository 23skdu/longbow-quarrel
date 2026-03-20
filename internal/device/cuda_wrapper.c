//go:build linux && cuda

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Extern declarations for CUDA kernels from cuda_kernels.cu
extern void cudaRMSNormF16(cudaStream_t stream, void* input, void* weight, void* output, int rows, int cols, float eps);
extern void cudaSwiGLUF16(cudaStream_t stream, void* gate, void* up, void* output, int size);
extern void cudaSoftmaxF16(cudaStream_t stream, void* input, void* output, int rows, int cols);
extern void cudaRoPEF16(cudaStream_t stream, void* tensor, int pos, int heads, int headDim, float theta);
extern void cudaAddF16(cudaStream_t stream, void* a, void* b, void* out, int size);
extern void cudaSiLUF16(cudaStream_t stream, void* input, void* output, int size);

// C wrappers for Go
void cudaRMSNorm(float* input, float* weight, float* output, int rows, int cols, float eps, cudaStream_t stream) {
    cudaRMSNormF16(stream, input, weight, output, rows, cols, eps);
}

void cudaSwiGLU(float* gate, float* up, float* output, int size, cudaStream_t stream) {
    cudaSwiGLUF16(stream, gate, up, output, size);
}

void cudaSoftmax(float* input, float* output, int rows, int cols, cudaStream_t stream) {
    cudaSoftmaxF16(stream, input, output, rows, cols);
}

void cudaRoPE(float* tensor, int pos, int heads, int headDim, float theta, cudaStream_t stream) {
    cudaRoPEF16(stream, tensor, pos, heads, headDim, theta);
}

void cudaAdd(float* a, float* b, float* out, int size, cudaStream_t stream) {
    cudaAddF16(stream, a, b, out, size);
}

void cudaSiLU(float* input, float* output, int size, cudaStream_t stream) {
    cudaSiLUF16(stream, input, output, size);
}
