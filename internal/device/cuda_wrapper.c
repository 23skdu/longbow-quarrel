// CUDA C wrapper for Go CGO
// This file provides C bindings for the CUDA kernels

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Extern declarations for CUDA kernels
extern void launchRMSNormKernel(float* input, float* weight, float* output, int rows, int cols, float eps, cudaStream_t stream);
extern void launchSwiGLUKernel(float* gate, float* up, float* output, int size, cudaStream_t stream);
extern void launchSoftmaxKernel(float* input, float* output, int rows, int cols, cudaStream_t stream);
extern void launchRoPEKernel(float* tensor, int pos, int heads, int headDim, float theta, cudaStream_t stream);
extern void launchAddKernel(float* a, float* b, float* out, int size, cudaStream_t stream);
extern void launchSiLUKernel(float* input, float* output, int size, cudaStream_t stream);

// C wrappers for Go
void cudaRMSNorm(float* input, float* weight, float* output, int rows, int cols, float eps, cudaStream_t stream) {
    launchRMSNormKernel(input, weight, output, rows, cols, eps, stream);
}

void cudaSwiGLU(float* gate, float* up, float* output, int size, cudaStream_t stream) {
    launchSwiGLUKernel(gate, up, output, size, stream);
}

void cudaSoftmax(float* input, float* output, int rows, int cols, cudaStream_t stream) {
    launchSoftmaxKernel(input, output, rows, cols, stream);
}

void cudaRoPE(float* tensor, int pos, int heads, int headDim, float theta, cudaStream_t stream) {
    launchRoPEKernel(tensor, pos, heads, headDim, theta, stream);
}

void cudaAdd(float* a, float* b, float* out, int size, cudaStream_t stream) {
    launchAddKernel(a, b, out, size, stream);
}

void cudaSiLU(float* input, float* output, int size, cudaStream_t stream) {
    launchSiLUKernel(input, output, size, stream);
}
