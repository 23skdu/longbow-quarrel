//go:build linux && cuda

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Extern declarations for CUDA kernels from cuda_kernels.cu
extern void cudaRMSNormF16(cudaStream_t stream, void* input, void* weight, void* output, int rows, int cols, float eps);
extern void cudaSwiGLUF16(cudaStream_t stream, void* gate, void* up, void* output, int size);
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

void cudaRoPE(float* tensor, int pos, int heads, int headDim, float theta, cudaStream_t stream) {
    cudaRoPEF16(stream, tensor, pos, heads, headDim, theta);
}

void cudaAdd(float* a, float* b, float* out, int size, cudaStream_t stream) {
    cudaAddF16(stream, a, b, out, size);
}

void cudaSiLU(float* input, float* output, int size, cudaStream_t stream) {
    cudaSiLUF16(stream, input, output, size);
}

// CUDA Graph wrappers
int cudaGraphBeginCaptureWrapper(cudaStream_t stream) {
    return (int)cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
}

int cudaGraphEndCaptureWrapper(cudaStream_t stream, cudaGraph_t* pGraph) {
    return (int)cudaStreamEndCapture(stream, pGraph);
}

int cudaGraphInstantiateWrapper(cudaGraphExec_t* pGraphExec, cudaGraph_t graph) {
    return (int)cudaGraphInstantiate(pGraphExec, graph, 0);
}

int cudaGraphLaunchWrapper(cudaGraphExec_t graphExec, cudaStream_t stream) {
    return (int)cudaGraphLaunch(graphExec, stream);
}

int cudaGraphExecDestroyWrapper(cudaGraphExec_t graphExec) {
    return (int)cudaGraphExecDestroy(graphExec);
}

int cudaGraphDestroyWrapper(cudaGraph_t graph) {
    return (int)cudaGraphDestroy(graph);
}

// On-device sampling extern declaration
extern void cudaSampleOnDevice(
    cudaStream_t stream,
    float* logits,
    const int32_t* prevTokens,
    int numPrevTokens,
    float repetitionPenalty,
    float frequencyPenalty,
    float presencePenalty,
    float temperature,
    int topK,
    float topP,
    unsigned long long seed,
    int vocabSize,
    int32_t* outToken
);

int cudaSample(
    float* logits,
    const int32_t* prevTokens,
    int numPrevTokens,
    float repetitionPenalty,
    float frequencyPenalty,
    float presencePenalty,
    float temperature,
    int topK,
    float topP,
    unsigned long long seed,
    int vocabSize,
    cudaStream_t stream
) {
    int32_t h_token = 0;
    int32_t* d_token = NULL;
    int32_t* d_prev = NULL;

    cudaMalloc((void**)&d_token, sizeof(int32_t));
    if (numPrevTokens > 0 && prevTokens != NULL) {
        cudaMalloc((void**)&d_prev, numPrevTokens * sizeof(int32_t));
        cudaMemcpyAsync(d_prev, prevTokens, numPrevTokens * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    }

    cudaSampleOnDevice(
        stream, logits, d_prev, numPrevTokens,
        repetitionPenalty, frequencyPenalty, presencePenalty,
        temperature, topK, topP, seed, vocabSize, d_token
    );
    cudaMemcpyAsync(&h_token, d_token, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(d_token);
    if (d_prev != NULL) {
        cudaFree(d_prev);
    }
    return (int)h_token;
}

