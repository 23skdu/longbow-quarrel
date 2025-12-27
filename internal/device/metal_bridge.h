#ifndef METAL_BRIDGE_H
#define METAL_BRIDGE_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque pointer types
typedef void *MetalContextRef;
typedef void *MetalBufferRef;

// Device management
MetalContextRef Metal_Init(const char *libSource);
void Metal_Free(MetalContextRef ctx);
void Metal_Synchronize(MetalContextRef ctx);

// Buffer Management
MetalBufferRef Metal_Alloc(MetalContextRef ctx, int size);
void Metal_FreeBuffer(MetalContextRef ctx, MetalBufferRef buf);
void Metal_CopyToDevice(MetalBufferRef buf, int offset, const void *data,
                        int size);
void Metal_CopyToHost(MetalBufferRef buf, int offset, void *data, int size);
void *Metal_GetBufferContents(MetalBufferRef buf);

// Basic Ops
void Metal_Add_F16(MetalContextRef ctx, MetalBufferRef a, int offA,
                   MetalBufferRef b, int offB, MetalBufferRef result,
                   int offRes, int count);
void Metal_Scale_F16(MetalContextRef ctx, MetalBufferRef a, int offA,
                     uint16_t val, MetalBufferRef result, int offRes,
                     int count);

void Metal_Embedding_F16(MetalContextRef ctx, MetalBufferRef weights, int offW,
                         MetalBufferRef result, int offRes, int rowIdx,
                         int cols);

// Llama Specific Kernels
void Metal_RMSNorm_F16(MetalContextRef ctx, MetalBufferRef input, int offIn,
                       MetalBufferRef weight, int offWeight,
                       MetalBufferRef result, int offRes, int rows, int cols,
                       float eps);

void Metal_RoPE_F16(MetalContextRef ctx, MetalBufferRef data, int offData,
                    int batchSize, int seqLen, int numHeads, int headDim,
                    int posOffset);

void Metal_SwiGLU_F16(MetalContextRef ctx, MetalBufferRef inputVal, int offVal,
                      MetalBufferRef inputGate, int offGate,
                      MetalBufferRef output, int offOut, int n, int interSize);

void Metal_Softmax_F16(MetalContextRef ctx, MetalBufferRef input, int offIn,
                       MetalBufferRef result, int offRes, int rows, int cols);

void Metal_StoreKV_F16(MetalContextRef ctx, MetalBufferRef k, int offK,
                       MetalBufferRef v, int offV, MetalBufferRef kCache,
                       MetalBufferRef vCache, int pos, int heads, int headDim);

void Metal_Attention_F16(MetalContextRef ctx, MetalBufferRef q, int offQ,
                         MetalBufferRef kCache, MetalBufferRef vCache,
                         MetalBufferRef result, int offRes, int pos,
                         int numHeads, int kvHeads, int headDim);

void Metal_RMSNormLinear_F16(MetalContextRef ctx, MetalBufferRef input,
                             int offIn, MetalBufferRef normWeight,
                             int offNormWeight, MetalBufferRef weight,
                             int offWeight, MetalBufferRef result, int offRes,
                             int inDim, int outDim, float eps);

// Matrix Multiplication (MPS)
void Metal_MatMul_F16(MetalContextRef ctx, MetalBufferRef a, int offA,
                      bool transA, MetalBufferRef b, int offB, bool transB,
                      MetalBufferRef c, int offC, int M, int N, int K);

void Metal_BatchedMatMul_F16(MetalContextRef ctx, MetalBufferRef a, int offA,
                             int strideA, bool transA, MetalBufferRef b,
                             int offB, int strideB, bool transB,
                             MetalBufferRef c, int offC, int strideC, int M,
                             int N, int K, int batchCount);

#ifdef __cplusplus
}
#endif

#endif
