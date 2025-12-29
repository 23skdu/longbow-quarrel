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
void Metal_ZeroBuffer(MetalBufferRef buf, int offset, int size);

// Basic Ops
void Metal_Add_F16(MetalContextRef ctx, MetalBufferRef a, int offA,
                   MetalBufferRef b, int offB, MetalBufferRef result,
                   int offRes, int count);
void Metal_Scale_F16(MetalContextRef ctx, MetalBufferRef x, int offX,
                     float scale, MetalBufferRef result, int offRes, int count);

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
                    int posOffset, float ropeTheta);

void Metal_SwiGLU_F16(MetalContextRef ctx, MetalBufferRef inputVal, int offVal,
                      MetalBufferRef inputGate, int offGate,
                      MetalBufferRef output, int offOut, int n, int interSize);

void Metal_Softmax_F16(MetalContextRef ctx, MetalBufferRef input, int offIn,
                       MetalBufferRef result, int offRes, int rows, int cols);

void Metal_StoreKV_F16(MetalContextRef ctx, MetalBufferRef k, int offK,
                       MetalBufferRef v, int offV, MetalBufferRef kCache,
                       MetalBufferRef vCache, int pos, int heads, int headDim);

void Metal_Attention_F16(MetalContextRef ctx, MetalBufferRef q, int offQ,
                         MetalBufferRef kC, MetalBufferRef vC, MetalBufferRef r,
                         int oR, MetalBufferRef s, int oS, int p, int nh,
                         int kh, int hd, int ctxLen);

void Metal_AttScores_F16(MetalContextRef ctx, MetalBufferRef q, int offQ,
                         MetalBufferRef kC, MetalBufferRef s, int oS, int p,
                         int nh, int kh, int hd, int ctxLen);

void Metal_AttSoftmax_F16(MetalContextRef ctx, MetalBufferRef s, int oS, int p,
                          int nh, int ctxLen);

void Metal_AttValues_F16(MetalContextRef ctx, MetalBufferRef s, int oS,
                         MetalBufferRef vC, MetalBufferRef r, int oR, int p,
                         int nh, int kh, int hd, int ctxLen);

void Metal_RMSNormLinear_F16(MetalContextRef ctx, MetalBufferRef input,
                             int offIn, MetalBufferRef normWeight,
                             int offNormWeight, MetalBufferRef weight,
                             int offWeight, MetalBufferRef result, int offRes,
                             int inDim, int outDim, float eps);

// Matrix Multiplication (MPS)
void Metal_MatMul_F16(MetalContextRef ctx, MetalBufferRef a, int offA,
                      bool transA, MetalBufferRef b, int offB, bool transB,
                      MetalBufferRef c, int offC, int M, int N, int K);

void Metal_MatMul_Q4K_F16(MetalContextRef ctx, MetalBufferRef a, int offA,
                          bool transA, MetalBufferRef b, int offB, bool transB,
                          MetalBufferRef c, int offC, int M, int N, int K);

void Metal_MatMul_Q3K_F16(MetalContextRef ctx, MetalBufferRef a, int offA,
                          bool transA, MetalBufferRef b, int offB, bool transB,
                          MetalBufferRef c, int offC, int M, int N, int K);

void Metal_MatMul_Q6K_F16(MetalContextRef ctx, MetalBufferRef a, int offA,
                          bool transA, MetalBufferRef b, int offB, bool transB,
                          MetalBufferRef c, int offC, int M, int N, int K);

void Metal_BatchedMatMul_F16(MetalContextRef ctx, MetalBufferRef a, int offA,
                             int strideA, bool transA, MetalBufferRef b,
                             int offB, int strideB, bool transB,
                             MetalBufferRef c, int offC, int strideC, int M,
                             int N, int K, int batchCount);

void Metal_RMSNormQKV_F16(MetalContextRef ctx, MetalBufferRef input, int offIn,
                          MetalBufferRef normWeight, int offNormWeight,
                          MetalBufferRef qWeight, int offQW,
                          MetalBufferRef kWeight, int offKW,
                          MetalBufferRef vWeight, int offVW,
                          MetalBufferRef qOut, int offQO, MetalBufferRef kOut,
                          int offKO, MetalBufferRef vOut, int offVO, int inDim,
                          int qDim, int kvDim, float eps);

void Metal_FusedFFN_F16(MetalContextRef ctx, MetalBufferRef input, int offIn,
                        MetalBufferRef normWeight, int offNormWeight,
                        MetalBufferRef gateWeight, int offGW,
                        MetalBufferRef upWeight, int offUW,
                        MetalBufferRef downWeight, int offDW,
                        MetalBufferRef output, int offOut, int inDim,
                        int interDim, float eps);

// FP32 Ops
void Metal_RMSNorm_F32(MetalContextRef ctx, MetalBufferRef input, int offIn,
                       MetalBufferRef weight, int offWeight,
                       MetalBufferRef result, int offRes, int rows, int cols,
                       float eps);

void Metal_MatMul_Q4K_F32(MetalContextRef ctx, MetalBufferRef a, int offA,
                          int transA, MetalBufferRef b, int offB, int transB,
                          MetalBufferRef c, int offC, int M, int N, int K);

void Metal_Add_F32(MetalContextRef ctx, MetalBufferRef a, int offA,
                   MetalBufferRef b, int offB, MetalBufferRef result,
                   int offRes, int count);

void Metal_Copy_F16(MetalContextRef ctx, MetalBufferRef src, int oS,
                    MetalBufferRef dst, int oD, int count);

void Metal_Copy_F16_F32(MetalContextRef ctx, MetalBufferRef src, int oS,
                        MetalBufferRef dst, int oD, int n);

void Metal_Copy_F32_F16(MetalContextRef ctx, MetalBufferRef src, int oS,
                        MetalBufferRef dst, int oD, int n);

void Metal_SwiGLU_F32(MetalContextRef ctx, MetalBufferRef iV, int oV,
                      MetalBufferRef iG, int oG, MetalBufferRef o, int oO,
                      int n, int iS);

void Metal_MatMul_F16_F32(MetalContextRef ctx, MetalBufferRef a, int offA,
                          MetalBufferRef b, int offB, MetalBufferRef c,
                          int offC, int M, int N, int K);

void Metal_AttFused_F16(MetalContextRef ctx, MetalBufferRef q, int offQ,
                        MetalBufferRef kC, MetalBufferRef vC, MetalBufferRef r,
                        int oR, int p, int nh, int kh, int hd);

#ifdef __cplusplus
}
#endif

#endif
