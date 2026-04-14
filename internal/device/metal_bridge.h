#ifndef METAL_BRIDGE_H
#define METAL_BRIDGE_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque pointer types
typedef void *MetalWrapperRef;
typedef void *MetalBufferRef;

void Metal_LinearQ6K_F16_F32(MetalWrapperRef ctx, MetalBufferRef weight,
                             int offWeight, MetalBufferRef input, int offInput,
                             MetalBufferRef output, int offOutput, int rows,
                             int dimIn, int dimOut, float scale);

void Metal_LinearQ4K_F16_F32(MetalWrapperRef ctx, MetalBufferRef weight,
                              int offWeight, MetalBufferRef input, int offInput,
                              MetalBufferRef output, int offOutput, int rows,
                              int dimIn, int dimOut, float scale);

void Metal_LinearQ4_0_F16(MetalWrapperRef ctx, MetalBufferRef weight,
                          int offWeight, MetalBufferRef input, int offInput,
                          MetalBufferRef output, int offOutput, int rows,
                          int dimIn, int dimOut, float scale);

void Metal_LinearQ4_0_F32(MetalWrapperRef ctx, MetalBufferRef weight,
                          int offWeight, MetalBufferRef input, int offInput,
                          MetalBufferRef output, int offOutput, int rows,
                          int dimIn, int dimOut, float scale);

void Metal_LinearQ8_0_F16(MetalWrapperRef ctx, MetalBufferRef weight,
                          int offWeight, MetalBufferRef input, int offInput,
                          MetalBufferRef output, int offOutput, int rows,
                          int dimIn, int dimOut, float scale);

void Metal_LinearQ8_0_F32(MetalWrapperRef ctx, MetalBufferRef weight,
                          int offWeight, MetalBufferRef input, int offInput,
                          MetalBufferRef output, int offOutput, int rows,
                          int dimIn, int dimOut, float scale);

void Metal_EmbeddingQ4_0_F16(MetalWrapperRef ctx, MetalBufferRef weights,
                             int offW, MetalBufferRef result, int offRes,
                             int rowIdx, int cols);
// Device management
MetalWrapperRef Metal_Init(const char *libSource);
void Metal_Free(MetalWrapperRef ctx);
void Metal_Synchronize(MetalWrapperRef ctx);

// Buffer Management
MetalBufferRef Metal_Alloc(MetalWrapperRef ctx, long long size);
void Metal_FreeBuffer(MetalWrapperRef ctx, MetalBufferRef buf);
void Metal_CopyToDevice(MetalBufferRef buf, int offset, const void *data,
                        int size);
void Metal_CopyToHost(MetalBufferRef buf, int offset, void *data, int size);
void *Metal_GetBufferContents(MetalBufferRef buf);
void Metal_ZeroBuffer(MetalBufferRef buf, int offset, int size);
void Metal_ZeroBufferGPU(MetalWrapperRef ctx, MetalBufferRef buf, int offset,
                         int size);

// Basic Ops
void Metal_Add_F16(MetalWrapperRef ctx, MetalBufferRef a, int offA,
                   MetalBufferRef b, int offB, MetalBufferRef result,
                   int offRes, int count);
void Metal_Scale_F16(MetalWrapperRef ctx, MetalBufferRef x, int offX,
                     float scale, MetalBufferRef result, int offRes, int count);

void Metal_Embedding_F16(MetalWrapperRef ctx, MetalBufferRef weights, int offW,
                         MetalBufferRef result, int offRes, int rowIdx,
                         int cols);
void Metal_Embedding_Q4K(MetalWrapperRef ctx, MetalBufferRef weights, int offW,
                         MetalBufferRef result, int offRes, int rowIdx,
                         int cols, float scale);
void Metal_Embedding_Q4K_Optimized(MetalWrapperRef ctx, MetalBufferRef weights,
                                   int offW, MetalBufferRef result, int offRes,
                                   int rowIdx, int cols, float scale);

// Llama Specific Kernels
void Metal_RMSNorm_F16(MetalWrapperRef ctx, MetalBufferRef input, int offIn,
                       MetalBufferRef weight, int offWeight,
                       MetalBufferRef result, int offRes, int rows, int cols,
                       float eps);

void Metal_RoPE_F16(MetalWrapperRef ctx, MetalBufferRef data, int offData,
                    int batchSize, int seqLen, int numHeads, int headDim,
                    int posOffset, float ropeTheta);

void Metal_RoPE_Ragged_F16(MetalWrapperRef ctx, MetalBufferRef data, int offData,
                           MetalBufferRef positions, int numTokens, int numHeads,
                           int headDim, float ropeTheta);

void Metal_SwiGLU_F16(MetalWrapperRef ctx, MetalBufferRef inputVal, int offVal,
                      MetalBufferRef inputGate, int offGate,
                      MetalBufferRef output, int offOut, int n, int interSize);

void Metal_Softmax_F16(MetalWrapperRef ctx, MetalBufferRef input, int offIn,
                     MetalBufferRef output, int offOut, int rows, int cols);

void Metal_StoreKV_F16_Batch(MetalWrapperRef ctx, MetalBufferRef k, int offK,
                             MetalBufferRef v, int offV, MetalBufferRef kCache,
                             int offKC, MetalBufferRef vCache, int offVC,
                             int pos, int heads, int headDim, int windowSize,
                             int batchSize);
void Metal_StoreKV_F16(MetalWrapperRef ctx, MetalBufferRef k, int offK,
                       MetalBufferRef v, int offV, MetalBufferRef kCache,
                       int offKC, MetalBufferRef vCache, int offVC, int pos,
                       int heads, int headDim, int windowSize);

void Metal_Attention_F16(MetalWrapperRef ctx, MetalBufferRef q, int offQ,
                         MetalBufferRef kC, int offK, MetalBufferRef vC,
                         int offV, MetalBufferRef r, int oR, MetalBufferRef s,
                         int oS, int p, int nh, int kh, int hd, int ctxLen,
                         int windowSize);

void Metal_AttScores_F16(MetalWrapperRef ctx, MetalBufferRef q, int offQ,
                         MetalBufferRef kC, int offK, MetalBufferRef s, int oS,
                         int p, int nh, int kh, int hd, int ctxLen,
                         int windowSize);

void Metal_AttSoftmax_F16(MetalWrapperRef ctx, MetalBufferRef s, int oS, int p,
                          int nh, int ctxLen);

void Metal_AttValues_F16(MetalWrapperRef ctx, MetalBufferRef s, int oS,
                         MetalBufferRef vC, int offV, MetalBufferRef r, int oR,
                         int p, int nh, int kh, int hd, int ctxLen,
                         int windowSize);

void Metal_RMSNormLinear_F16(MetalWrapperRef ctx, MetalBufferRef input,
                             int offIn, MetalBufferRef normWeight,
                             int offNormWeight, MetalBufferRef weight,
                             int offWeight, MetalBufferRef result, int offRes,
                             int inDim, int outDim, float eps, int batchSize);

// Matrix Multiplication (MPS)
void Metal_MatMul_F16(MetalWrapperRef ctx, MetalBufferRef a, int offA,
                      bool transA, MetalBufferRef b, int offB, bool transB,
                      MetalBufferRef c, int offC, int M, int N, int K);

void Metal_MatMul_Q4K_F16(MetalWrapperRef ctx, MetalBufferRef a, int offA,
                          bool transA, MetalBufferRef b, int offB, bool transB,
                          MetalBufferRef c, int offC, int M, int N, int K,
                          float scale);

void Metal_MatMul_Q3K_F16(MetalWrapperRef ctx, MetalBufferRef a, int offA,
                          bool transA, MetalBufferRef b, int offB, bool transB,
                          MetalBufferRef c, int offC, int M, int N, int K,
                          float scale);

void Metal_MatMul_Q6K_F16(MetalWrapperRef ctx, MetalBufferRef a, int offA,
                          bool transA, MetalBufferRef b, int offB, bool transB,
                          MetalBufferRef c, int offC, int M, int N, int K,
                          float scale);

void Metal_BatchedMatMul_F16(MetalWrapperRef ctx, MetalBufferRef a, int offA,
                             int strideA, bool transA, MetalBufferRef b,
                             int offB, int strideB, bool transB,
                             MetalBufferRef c, int offC, int strideC, int M,
                             int N, int K, int batchCount);

void Metal_RMSNormQKV_Q4K_F16(
    MetalWrapperRef ctx, MetalBufferRef input, int offIn,
    MetalBufferRef normWeight, int offNormWeight, MetalBufferRef qWeight,
    int offQW, MetalBufferRef kWeight, int offKW, MetalBufferRef vWeight,
    int offVW, MetalBufferRef qOut, int offQO, MetalBufferRef kOut, int offKO,
    MetalBufferRef vOut, int offVO, int inDim, int qDim, int kvDim, float eps,
    float scale, int batchSize);
void Metal_RMSNormQKV_F16(MetalWrapperRef ctx, MetalBufferRef input, int offIn,
                          MetalBufferRef normWeight, int offNormWeight,
                          MetalBufferRef qWeight, int offQW,
                          MetalBufferRef kWeight, int offKW,
                          MetalBufferRef vWeight, int offVW,
                          MetalBufferRef qOut, int offQO, MetalBufferRef kOut,
                          int offKO, MetalBufferRef vOut, int offVO, int inDim,
                          int qDim, int kvDim, float eps, int batchSize);

void Metal_FusedFFN_F16(MetalWrapperRef ctx, MetalBufferRef input, int offIn,
                        MetalBufferRef normWeight, int offNormWeight,
                        MetalBufferRef gateWeight, int offGW,
                        MetalBufferRef upWeight, int offUW,
                        MetalBufferRef downWeight, int offDW,
                        MetalBufferRef output, int offOut, int inDim,
                        int interDim, float eps, int batchSize);

void Metal_RMSNormLinear_Q4K_F16(MetalWrapperRef ctx, MetalBufferRef input,
                                 int offIn, MetalBufferRef normWeight,
                                 int offNormWeight, MetalBufferRef weight,
                                 int offWeight, MetalBufferRef result,
                                 int offRes, int M, int N, int K, float eps,
                                 float scale, int batchSize);

void Metal_SwiGLULinear_Q4K_F16(MetalWrapperRef ctx, MetalBufferRef gateIn,
                                int offGate, MetalBufferRef upIn, int offUp,
                                MetalBufferRef weight, int offWeight,
                                MetalBufferRef result, int offRes, int M, int N,
                                int K, float scale);

void Metal_RMSNormLinear_Q6K_F16(MetalWrapperRef ctx, MetalBufferRef input,
                                 int offIn, MetalBufferRef normWeight,
                                 int offNormWeight, MetalBufferRef weight,
                                 int offWeight, MetalBufferRef result,
                                 int offRes, int M, int N, int K, float eps,
                                 float scale, int batchSize);

void Metal_SwiGLULinear_Q6K_F16(MetalWrapperRef ctx, MetalBufferRef gateIn,
                                int offGate, MetalBufferRef upIn, int offUp,
                                MetalBufferRef weight, int offWeight,
                                MetalBufferRef result, int offRes, int M, int N,
                                int K, float scale);

void Metal_RMSNormQKV_Q6K_F16(
    MetalWrapperRef ctx, MetalBufferRef input, int offIn,
    MetalBufferRef normWeight, int offNormWeight, MetalBufferRef qWeight,
    int offQW, MetalBufferRef kWeight, int offKW, MetalBufferRef vWeight,
    int offVW, MetalBufferRef qOut, int offQO, MetalBufferRef kOut, int offKO,
    MetalBufferRef vOut, int offVO, int dimIn, int qDim, int kvDim, float eps,
    float scale, int batchSize);

// FP32 Ops
void Metal_RMSNorm_F32(MetalWrapperRef ctx, MetalBufferRef input, int offIn,
                       MetalBufferRef weight, int offWeight,
                       MetalBufferRef result, int offRes, int rows, int cols,
                       float eps);

void Metal_MatMul_Q4K_F32(MetalWrapperRef ctx, MetalBufferRef a, int offA,
                          int transA, MetalBufferRef b, int offB, int transB,
                          MetalBufferRef c, int offC, int M, int N, int K,
                          float scale);

void Metal_MatMul_Q6K_F32(MetalWrapperRef ctx, MetalBufferRef a, int offA,
                          int transA, MetalBufferRef b, int offB, int transB,
                          MetalBufferRef c, int offC, int M, int N, int K,
                          float scale);

void Metal_MatMul_Q4K_F32_F16(MetalWrapperRef ctx, MetalBufferRef a, int offA,
                              MetalBufferRef b, int offB, MetalBufferRef c,
                              int offC, int M, int N, int K, float scale);

void Metal_MatMul_Q6K_F32_F16(MetalWrapperRef ctx, MetalBufferRef a, int offA,
                              MetalBufferRef b, int offB, MetalBufferRef c,
                              int offC, int M, int N, int K, float scale);

void Metal_DebugRoPEFreq(MetalWrapperRef ctx, MetalBufferRef output,
                         int headDim, float theta, int pos);

void Metal_DebugDot(MetalWrapperRef ctx, MetalBufferRef a, MetalBufferRef b,
                    MetalBufferRef output, int dim);

void Metal_Add_F32(MetalWrapperRef ctx, MetalBufferRef a, int offA,
                   MetalBufferRef b, int offB, MetalBufferRef c, int offC,
                   int count);

void Metal_CopyBufferToF32(MetalWrapperRef ctx, MetalBufferRef src, void *dst,
                           int count);

void Metal_Copy_F16(MetalWrapperRef ctx, MetalBufferRef src, int oS,
                    MetalBufferRef dst, int oD, int count);

void Metal_Copy_F16_F32(MetalWrapperRef ctx, MetalBufferRef src, int oS,
                        MetalBufferRef dst, int oD, int n);

void Metal_Copy_F32_F16(MetalWrapperRef ctx, MetalBufferRef src, int oS,
                        MetalBufferRef dst, int oD, int n);

void Metal_Copy_F32(MetalWrapperRef ctx, MetalBufferRef src, int oS,
                    MetalBufferRef dst, int oD, int n);

void Metal_SwiGLU_F32(MetalWrapperRef ctx, MetalBufferRef iV, int oV,
                      MetalBufferRef iG, int oG, MetalBufferRef o, int oO,
                      int n, int iS);

void Metal_MatMul_F16_F16_F32(MetalWrapperRef ctx, MetalBufferRef weight,
                              int offWeight, MetalBufferRef input, int offInput,
                              MetalBufferRef output, int offOutput, int rows,
                              int dimIn, int dimOut);

void Metal_MatMul_F16_F32_F32(MetalWrapperRef ctx, MetalBufferRef weight,
                              int offWeight, MetalBufferRef input, int offInput,
                              MetalBufferRef output, int offOutput, int rows,
                              int dimIn, int dimOut);

void Metal_AttFused_F16(MetalWrapperRef ctx, MetalBufferRef q, int offQ,
                        MetalBufferRef kC, int offK, MetalBufferRef vC,
                        int offV, MetalBufferRef r, int offR, int p, int nh,
                        int kh, int hd, int windowSize, int maxCtxLen);

// FP32 FFN Kernels for Small Models
void Metal_LinearF16ToF32(MetalWrapperRef ctx, MetalBufferRef weight,
                          int offWeight, MetalBufferRef input, int offInput,
                          MetalBufferRef output, int offOutput, int rows,
                          int dimIn, int dimOut);

void Metal_LinearF32ToF16(MetalWrapperRef ctx, MetalBufferRef weight,
                          int offWeight, MetalBufferRef input, int offInput,
                          MetalBufferRef output, int offOutput, int rows,
                          int dimIn, int dimOut);

void Metal_RMSNorm_F32_F16(MetalWrapperRef ctx, MetalBufferRef input, int offIn,
                           MetalBufferRef weight, int offWeight,
                           MetalBufferRef result, int offRes, int rows,
                           int cols, float eps);

void Metal_Add_F32_F16(MetalWrapperRef ctx, MetalBufferRef a, int offA,
                       MetalBufferRef b, int offB, MetalBufferRef result,
                       int offRes, int count);

void Metal_AttPaged_F16(MetalWrapperRef ctx, MetalBufferRef q, int offQ,
                        MetalBufferRef kC, int offK, MetalBufferRef vC,
                        int offV, MetalBufferRef r, int offR,
                        MetalBufferRef blockTable, int offBT, int p, int nh,
                        int kh, int hd, int blockSize, int maxCtxLen);

// Mamba / SSM Kernels
void Metal_MambaConv1d_F16(MetalWrapperRef ctx, MetalBufferRef input, int offIn,
                           MetalBufferRef weight, int offW, MetalBufferRef bias,
                           int offB, MetalBufferRef state, int offS,
                           MetalBufferRef output, int offOut, int dim,
                           int kernelSize);

void Metal_MambaScan_F16(MetalWrapperRef ctx, MetalBufferRef u, int offU,
                         MetalBufferRef h, int offH, MetalBufferRef A, int offA,
                         MetalBufferRef B, int offB, MetalBufferRef C, int offC,
                         MetalBufferRef D, int offD, MetalBufferRef dt,
                         int offDt, MetalBufferRef y, int offY, int dim,
                         int d_state);

void Metal_SiLU_F16(MetalWrapperRef ctx, MetalBufferRef input, int offIn,
                    MetalBufferRef output, int offOut, int n);

void Metal_Slice_F16(MetalWrapperRef ctx, MetalBufferRef input, int offIn,
                     MetalBufferRef output, int offOut, int startCol,
                     int numCols, int totalCols, int rows);

void Metal_Mul_F16(MetalWrapperRef ctx, MetalBufferRef a, int offA,
                   MetalBufferRef b, int offB, MetalBufferRef result,
                   int offRes, int n);

// MOE (Mixture of Experts) Kernels
void Metal_MOE_RouterLogits(MetalWrapperRef ctx, MetalBufferRef input,
                            int offInput, MetalBufferRef gateWeight,
                            int offGateWeight, MetalBufferRef logits,
                            int offLogits, int batchSize, int dim,
                            int numExperts);

void Metal_MOE_TopKSelection(MetalWrapperRef ctx, MetalBufferRef logits,
                             int offLogits, MetalBufferRef expertIndices,
                             int offIndices, MetalBufferRef expertWeights,
                             int offWeights, int batchSize, int numExperts,
                             int topK);

void Metal_MOE_ExpertForward(MetalWrapperRef ctx, MetalBufferRef input,
                             int offInput, MetalBufferRef expertWeight,
                             int offWeight, MetalBufferRef expertIndices,
                             int offIndices, MetalBufferRef expertWeights,
                             int offWeights, MetalBufferRef output,
                             int offOutput, int batchSize, int dim,
                             int hiddenDim, int topK);

void Metal_MOE_ExpertGateUpSwiGLU(MetalWrapperRef ctx, MetalBufferRef input,
                                  int offInput, MetalBufferRef gateWeight,
                                  int offGate, MetalBufferRef upWeight,
                                  int offUp, MetalBufferRef expertIndices,
                                  int offIndices, MetalBufferRef expertWeights,
                                  int offWeights, MetalBufferRef output,
                                  int offOutput, int batchSize, int dim,
    int hiddenDim, int topK);

void Metal_Softmax_F16(MetalWrapperRef ctx, MetalBufferRef input, int offIn,
                       MetalBufferRef output, int offOut, int n, int rows);

// TurboQuant Kernels
void Metal_TurboQuant_PolarQuant(MetalWrapperRef ctx, MetalBufferRef input,
                                  int offInput, MetalBufferRef rotationMatrix,
                                  int offRot, MetalBufferRef quantized,
                                  int offQuant, MetalBufferRef scaleOut,
                                  int offScale, MetalBufferRef residual,
                                  int offRes, int n, int numBlocks, int bits);
void Metal_TurboQuant_QJLTransform(MetalWrapperRef ctx, MetalBufferRef residual,
                                    int offRes, MetalBufferRef signMatrix,
                                    int offSign, MetalBufferRef quantized,
                                    int offQuant, MetalBufferRef scaleOut,
                                    int offScale, int rows, int cols, int numBlocks);
void Metal_TurboQuant_Encode(MetalWrapperRef ctx, MetalBufferRef input,
                              int offInput, MetalBufferRef rotationMatrix,
                              int offRot, MetalBufferRef qjlMatrix,
                              int offQJL, MetalBufferRef output,
                              int offOut, MetalBufferRef scaleOut,
                              int offScale, MetalBufferRef qjlScaleOut,
                              int offQJLScale, int blockSize, int qjlRows, int numBlocks, int bits);
void Metal_TurboQuant_Decode(MetalWrapperRef ctx, MetalBufferRef input,
                              int offInput, MetalBufferRef rotationMatrix,
                              int offRot, MetalBufferRef qjlMatrix,
                              int offQJL, MetalBufferRef output,
                              int offOut, MetalBufferRef scaleIn,
                              int offScale, int blockSize, int qjlRows, int numBlocks);
void Metal_Attention_TQ_Scores_F16(MetalWrapperRef ctx, MetalBufferRef q_prime, int offQP,
                                    MetalBufferRef q_double_prime, int offQDP,
                                    MetalBufferRef k_cache, int offKC,
                                    MetalBufferRef scores, int offS,
                                    int head_dim, int qjl_rows, int pos,
                                    int num_heads, int kv_heads, float sm_scale);
void Metal_Prepare_TQ_Query(MetalWrapperRef ctx, MetalBufferRef q, int offQ,
                             MetalBufferRef rotationMatrix, int offRot,
                             MetalBufferRef qjlMatrix, int offQJL,
                             MetalBufferRef q_prime, int offQP,
                             MetalBufferRef q_double_prime, int offQDP,
                             int head_dim, int qjl_rows, int num_heads);
void Metal_Attention_TQ_Values_F16(MetalWrapperRef ctx, MetalBufferRef probabilities, int offP,
                                    MetalBufferRef v_cache, int offVC,
                                    MetalBufferRef output, int offOut,
                                    int head_dim, int qjl_rows, int pos,
                                    int num_heads, int kv_heads);

void Metal_AttPagedBatch_F16(MetalWrapperRef ctx, MetalBufferRef q, int offQ,
                             MetalBufferRef kC, int offK, MetalBufferRef vC,
                             int offV, MetalBufferRef r, int offR,
                             MetalBufferRef batchPositions, int offBP,
                             MetalBufferRef blockTables, int offBT,
                             int maxBlocksPerSeq, int nh, int kh, int hd,
                             int blockSize, int batchSize);

void Metal_StoreKVPagedBatch_F16(MetalWrapperRef ctx, MetalBufferRef k, int offK,
                                 MetalBufferRef v, int offV,
                                 MetalBufferRef kCache, int offKC,
                                 MetalBufferRef vCache, int offVC,
                                 MetalBufferRef physicalPositions, int offPP,
                                 int kvDim, int batchSize);

#ifdef __cplusplus
}
#endif

#endif

void Metal_FlashAttention2_F16(MetalWrapperRef ctx, MetalBufferRef q,
                               MetalBufferRef k_cache, MetalBufferRef v_cache,
                               MetalBufferRef output, int num_heads,
                               int kv_heads, int headDim, MetalBufferRef seq_lens,
                               int block_size, MetalBufferRef block_table,
                               int max_blocks_per_seq, MetalBufferRef token_to_seq,
                               int batchSize);

void Metal_Linear_LoRA_Add_F16(MetalWrapperRef ctx, MetalBufferRef input, int offIn,
                                MetalBufferRef A, int offA, MetalBufferRef B, int offB,
                                MetalBufferRef output, int offOut, int M, int N, int K, int R,
                                float scale);

void Metal_Vision_Patch_Embed_F32(MetalWrapperRef ctx, MetalBufferRef pixels, int offPixels, MetalBufferRef weights, int offW, MetalBufferRef output, int offOut, int patchSize, int visionDim, int numPatchesX);
void Metal_Vision_Patch_Embed_Gemma4(MetalWrapperRef ctx, MetalBufferRef pixels, int offPixels,
                                  MetalBufferRef weights, int offW, MetalBufferRef bias, int offB,
                                  MetalBufferRef output, int offOut, int patchSize, int hiddenDim, int numPatches);

void Metal_Dequantize_Q2_K(MetalWrapperRef ctx, MetalBufferRef data, int offData, MetalBufferRef output, int offOut, int numElements);
void Metal_Dequantize_IQ4_XS(MetalWrapperRef ctx, MetalBufferRef data, int offData, MetalBufferRef output, int offOut, int numElements);

// Multi-GPU Collective Operations
void Metal_AllReduce_F16(MetalWrapperRef ctx, MetalBufferRef data, int offset, int count);
void Metal_AllGather_F16(MetalWrapperRef ctx, MetalBufferRef input, int offIn, MetalBufferRef output, int offOut, int count);
