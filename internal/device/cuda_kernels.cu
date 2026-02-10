#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

// =============================================================================
// Standard Operations
// =============================================================================

__global__ void rmsnorm_kernel(float* input, float* weight, float* output, int rows, int cols, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int i = idx; i < rows; i += gridDim.x * blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            float val = input[i * cols + j];
            sum += val * val;
        }
        float rms = sqrtf(sum / cols + eps);
        
        for (int j = 0; j < cols; j++) {
            output[i * cols + j] = (input[i * cols + j] / rms) * weight[j];
        }
    }
}

__global__ void swiglu_kernel(float* gate, float* up, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int i = idx; i < size; i += gridDim.x * blockDim.x) {
        float g = gate[i];
        float u = up[i];
        float swish = g / (1.0f + expf(-g));
        output[i] = swish * u;
    }
}

__global__ void softmax_kernel(float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;
    
    if (row < rows) {
        float maxVal = -INFINITY;
        for (int j = 0; j < cols; j++) {
            maxVal = fmaxf(maxVal, input[row * cols + j]);
        }
        
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            sum += expf(input[row * cols + j] - maxVal);
        }
        
        for (int j = 0; j < cols; j++) {
            output[row * cols + j] = expf(input[row * cols + j] - maxVal) / sum;
        }
    }
}

__global__ void rope_kernel(float* tensor, int pos, int heads, int headDim, float theta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int halfDim = headDim / 2;
    
    for (int i = idx; i < heads * halfDim; i += gridDim.x * blockDim.x) {
        int head = i / halfDim;
        int dim = i % halfDim;
        
        float freq = pos * powf(theta, -2.0f * dim / headDim);
        float cosVal = cosf(freq);
        float sinVal = sinf(freq);
        
        int base = head * headDim;
        float x0 = tensor[base + dim];
        float x1 = tensor[base + dim + halfDim];
        
        tensor[base + dim] = x0 * cosVal - x1 * sinVal;
        tensor[base + dim + halfDim] = x0 * sinVal + x1 * cosVal;
    }
}

__global__ void add_kernel(float* a, float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int i = idx; i < size; i += gridDim.x * blockDim.x) {
        out[i] = a[i] + b[i];
    }
}

__global__ void silu_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int i = idx; i < size; i += gridDim.x * blockDim.x) {
        float x = input[i];
        output[i] = x / (1.0f + expf(-x));
    }
}

// =============================================================================
// MOE (Mixture of Experts) Kernels
// =============================================================================

// Top-K selection with softmax
__global__ void moe_topk_kernel(float* logits, int* indices, float* weights,
                                  int batch, int num_experts, int top_k) {
    int token = blockIdx.x;
    if (token >= batch) return;

    float* logit_row = logits + token * num_experts;
    int* idx_row = indices + token * top_k;
    float* weight_row = weights + token * top_k;

    // Find top-k using simple selection
    for (int k = 0; k < top_k; k++) {
        float max_val = -FLT_MAX;
        int max_idx = 0;

        for (int e = 0; e < num_experts; e++) {
            bool selected = false;
            for (int prev = 0; prev < k; prev++) {
                if (idx_row[prev] == e) {
                    selected = true;
                    break;
                }
            }
            if (selected) continue;

            if (logit_row[e] > max_val) {
                max_val = logit_row[e];
                max_idx = e;
            }
        }

        idx_row[k] = max_idx;

        // Compute softmax weight
        float sum = 0.0f;
        for (int e = 0; e < num_experts; e++) {
            bool sel = false;
            for (int prev = 0; prev <= k; prev++) {
                if (idx_row[prev] == e) { sel = true; break; }
            }
            if (!sel) continue;
            sum += expf(logit_row[e] - max_val);
        }
        weight_row[k] = expf(max_val - max_val) / (sum + 1e-9f);
    }
}

// Router logits computation: input @ gate.T
__global__ void moe_router_logits_kernel(
    float* input,      // [batch, dim]
    float* gate,       // [dim, num_experts]
    float* output,     // [batch, num_experts]
    int batch, int dim, int num_experts) {

    int b = blockIdx.x;
    int e = threadIdx.x;

    if (b >= batch || e >= num_experts) return;

    float sum = 0.0f;
    for (int d = 0; d < dim; d++) {
        sum += input[b * dim + d] * gate[e * dim + d];
    }
    output[b * num_experts + e] = sum;
}

// Fused Gate + Up + SwiGLU for multiple experts
__global__ void moe_gate_up_swiglu_kernel(
    float* input,          // [batch, dim]
    float* gate_experts,   // [hidden_dim, dim, num_experts]
    float* up_experts,     // [hidden_dim, dim, num_experts]
    int* indices,         // [batch, top_k]
    float* expert_weights, // [batch, top_k]
    float* output,         // [batch, hidden_dim]
    int batch, int dim, int hidden_dim, int num_experts, int top_k) {

    int b = blockIdx.x;
    int h = threadIdx.x;

    if (b >= batch || h >= hidden_dim) return;

    float sum = 0.0f;

    for (int k = 0; k < top_k; k++) {
        int expert = indices[b * top_k + k];
        float weight = expert_weights[b * top_k + k];

        // Gate computation
        float gate_val = 0.0f;
        for (int d = 0; d < dim; d++) {
            gate_val += input[b * dim + d] * gate_experts[((size_t)expert * hidden_dim + h) * dim + d];
        }
        gate_val = gate_val / (1.0f + expf(-gate_val));

        // Up computation
        float up_val = 0.0f;
        for (int d = 0; d < dim; d++) {
            up_val += input[b * dim + d] * up_experts[((size_t)expert * hidden_dim + h) * dim + d];
        }

        sum += weight * gate_val * up_val;
    }

    output[b * hidden_dim + h] = sum;
}

// Expert forward pass
__global__ void moe_expert_forward_kernel(
    float* input,          // [batch, dim]
    float* expert_weights, // [hidden_dim, dim, num_experts]
    int* indices,          // [batch, top_k]
    float* expert_weights_w, // [batch, top_k]
    float* output,          // [batch, dim]
    int batch, int dim, int hidden_dim, int num_experts, int top_k) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * dim;

    for (int idx = tid; idx < total; idx += gridDim.x * blockDim.x) {
        int b = idx / dim;
        int d = idx % dim;

        float sum = 0.0f;
        for (int k = 0; k < top_k; k++) {
            int expert = indices[b * top_k + k];
            float weight = expert_weights_w[b * top_k + k];
            float expert_out = input[b * dim + d];
            sum += weight * expert_out;
        }
            output[idx] = sum;
    }
}

// =============================================================================
// Dequantization Kernels
// =============================================================================

// Helper: Convert FP16 to FP32
__device__ __forceinline__ float fp16_to_fp32(unsigned short h) {
    unsigned int sign = (h & 0x8000) << 16;
    unsigned int exp = (h & 0x7C00) >> 10;
    unsigned int frac = (h & 0x03FF) << 13;

    if (exp == 0) {
        if (frac == 0) {
            return __int_as_float(sign);
        }
        // subnormal
        unsigned int subnormal_bits = sign | ((frac + 0x3F800000U) >> 13);
        float subnormal = __int_as_float(subnormal_bits) * 5.960464477539063e-8f;
        return subnormal;
    }
    if (exp == 31) {
        if (frac == 0) {
            return __int_as_float(sign | 0x7F800000);  // Infinity
        }
        return __int_as_float(sign | 0x7FC00000);  // NaN
    }
    return __int_as_float(sign | ((exp + 112) << 23) | frac);
}

// Helper: Convert FP32 to BF16
__device__ __forceinline__ unsigned short fp32_to_bf16(float f) {
    unsigned int bits = __float_as_uint(f);
    unsigned short bf16 = (bits >> 16);
    // Round to nearest even
    if ((bits & 0x00007FFF) > 0x00004000) {
        bf16 += 1;
    }
    return bf16;
}

// Q8_0 Dequantization: 32 elements per block, 34 bytes (2 bytes scale + 32 int8)
__global__ void dequant_q8_0_kernel(const unsigned char* __restrict__ src,
                                      float* __restrict__ dst,
                                      int numElements) {
    const int stride = blockDim.x * gridDim.x;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int blockIdx = tid; blockIdx < numElements / 32; blockIdx += stride) {
        const int64_t srcOffset = blockIdx * 34;
        const int64_t dstOffset = blockIdx * 32;

        // Read scale (FP16)
        unsigned short scale16 = *(unsigned short*)&src[srcOffset];
        float d = fp16_to_fp32(scale16);

        // Process 32 elements
        for (int j = 0; j < 32; j++) {
            float val = d * (float)(int8_t)src[srcOffset + 2 + j];
            dst[dstOffset + j] = val;
        }
    }
}

// Q8_0 Dequantization to BF16
__global__ void dequant_q8_0_to_bf16_kernel(const unsigned char* __restrict__ src,
                                              unsigned short* __restrict__ dst,
                                              int numElements) {
    const int stride = blockDim.x * gridDim.x;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int blockIdx = tid; blockIdx < numElements / 32; blockIdx += stride) {
        const int64_t srcOffset = blockIdx * 34;
        const int64_t dstOffset = blockIdx * 32;

        unsigned short scale16 = *(unsigned short*)&src[srcOffset];
        float d = fp16_to_fp32(scale16);

        for (int j = 0; j < 32; j++) {
            float val = d * (float)(int8_t)src[srcOffset + 2 + j];
            dst[dstOffset + j] = fp32_to_bf16(val);
        }
    }
}

// Q4_K Dequantization: 256 elements per block, 144 bytes
// Layout: d(2) + dmin(2) + scales(12) + qs(128)
__global__ void dequant_q4_k_kernel(const unsigned char* __restrict__ src,
                                     float* __restrict__ dst,
                                     int numElements) {
    const int stride = blockDim.x * gridDim.x;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int blockIdx = tid; blockIdx < numElements / 256; blockIdx += stride) {
        const int64_t srcOffset = blockIdx * 144;
        const int64_t dstOffset = blockIdx * 256;

        // Read super scales
        float d = fp16_to_fp32(*(unsigned short*)&src[srcOffset]);
        float dmin = fp16_to_fp32(*(unsigned short*)&src[srcOffset + 2]);

        // Unpack scales: 12 bytes -> 8 uint8 scales + 8 uint8 mins
        unsigned char scales[8];
        unsigned char mins[8];
        const unsigned char* scalesPtr = &src[srcOffset + 4];

        for (int j = 0; j < 4; j++) {
            scales[j] = scalesPtr[j] & 63;
            mins[j] = scalesPtr[j + 4] & 63;
        }
        for (int j = 4; j < 8; j++) {
            scales[j] = (scalesPtr[j + 4] & 0xF) | ((scalesPtr[j - 4] >> 6) << 4);
            mins[j] = (scalesPtr[j + 4] >> 4) | ((scalesPtr[j] >> 6) << 4);
        }

        // Compute D and M arrays
        float D[8], M[8];
        for (int j = 0; j < 8; j++) {
            D[j] = d * (float)scales[j];
            M[j] = dmin * (float)mins[j];
        }

        // Decode 256 weights: 8 sub-blocks of 32
        const unsigned char* qs = &src[srcOffset + 16];
        for (int j = 0; j < 8; j++) {
            float dj = D[j];
            float mj = M[j];
            int qsOffset = j * 16;
            int idxBase = dstOffset + j * 32;

            for (int k = 0; k < 16; k++) {
                unsigned char b = qs[qsOffset + k];
                dst[idxBase + k] = dj * (float)(b & 0xF) - mj;
                dst[idxBase + k + 16] = dj * (float)(b >> 4) - mj;
            }
        }
    }
}

// Q4_K Dequantization to BF16
__global__ void dequant_q4_k_to_bf16_kernel(const unsigned char* __restrict__ src,
                                              unsigned short* __restrict__ dst,
                                              int numElements) {
    const int stride = blockDim.x * gridDim.x;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int blockIdx = tid; blockIdx < numElements / 256; blockIdx += stride) {
        const int64_t srcOffset = blockIdx * 144;
        const int64_t dstOffset = blockIdx * 256;

        float d = fp16_to_fp32(*(unsigned short*)&src[srcOffset]);
        float dmin = fp16_to_fp32(*(unsigned short*)&src[srcOffset + 2]);

        unsigned char scales[8];
        unsigned char mins[8];
        const unsigned char* scalesPtr = &src[srcOffset + 4];

        for (int j = 0; j < 4; j++) {
            scales[j] = scalesPtr[j] & 63;
            mins[j] = scalesPtr[j + 4] & 63;
        }
        for (int j = 4; j < 8; j++) {
            scales[j] = (scalesPtr[j + 4] & 0xF) | ((scalesPtr[j - 4] >> 6) << 4);
            mins[j] = (scalesPtr[j + 4] >> 4) | ((scalesPtr[j] >> 6) << 4);
        }

        float D[8], M[8];
        for (int j = 0; j < 8; j++) {
            D[j] = d * (float)scales[j];
            M[j] = dmin * (float)mins[j];
        }

        const unsigned char* qs = &src[srcOffset + 16];
        for (int j = 0; j < 8; j++) {
            float dj = D[j];
            float mj = M[j];
            int qsOffset = j * 16;
            int idxBase = dstOffset + j * 32;

            for (int k = 0; k < 16; k++) {
                unsigned char b = qs[qsOffset + k];
                dst[idxBase + k] = fp32_to_bf16(dj * (float)(b & 0xF) - mj);
                dst[idxBase + k + 16] = fp32_to_bf16(dj * (float)(b >> 4) - mj);
            }
        }
    }
}

// Q6_K Dequantization: 256 elements per block, 210 bytes
// Layout: qs(128) + qh(64) + scales(16) + d(2)
__global__ void dequant_q6_k_kernel(const unsigned char* __restrict__ src,
                                     float* __restrict__ dst,
                                     int numElements) {
    const int stride = blockDim.x * gridDim.x;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int blockIdx = tid; blockIdx < numElements / 256; blockIdx += stride) {
        const int64_t srcOffset = blockIdx * 210;
        const int64_t dstOffset = blockIdx * 256;

        const unsigned char* qs = &src[srcOffset];
        const unsigned char* qh = &src[srcOffset + 128];
        const unsigned char* scales = &src[srcOffset + 192];
        float d = fp16_to_fp32(*(unsigned short*)&src[srcOffset + 208]);

        // Process 16 sub-blocks of 16 elements
        for (int l = 0; l < 16; l++) {
            float s = d * (float)(int8_t)scales[l];

            for (int k = 0; k < 16; k++) {
                int idx = l * 16 + k;
                int dstIdx = dstOffset + idx;

                // Low 4 bits from qs
                unsigned char q4 = qs[idx / 2];
                if (idx % 2 == 0) {
                    q4 &= 0x0F;
                } else {
                    q4 >>= 4;
                }

                // High 2 bits from qh
                unsigned char q2 = (qh[idx / 4] >> ((idx % 4) * 2)) & 0x03;

                // Combined 6 bits as signed
                int8_t q = (int8_t)((q2 << 4) | q4);

                dst[dstIdx] = s * ((float)q - 32.0f);
            }
        }
    }
}

// Q6_K Dequantization to BF16
__global__ void dequant_q6_k_to_bf16_kernel(const unsigned char* __restrict__ src,
                                              unsigned short* __restrict__ dst,
                                              int numElements) {
    const int stride = blockDim.x * gridDim.x;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int blockIdx = tid; blockIdx < numElements / 256; blockIdx += stride) {
        const int64_t srcOffset = blockIdx * 210;
        const int64_t dstOffset = blockIdx * 256;

        const unsigned char* qs = &src[srcOffset];
        const unsigned char* qh = &src[srcOffset + 128];
        const unsigned char* scales = &src[srcOffset + 192];
        float d = fp16_to_fp32(*(unsigned short*)&src[srcOffset + 208]);

        for (int l = 0; l < 16; l++) {
            float s = d * (float)(int8_t)scales[l];

            for (int k = 0; k < 16; k++) {
                int idx = l * 16 + k;
                int dstIdx = dstOffset + idx;

                unsigned char q4 = qs[idx / 2];
                if (idx % 2 == 0) {
                    q4 &= 0x0F;
                } else {
                    q4 >>= 4;
                }

                unsigned char q2 = (qh[idx / 4] >> ((idx % 4) * 2)) & 0x03;
                int8_t q = (int8_t)((q2 << 4) | q4);

                dst[dstIdx] = fp32_to_bf16(s * ((float)q - 32.0f));
            }
        }
    }
}

// =============================================================================
// C Export Functions
// =============================================================================

extern "C" {
    void cudaRMSNormF16(cudaStream_t stream, void* input, void* weight, void* output, int rows, int cols, float eps) {
        rmsnorm_kernel<<<256, 256, 0, stream>>>((float*)input, (float*)weight, (float*)output, rows, cols, eps);
    }
    
    void cudaSwiGLUF16(cudaStream_t stream, void* gate, void* up, void* output, int size) {
        swiglu_kernel<<<256, 256, 0, stream>>>((float*)gate, (float*)up, (float*)output, size);
    }
    
    void cudaSoftmaxF16(cudaStream_t stream, void* input, void* output, int rows, int cols) {
        softmax_kernel<<<rows, 256, 0, stream>>>((float*)input, (float*)output, rows, cols);
    }
    
    void cudaRoPEF16(cudaStream_t stream, void* tensor, int pos, int heads, int headDim, float theta) {
        int size = heads * (headDim / 2);
        rope_kernel<<<256, 256, 0, stream>>>((float*)tensor, pos, heads, headDim, theta);
    }
    
    void cudaAddF16(cudaStream_t stream, void* a, void* b, void* out, int size) {
        add_kernel<<<256, 256, 0, stream>>>((float*)a, (float*)b, (float*)out, size);
    }
    
    void cudaSiLUF16(cudaStream_t stream, void* input, void* output, int size) {
        silu_kernel<<<256, 256, 0, stream>>>((float*)input, (float*)output, size);
    }
    
    // MOE Exports
    void cudaMOERouterLogits(cudaStream_t stream, void* input, void* gate, void* output,
                            int batch, int dim, int num_experts) {
        dim3 grid(batch);
        dim3 block(min(num_experts, 256));
        moe_router_logits_kernel<<<grid, block, 0, stream>>>(
            (float*)input, (float*)gate, (float*)output, batch, dim, num_experts);
    }
    
    void cudaMOETopKSelection(cudaStream_t stream, void* logits, int top_k,
                             void* indices, void* weights, int batch, int num_experts) {
        dim3 grid(batch);
        moe_topk_kernel<<<grid, 1, 0, stream>>>(
            (float*)logits, (int*)indices, (float*)weights, batch, num_experts, top_k);
    }
    
    void cudaMOEExpertForward(cudaStream_t stream, void* input, void* expert_weights,
                             void* indices, void* expert_weights_w, void* output,
                             int batch, int dim, int hidden_dim, int num_experts, int top_k) {
        int threads = 256;
        int blocks = (batch * dim + threads - 1) / threads;
        moe_expert_forward_kernel<<<blocks, threads, 0, stream>>>(
            (float*)input, (float*)expert_weights, (int*)indices,
            (float*)expert_weights_w, (float*)output,
            batch, dim, hidden_dim, num_experts, top_k);
    }
    
    void cudaMOEExpertGateUpSwiGLU(cudaStream_t stream, void* input,
                                   void* gate_experts, void* up_experts,
                                   void* indices, void* weights, void* output,
                                   int batch, int dim, int hidden_dim,
                                   int num_experts, int top_k) {
        dim3 grid(batch);
        dim3 block(hidden_dim);
        moe_gate_up_swiglu_kernel<<<grid, block, 0, stream>>>(
            (float*)input, (float*)gate_experts, (float*)up_experts,
            (int*)indices, (float*)weights, (float*)output,
            batch, dim, hidden_dim, num_experts, top_k);
    }

    // Dequantization Exports
    void cudaDequantQ8_0(cudaStream_t stream, void* src, void* dst, int numElements) {
        int blocks = (numElements / 32 + 255) / 256;
        dequant_q8_0_kernel<<<blocks, 256, 0, stream>>>(
            (const unsigned char*)src, (float*)dst, numElements);
    }

    void cudaDequantQ8_0ToBF16(cudaStream_t stream, void* src, void* dst, int numElements) {
        int blocks = (numElements / 32 + 255) / 256;
        dequant_q8_0_to_bf16_kernel<<<blocks, 256, 0, stream>>>(
            (const unsigned char*)src, (unsigned short*)dst, numElements);
    }

    void cudaDequantQ4_K(cudaStream_t stream, void* src, void* dst, int numElements) {
        int blocks = (numElements / 256 + 255) / 256;
        dequant_q4_k_kernel<<<blocks, 256, 0, stream>>>(
            (const unsigned char*)src, (float*)dst, numElements);
    }

    void cudaDequantQ4_KToBF16(cudaStream_t stream, void* src, void* dst, int numElements) {
        int blocks = (numElements / 256 + 255) / 256;
        dequant_q4_k_to_bf16_kernel<<<blocks, 256, 0, stream>>>(
            (const unsigned char*)src, (unsigned short*)dst, numElements);
    }

    void cudaDequantQ6_K(cudaStream_t stream, void* src, void* dst, int numElements) {
        int blocks = (numElements / 256 + 255) / 256;
        dequant_q6_k_kernel<<<blocks, 256, 0, stream>>>(
            (const unsigned char*)src, (float*)dst, numElements);
    }

    void cudaDequantQ6_KToBF16(cudaStream_t stream, void* src, void* dst, int numElements) {
        int blocks = (numElements / 256 + 255) / 256;
        dequant_q6_k_to_bf16_kernel<<<blocks, 256, 0, stream>>>(
            (const unsigned char*)src, (unsigned short*)dst, numElements);
    }
}

// =============================================================================
// Fused Attention Kernel (Q @ K^T, softmax, @ V in one pass)
// =============================================================================

__global__ void fused_attention_kernel(
    const float* __restrict__ q,      // [batch, heads, seqLen, headDim]
    const float* __restrict__ k,      // [batch, heads, seqLen, headDim]
    const float* __restrict__ v,      // [batch, heads, seqLen, headDim]
    float* __restrict__ output,        // [batch, heads, seqLen, headDim]
    const float* __restrict__ kCache,  // [batch, heads, seqLen, headDim] or nullptr
    const float* __restrict__ vCache,  // [batch, heads, seqLen, headDim] or nullptr
    int batch, int heads, int seqLen, int kvSeqLen, int headDim,
    float scale, int useCache) {

    int bh = blockIdx.x;
    int head = bh % heads;
    int batchIdx = bh / heads;
    int pos = threadIdx.y;  // Position in sequence

    int tid = threadIdx.x;

    extern __shared__ float shared[];
    float* s_q = shared;
    float* s_scores = shared + headDim;

    if (pos >= seqLen) return;

    // Load Q for this position
    int qOffset = ((batchIdx * heads + head) * seqLen + pos) * headDim;
    for (int d = tid; d < headDim; d += blockDim.x) {
        s_q[d] = q[qOffset + d];
    }
    __syncthreads();

    // Compute attention scores against K (with optional caching)
    int totalKV = kvSeqLen;
    float maxScore = -INFINITY;

    for (int t = 0; t < totalKV; t++) {
        float score = 0.0f;
        int kOffset;
        if (useCache && kCache) {
            kOffset = ((batchIdx * heads + head) * kvSeqLen + t) * headDim;
        } else {
            kOffset = ((batchIdx * heads + head) * seqLen + t) * headDim;
        }

        for (int d = tid; d < headDim; d += blockDim.x) {
            score += s_q[d] * k[kOffset + d];
        }
        score *= scale;

        if (tid == 0) {
            if (score > maxScore) maxScore = score;
        }
        __syncthreads();
    }
    __syncthreads();

    // Second pass: softmax and weighted sum with V
    float sum = 0.0f;
    for (int t = 0; t < totalKV; t++) {
        float score = 0.0f;
        int kOffset, vOffset;
        if (useCache && kCache) {
            kOffset = ((batchIdx * heads + head) * kvSeqLen + t) * headDim;
            vOffset = ((batchIdx * heads + head) * kvSeqLen + t) * headDim;
        } else {
            kOffset = ((batchIdx * heads + head) * seqLen + t) * headDim;
            vOffset = ((batchIdx * heads + head) * seqLen + t) * headDim;
        }

        for (int d = tid; d < headDim; d += blockDim.x) {
            score += s_q[d] * k[kOffset + d];
        }
        score *= scale;
        float attn = expf(score - maxScore);
        sum += attn;

        for (int d = tid; d < headDim; d += blockDim.x) {
            s_scores[d] = attn * v[vOffset + d];
        }
        __syncthreads();
    }

    // Reduce sum
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Output: normalized weighted sum
    float invSum = 1.0f / (sum + 1e-8f);
    for (int d = tid; d < headDim; d += blockDim.x) {
        float val = 0.0f;
        for (int t = 0; t < totalKV; t++) {
            int vOffset;
            if (useCache && vCache) {
                vOffset = ((batchIdx * heads + head) * kvSeqLen + t) * headDim;
            } else {
                vOffset = ((batchIdx * heads + head) * seqLen + t) * headDim;
            }
            float score = 0.0f;
            for (int dd = 0; dd < headDim; dd++) {
                score += s_q[dd] * k[vOffset + dd];  // Use k as temp
            }
            score *= scale;
            float attnVal = expf(score - maxScore);
            val += attnVal * v[vOffset + d];
        }
        output[qOffset + d] = val * invSum;
    }
}

// =============================================================================
// Fused RoPE Kernel (apply rotary positional encoding)
// =============================================================================

__global__ void fused_rope_kernel(
    float* __restrict__ tensor,        // [batch, heads, seqLen, headDim]
    const float* __restrict__ posIds,  // Position IDs [seqLen]
    int batch, int heads, int seqLen, int headDim,
    float theta) {

    int idx = (blockIdx.x * blockDim.x + threadIdx.x);
    int total = batch * heads * seqLen * (headDim / 2);

    if (idx >= total) return;

    int tmp = idx;
    int headDimHalf = headDim / 2;
    int pos = tmp % seqLen;
    tmp /= seqLen;
    int head = tmp % heads;
    tmp /= heads;
    int batchIdx = tmp;

    float pos_f = (float)posIds[pos];
    float invFreq = 1.0f / powf(theta, (float)(idx % headDimHalf) / headDimHalf);

    int offset = ((batchIdx * heads + head) * seqLen + pos) * headDim;
    int dim = idx % headDimHalf;

    float freq = pos_f * invFreq;
    float cosVal = cosf(freq);
    float sinVal = sinf(freq);

    float x0 = tensor[offset + dim];
    float x1 = tensor[offset + dim + headDimHalf];

    tensor[offset + dim] = x0 * cosVal - x1 * sinVal;
    tensor[offset + dim + headDimHalf] = x0 * sinVal + x1 * cosVal;
}

// =============================================================================
// Fused SwiGLU Kernel (gate @ up -> silu(gate * up) -> down)
// =============================================================================

__global__ void fused_swiglu_kernel(
    const float* __restrict__ input,     // [batch, dim]
    const float* __restrict__ gateWeight, // [hiddenDim, dim]
    const float* __restrict__ upWeight,   // [hiddenDim, dim]
    const float* __restrict__ downWeight, // [dim, hiddenDim]
    float* __restrict__ output,          // [batch, dim]
    int batch, int dim, int hiddenDim) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * dim;

    if (idx >= total) return;

    int b = idx / dim;
    int d = idx % dim;

    // Compute gate and up projections
    float gate = 0.0f, up = 0.0f;
    for (int h = 0; h < hiddenDim; h++) {
        gate += input[b * dim + d] * gateWeight[h * dim + d];
        up += input[b * dim + d] * upWeight[h * dim + d];
    }

    // SiLU activation
    float swiglu = gate / (1.0f + expf(-gate)) * up;

    // Down projection
    float result = 0.0f;
    for (int h = 0; h < hiddenDim; h++) {
        result += swiglu * downWeight[d * hiddenDim + h];
    }

    output[idx] = result;
}

// =============================================================================
// Fused MLP Kernel (single pass through FFN)
// =============================================================================

__global__ void fused_mlp_kernel(
    const float* __restrict__ input,      // [batch, dim]
    const float* __restrict__ gateWeight, // [hiddenDim, dim]
    const float* __restrict__ upWeight,   // [hiddenDim, dim]
    const float* __restrict__ downWeight, // [dim, hiddenDim]
    float* __restrict__ output,          // [batch, dim]
    int batch, int dim, int hiddenDim) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * dim;

    if (idx >= total) return;

    int b = idx / dim;
    int d = idx % dim;

    // Shared memory for intermediate values
    extern __shared__ float shared[];
    float* gate = shared;
    float* up = shared + hiddenDim;

    // Compute gate and up in parallel
    for (int h = 0; h < hiddenDim; h++) {
        float val = input[b * dim + d];
        gate[h] = val * gateWeight[h * dim + d];
        up[h] = val * upWeight[h * dim + d];
    }
    __syncthreads();

    // SiLU + multiply
    for (int h = 0; h < hiddenDim; h++) {
        gate[h] = gate[h] / (1.0f + expf(-gate[h])) * up[h];
    }
    __syncthreads();

    // Down projection
    float result = 0.0f;
    for (int h = 0; h < hiddenDim; h++) {
        result += gate[h] * downWeight[d * hiddenDim + h];
    }

    output[idx] = result;
}

// =============================================================================
// Fused RMSNorm + Add Residual Kernel
// =============================================================================

__global__ void fused_rmsnorm_add_kernel(
    const float* __restrict__ input,     // [batch, dim] - residual
    const float* __restrict__ hidden,     // [batch, dim] - layer output
    const float* __restrict__ weight,    // [dim] - RMSNorm weight
    float* __restrict__ output,          // [batch, dim]
    int batch, int dim, float eps) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * dim;

    if (idx >= total) return;

    int b = idx / dim;
    int d = idx % dim;

    // Compute RMS for this row
    float sumSq = 0.0f;
    for (int i = 0; i < dim; i++) {
        float val = hidden[b * dim + i] + input[b * dim + i];
        sumSq += val * val;
    }
    sumSq = sqrtf(sumSq / dim + eps);
    sumSq = 1.0f / sumSq;

    // Normalize and apply weight
    float val = (hidden[b * dim + d] + input[b * dim + d]) * sumSq * weight[d];
    output[idx] = val;
}

// =============================================================================
// Flash Attention-style Fused Kernel (memory-efficient for long sequences)
// =============================================================================

__device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void flash_fused_attention_kernel(
    const float* __restrict__ q,      // [batch, heads, seqLen, headDim]
    const float* __restrict__ k,       // [batch, heads, kvSeqLen, headDim]
    const float* __restrict__ v,       // [batch, heads, kvSeqLen, headDim]
    float* __restrict__ output,        // [batch, heads, seqLen, headDim]
    int batch, int heads, int seqLen, int kvSeqLen, int headDim,
    float scale) {

    int bh = blockIdx.x;
    int head = bh % heads;
    int batchIdx = bh / heads;

    int pos = blockIdx.y;  // Sequence position
    int tid = threadIdx.x;
    int warpId = tid / 32;
    int laneId = tid % 32;

    if (pos >= seqLen) return;

    extern __shared__ float shared[];
    float* s_q = shared;
    float* s_max = shared + headDim;
    float* s_sum = shared + headDim + 32;
    float* s_out = shared + headDim + 64;

    // Load Q
    int qOffset = ((batchIdx * heads + head) * seqLen + pos) * headDim;
    for (int d = tid; d < headDim; d += blockDim.x) {
        s_q[d] = q[qOffset + d];
    }

    // Initialize
    if (warpId < 1) s_max[warpId] = -INFINITY;
    if (warpId < 1) s_sum[warpId] = 0.0f;

    // First pass: compute max score
    float threadMax = -INFINITY;
    for (int t = 0; t < kvSeqLen; t += 32) {
        int kOffset = ((batchIdx * heads + head) * kvSeqLen + t) * headDim;

        float score = 0.0f;
        for (int d = tid; d < headDim; d += 32) {
            score += s_q[d] * k[kOffset + d];
        }
        score *= scale;

        if (t + laneId < kvSeqLen && score > threadMax) {
            threadMax = score;
        }
    }
    threadMax = warp_reduce_max(threadMax);
    if (warpId == 0) {
        s_max[laneId] = threadMax;
    }

    // Second pass: softmax and accumulate
    __syncthreads();
    float rowMax = -INFINITY;
    for (int w = 0; w < 1; w++) {
        rowMax = fmaxf(rowMax, s_max[w]);
    }

    float threadSum = 0.0f;
    for (int t = 0; t < kvSeqLen; t++) {
        int kOffset = ((batchIdx * heads + head) * kvSeqLen + t) * headDim;
        int vOffset = ((batchIdx * heads + head) * kvSeqLen + t) * headDim;

        float score = 0.0f;
        for (int d = tid; d < headDim; d += 32) {
            score += s_q[d] * k[kOffset + d];
        }
        score *= scale;
        float attn = expf(score - rowMax);
        threadSum += attn;

        for (int d = tid; d < headDim; d += 32) {
            s_out[d] += attn * v[vOffset + d];
        }
    }
    threadSum = warp_reduce_sum(threadSum);

    // Finalize
    float invSum = 1.0f / (threadSum + 1e-8f);
    for (int d = tid; d < headDim; d += blockDim.x) {
        output[qOffset + d] = s_out[d] * invSum;
    }
}

// =============================================================================
// Fused QKV + RoPE Kernel
// =============================================================================

__global__ void fused_qkv_rope_kernel(
    float* input, float* qWeight, float* kWeight, float* vWeight,
    float* qOut, float* kOut, float* vOut,
    int batch, int dim, int qDim, int kvDim,
    float* ropeFreqCos, float* ropeFreqSin,
    int headDim) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * dim;
    
    for (int i = idx; i < totalElements; i += gridDim.x * blockDim.x) {
        int batchIdx = i / dim;
        int dimIdx = i % dim;
        
        float val = input[i];
        
        if (dimIdx < qDim) {
            qOut[i] = val;
        }
        if (dimIdx < kvDim) {
            kOut[batchIdx * kvDim + dimIdx] = val;
        }
        if (dimIdx < kvDim) {
            vOut[batchIdx * kvDim + dimIdx] = val;
        }
    }
}

// =============================================================================
// Fused Kernel C Export Functions
// =============================================================================

extern "C" {
    void cudaFusedAttention(
        cudaStream_t stream,
        const void* q, const void* k, const void* v,
        void* output,
        const void* kCache, const void* vCache,
        int batch, int heads, int seqLen, int kvSeqLen, int headDim,
        float scale, int useCache) {

        dim3 grid(batch * heads * seqLen);
        dim3 block(256, 1);
        size_t sharedSize = (headDim + 256) * sizeof(float);

        fused_attention_kernel<<<grid, block, sharedSize, stream>>>(
            (const float*)q, (const float*)k, (const float*)v,
            (float*)output, (const float*)kCache, (const float*)vCache,
            batch, heads, seqLen, kvSeqLen, headDim, scale, useCache);
    }

    void cudaFlashFusedAttention(
        cudaStream_t stream,
        const void* q, const void* k, const void* v,
        void* output,
        int batch, int heads, int seqLen, int kvSeqLen, int headDim,
        float scale) {

        dim3 grid(batch * heads, seqLen);
        dim3 block(128);
        size_t sharedSize = (headDim + 128) * sizeof(float);

        flash_fused_attention_kernel<<<grid, block, sharedSize, stream>>>(
            (const float*)q, (const float*)k, (const float*)v,
            (float*)output,
            batch, heads, seqLen, kvSeqLen, headDim, scale);
    }

    void cudaFusedRoPE(
        cudaStream_t stream,
        void* tensor,
        const int* posIds,
        int batch, int heads, int seqLen, int headDim,
        float theta) {

        int total = batch * heads * seqLen * (headDim / 2);
        int blockSize = 256;
        int gridSize = (total + blockSize - 1) / blockSize;

        fused_rope_kernel<<<gridSize, blockSize, 0, stream>>>(
            (float*)tensor, (const float*)posIds,
            batch, heads, seqLen, headDim, theta);
    }

    void cudaFusedSwiGLU(
        cudaStream_t stream,
        const void* input,
        const void* gateWeight,
        const void* upWeight,
        const void* downWeight,
        void* output,
        int batch, int dim, int hiddenDim) {

        int total = batch * dim;
        int blockSize = 256;
        int gridSize = (total + blockSize - 1) / blockSize;

        fused_swiglu_kernel<<<gridSize, blockSize, 0, stream>>>(
            (const float*)input,
            (const float*)gateWeight,
            (const float*)upWeight,
            (const float*)downWeight,
            (float*)output,
            batch, dim, hiddenDim);
    }

    void cudaFusedMLP(
        cudaStream_t stream,
        const void* input,
        const void* gateWeight,
        const void* upWeight,
        const void* downWeight,
        void* output,
        int batch, int dim, int hiddenDim) {

        int total = batch * dim;
        int blockSize = 256;
        int gridSize = (total + blockSize - 1) / blockSize;
        size_t sharedSize = hiddenDim * 2 * sizeof(float);

        fused_mlp_kernel<<<gridSize, blockSize, sharedSize, stream>>>(
            (const float*)input,
            (const float*)gateWeight,
            (const float*)upWeight,
            (const float*)downWeight,
            (float*)output,
            batch, dim, hiddenDim);
    }

    void cudaFusedRMSNormAdd(
        cudaStream_t stream,
        const void* input,
        const void* hidden,
        const void* weight,
        void* output,
        int batch, int dim,
        float eps) {

        int total = batch * dim;
        int blockSize = 256;
        int gridSize = (total + blockSize - 1) / blockSize;

        fused_rmsnorm_add_kernel<<<gridSize, blockSize, 0, stream>>>(
            (const float*)input,
            (const float*)hidden,
            (const float*)weight,
            (float*)output,
            batch, dim, eps);
    }

    void cudaFusedQKVRope(
        cudaStream_t stream,
        const void* input,
        const void* qWeight, const void* kWeight, const void* vWeight,
        void* qOut, void* kOut, void* vOut,
        const void* ropeCos, const void* ropeSin,
        int batch, int dim, int qDim, int kvDim, int headDim,
        int seqLen) {

        int total = batch * dim * seqLen;
        int blockSize = 256;
        int gridSize = (total + blockSize - 1) / blockSize;

        // TODO: Add fused QKV + RoPE kernel with precomputed frequencies
        // For now, use the simple QKV split kernel
        fused_qkv_rope_kernel<<<gridSize, blockSize, 0, stream>>>(
            (float*)input, (float*)qWeight, (float*)kWeight, (float*)vWeight,
            (float*)qOut, (float*)kOut, (float*)vOut,
            batch, dim, qDim, kvDim, (float*)ropeCos, (float*)ropeSin, headDim);
    }
}