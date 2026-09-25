#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

// =============================================================================
// Helper Functions for MLA
// =============================================================================

#ifndef CUDA_MLA_HELPERS
#define CUDA_MLA_HELPERS

__device__ __forceinline__ float mla_to_float(float x) { return x; }
__device__ __forceinline__ float mla_to_float(__half x) { return __half2float(x); }

template <typename T>
__device__ __forceinline__ T mla_from_float(float x);

template <>
__device__ __forceinline__ float mla_from_float<float>(float x) { return x; }

template <>
__device__ __forceinline__ __half mla_from_float<__half>(float x) { return __float2half(x); }

__device__ __forceinline__ float mla_warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

#endif // CUDA_MLA_HELPERS

// =============================================================================
// DeepSeek MLA (Multi-Head Latent Attention) Kernels
// =============================================================================

// 1. Decompress KV Cache:
// compressed_kv: [num_tokens, kv_lora_rank]
// w_ukv:         [heads * (qk_nope_dim + v_head_dim), kv_lora_rank]
// k_nope:        [num_tokens, heads * qk_nope_dim]
// v:             [num_tokens, heads * v_head_dim]
template <typename InType, typename WType, typename OutType>
__global__ void mla_decompress_kv_kernel(
    const InType* __restrict__ compressed_kv,
    const WType* __restrict__ w_ukv,
    OutType* __restrict__ k_nope,
    OutType* __restrict__ v,
    int num_tokens, int kv_lora_rank,
    int heads, int qk_nope_dim, int v_head_dim) {

    int total_k_rows = heads * qk_nope_dim;
    int total_v_rows = heads * v_head_dim;
    int total_rows = total_k_rows + total_v_rows;
    int total_work = num_tokens * total_rows;

    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int totalWarps = (gridDim.x * blockDim.x) / 32;

    for (int work = warpId; work < total_work; work += totalWarps) {
        int t = work / total_rows;
        int r = work % total_rows;

        const InType* kv_ptr = compressed_kv + (size_t)t * kv_lora_rank;
        const WType* w_ptr = w_ukv + (size_t)r * kv_lora_rank;

        float sum = 0.0f;
        for (int j = lane; j < kv_lora_rank; j += 32) {
            sum += mla_to_float(kv_ptr[j]) * mla_to_float(w_ptr[j]);
        }
        sum = mla_warp_reduce_sum(sum);

        if (lane == 0) {
            if (r < total_k_rows) {
                k_nope[(size_t)t * total_k_rows + r] = mla_from_float<OutType>(sum);
            } else {
                v[(size_t)t * total_v_rows + (r - total_k_rows)] = mla_from_float<OutType>(sum);
            }
        }
    }
}

// 2. Query Projection Split & Decoupled RoPE:
// q_all:   [num_tokens, heads * (qk_nope_dim + qk_rope_dim)]
// pos_ids: [num_tokens] (optional, NULL defaults to 0..num_tokens-1)
// q_nope:  [num_tokens, heads * qk_nope_dim]
// q_rope:  [num_tokens, heads * qk_rope_dim]
template <typename InType, typename OutType>
__global__ void mla_project_query_split_rope_kernel(
    const InType* __restrict__ q_all,
    const int* __restrict__ pos_ids,
    OutType* __restrict__ q_nope,
    OutType* __restrict__ q_rope,
    int num_tokens, int heads,
    int qk_nope_dim, int qk_rope_dim, float theta) {

    int total_head_tokens = num_tokens * heads;
    int in_head_dim = qk_nope_dim + qk_rope_dim;
    int half_rope = qk_rope_dim / 2;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int ht = idx; ht < total_head_tokens; ht += gridDim.x * blockDim.x) {
        int t = ht / heads;
        int pos = pos_ids ? pos_ids[t] : t;

        const InType* in_ptr = q_all + (size_t)ht * in_head_dim;
        OutType* nope_ptr = q_nope + (size_t)ht * qk_nope_dim;
        OutType* rope_ptr = q_rope + (size_t)ht * qk_rope_dim;

        // Copy non-rotary content query
        for (int i = 0; i < qk_nope_dim; i++) {
            nope_ptr[i] = mla_from_float<OutType>(mla_to_float(in_ptr[i]));
        }

        // Apply RoPE on decoupled rotary query
        const InType* rope_in = in_ptr + qk_nope_dim;
        for (int i = 0; i < half_rope; i++) {
            float freq = (float)pos * powf(theta, -2.0f * (float)i / (float)qk_rope_dim);
            float cosVal = cosf(freq);
            float sinVal = sinf(freq);

            float x0 = mla_to_float(rope_in[i]);
            float x1 = mla_to_float(rope_in[i + half_rope]);

            rope_ptr[i] = mla_from_float<OutType>(x0 * cosVal - x1 * sinVal);
            rope_ptr[i + half_rope] = mla_from_float<OutType>(x0 * sinVal + x1 * cosVal);
        }
    }
}

// 3. Absorbed Query Projection:
// Projects q_nope [num_tokens, heads, qk_nope_dim] into latent space:
// q_absorbed[t, h, j] = sum_{k=0}^{qk_nope_dim-1} q_nope[t, h, k] * w_uk[h * qk_nope_dim + k, j]
// w_uk:       [heads * qk_nope_dim, kv_lora_rank]
// q_absorbed: [num_tokens, heads, kv_lora_rank]
template <typename InType, typename WType, typename OutType>
__global__ void mla_absorbed_query_kernel(
    const InType* __restrict__ q_nope,
    const WType* __restrict__ w_uk,
    OutType* __restrict__ q_absorbed,
    int num_tokens, int heads,
    int qk_nope_dim, int kv_lora_rank) {

    int total_work = num_tokens * heads * kv_lora_rank;
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int totalWarps = (gridDim.x * blockDim.x) / 32;

    for (int work = warpId; work < total_work; work += totalWarps) {
        int th = work / kv_lora_rank;
        int j = work % kv_lora_rank;
        int h = th % heads;

        const InType* q_ptr = q_nope + (size_t)th * qk_nope_dim;
        const WType* w_ptr = w_uk + (size_t)(h * qk_nope_dim) * kv_lora_rank;

        float sum = 0.0f;
        for (int k = lane; k < qk_nope_dim; k += 32) {
            sum += mla_to_float(q_ptr[k]) * mla_to_float(w_ptr[(size_t)k * kv_lora_rank + j]);
        }
        sum = mla_warp_reduce_sum(sum);

        if (lane == 0) {
            q_absorbed[(size_t)th * kv_lora_rank + j] = mla_from_float<OutType>(sum);
        }
    }
}

// 4. Absorbed Decode Attention Kernel:
// Directly computes attention against compressed KV cache in decode mode:
// q_absorbed:   [num_tokens, heads, kv_lora_rank]
// q_rope:       [num_tokens, heads, qk_rope_dim]
// k_cache:      [seq_len, kv_lora_rank]
// k_rope_cache: [seq_len, qk_rope_dim]
// w_uv:         [heads * v_head_dim, kv_lora_rank]
// output:       [num_tokens, heads * v_head_dim]
template <typename InType, typename WType, typename OutType>
__global__ void mla_absorbed_decode_attention_kernel(
    const InType* __restrict__ q_absorbed,
    const InType* __restrict__ q_rope,
    const InType* __restrict__ k_cache,
    const InType* __restrict__ k_rope_cache,
    const WType* __restrict__ w_uv,
    OutType* __restrict__ output,
    int num_tokens, int seq_len, int heads,
    int kv_lora_rank, int qk_rope_dim, int v_head_dim, float scale) {

    extern __shared__ float s_mem[];
    float* s_scores = s_mem;
    float* s_latent = s_mem + seq_len;

    int h = blockIdx.x;
    int t = blockIdx.y;
    int tid = threadIdx.x;

    if (h >= heads || t >= num_tokens) return;

    const InType* cur_q_abs = q_absorbed + (size_t)(t * heads + h) * kv_lora_rank;
    const InType* cur_q_rope = q_rope + (size_t)(t * heads + h) * qk_rope_dim;

    // Step 1: Compute attention scores for each token s in [0, seq_len)
    for (int s = tid; s < seq_len; s += blockDim.x) {
        const InType* cur_k_cache = k_cache + (size_t)s * kv_lora_rank;
        const InType* cur_k_rope = k_rope_cache + (size_t)s * qk_rope_dim;

        float dot_c = 0.0f;
        for (int j = 0; j < kv_lora_rank; j++) {
            dot_c += mla_to_float(cur_q_abs[j]) * mla_to_float(cur_k_cache[j]);
        }
        float dot_r = 0.0f;
        for (int j = 0; j < qk_rope_dim; j++) {
            dot_r += mla_to_float(cur_q_rope[j]) * mla_to_float(cur_k_rope[j]);
        }
        s_scores[s] = (dot_c + dot_r) * scale;
    }
    __syncthreads();

    // Step 2: Softmax over scores
    float local_max = -INFINITY;
    for (int s = tid; s < seq_len; s += blockDim.x) {
        if (s_scores[s] > local_max) local_max = s_scores[s];
    }
    local_max = mla_warp_reduce_sum(local_max);

    __shared__ float s_max;
    if (tid == 0) s_max = local_max;
    __syncthreads();

    float local_sum = 0.0f;
    for (int s = tid; s < seq_len; s += blockDim.x) {
        float exp_val = expf(s_scores[s] - s_max);
        s_scores[s] = exp_val;
        local_sum += exp_val;
    }
    local_sum = mla_warp_reduce_sum(local_sum);

    __shared__ float s_sum;
    if (tid == 0) s_sum = local_sum > 1e-8f ? local_sum : 1.0f;
    __syncthreads();

    for (int s = tid; s < seq_len; s += blockDim.x) {
        s_scores[s] /= s_sum;
    }
    __syncthreads();

    // Step 3: Accumulate in latent space
    for (int j = tid; j < kv_lora_rank; j += blockDim.x) {
        float sum_j = 0.0f;
        for (int s = 0; s < seq_len; s++) {
            sum_j += s_scores[s] * mla_to_float(k_cache[(size_t)s * kv_lora_rank + j]);
        }
        s_latent[j] = sum_j;
    }
    __syncthreads();

    // Step 4: Project to output heads
    const WType* cur_w_uv = w_uv + (size_t)(h * v_head_dim) * kv_lora_rank;
    for (int i = tid; i < v_head_dim; i += blockDim.x) {
        float out_val = 0.0f;
        const WType* w_row = cur_w_uv + (size_t)i * kv_lora_rank;
        for (int j = 0; j < kv_lora_rank; j++) {
            out_val += s_latent[j] * mla_to_float(w_row[j]);
        }
        output[(size_t)(t * heads + h) * v_head_dim + i] = mla_from_float<OutType>(out_val);
    }
}

// =============================================================================
// C API Wrappers
// =============================================================================

extern "C" {

void cudaMLADecompressKV(cudaStream_t stream, const void* compressed_kv, const void* w_ukv,
                         void* k_nope, void* v, int num_tokens, int kv_lora_rank,
                         int heads, int qk_nope_dim, int v_head_dim, int isF16) {
    if (num_tokens <= 0 || kv_lora_rank <= 0 || heads <= 0) return;
    int total_rows = heads * (qk_nope_dim + v_head_dim);
    int total_work = num_tokens * total_rows;
    int threadsPerBlock = 256;
    int warpsPerBlock = threadsPerBlock / 32;
    int numBlocks = (total_work + warpsPerBlock - 1) / warpsPerBlock;
    if (numBlocks > 1024) numBlocks = 1024;
    if (numBlocks < 1) numBlocks = 1;

    if (isF16) {
        mla_decompress_kv_kernel<__half, __half, __half><<<numBlocks, threadsPerBlock, 0, stream>>>(
            (const __half*)compressed_kv, (const __half*)w_ukv,
            (__half*)k_nope, (__half*)v,
            num_tokens, kv_lora_rank, heads, qk_nope_dim, v_head_dim);
    } else {
        mla_decompress_kv_kernel<float, float, float><<<numBlocks, threadsPerBlock, 0, stream>>>(
            (const float*)compressed_kv, (const float*)w_ukv,
            (float*)k_nope, (float*)v,
            num_tokens, kv_lora_rank, heads, qk_nope_dim, v_head_dim);
    }
}

void cudaMLAProjectQuerySplitRoPE(cudaStream_t stream, const void* q_all, const int* pos_ids,
                                  void* q_nope, void* q_rope, int num_tokens, int heads,
                                  int qk_nope_dim, int qk_rope_dim, float theta, int isF16) {
    if (num_tokens <= 0 || heads <= 0) return;
    int total_head_tokens = num_tokens * heads;
    int threadsPerBlock = 256;
    int numBlocks = (total_head_tokens + threadsPerBlock - 1) / threadsPerBlock;
    if (numBlocks > 1024) numBlocks = 1024;
    if (numBlocks < 1) numBlocks = 1;

    if (isF16) {
        mla_project_query_split_rope_kernel<__half, __half><<<numBlocks, threadsPerBlock, 0, stream>>>(
            (const __half*)q_all, pos_ids,
            (__half*)q_nope, (__half*)q_rope,
            num_tokens, heads, qk_nope_dim, qk_rope_dim, theta);
    } else {
        mla_project_query_split_rope_kernel<float, float><<<numBlocks, threadsPerBlock, 0, stream>>>(
            (const float*)q_all, pos_ids,
            (float*)q_nope, (float*)q_rope,
            num_tokens, heads, qk_nope_dim, qk_rope_dim, theta);
    }
}

void cudaMLAAbsorbedQuery(cudaStream_t stream, const void* q_nope, const void* w_uk,
                          void* q_absorbed, int num_tokens, int heads, int qk_nope_dim,
                          int kv_lora_rank, int isF16) {
    if (num_tokens <= 0 || heads <= 0 || kv_lora_rank <= 0) return;
    int total_work = num_tokens * heads * kv_lora_rank;
    int threadsPerBlock = 256;
    int warpsPerBlock = threadsPerBlock / 32;
    int numBlocks = (total_work + warpsPerBlock - 1) / warpsPerBlock;
    if (numBlocks > 1024) numBlocks = 1024;
    if (numBlocks < 1) numBlocks = 1;

    if (isF16) {
        mla_absorbed_query_kernel<__half, __half, __half><<<numBlocks, threadsPerBlock, 0, stream>>>(
            (const __half*)q_nope, (const __half*)w_uk,
            (__half*)q_absorbed, num_tokens, heads, qk_nope_dim, kv_lora_rank);
    } else {
        mla_absorbed_query_kernel<float, float, float><<<numBlocks, threadsPerBlock, 0, stream>>>(
            (const float*)q_nope, (const float*)w_uk,
            (float*)q_absorbed, num_tokens, heads, qk_nope_dim, kv_lora_rank);
    }
}

void cudaMLAAbsorbedDecodeAttention(cudaStream_t stream, const void* q_absorbed, const void* q_rope,
                                     const void* k_cache, const void* k_rope_cache, const void* w_uv,
                                     void* output, int num_tokens, int seq_len, int heads,
                                     int kv_lora_rank, int qk_rope_dim, int v_head_dim,
                                     float scale, int isF16) {
    if (num_tokens <= 0 || seq_len <= 0 || heads <= 0) return;
    dim3 grid(heads, num_tokens);
    int threadsPerBlock = 128;
    size_t smemSize = (seq_len * sizeof(float)) + (kv_lora_rank * sizeof(float)) + 64;

    if (isF16) {
        mla_absorbed_decode_attention_kernel<__half, __half, __half><<<grid, threadsPerBlock, smemSize, stream>>>(
            (const __half*)q_absorbed, (const __half*)q_rope,
            (const __half*)k_cache, (const __half*)k_rope_cache,
            (const __half*)w_uv, (__half*)output,
            num_tokens, seq_len, heads, kv_lora_rank, qk_rope_dim, v_head_dim, scale);
    } else {
        mla_absorbed_decode_attention_kernel<float, float, float><<<grid, threadsPerBlock, smemSize, stream>>>(
            (const float*)q_absorbed, (const float*)q_rope,
            (const float*)k_cache, (const float*)k_rope_cache,
            (const float*)w_uv, (float*)output,
            num_tokens, seq_len, heads, kv_lora_rank, qk_rope_dim, v_head_dim, scale);
    }
}

} // extern "C"
