#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

// =============================================================================
// Helper Functions
// =============================================================================

#ifndef CUDA_MOE_HELPERS
#define CUDA_MOE_HELPERS

__device__ __forceinline__ float moe_to_float(float x) { return x; }
__device__ __forceinline__ float moe_to_float(__half x) { return __half2float(x); }

template <typename T>
__device__ __forceinline__ T moe_from_float(float x);

template <>
__device__ __forceinline__ float moe_from_float<float>(float x) { return x; }

template <>
__device__ __forceinline__ __half moe_from_float<__half>(float x) { return __float2half(x); }

__device__ __forceinline__ float moe_warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

#endif // CUDA_MOE_HELPERS

// =============================================================================
// MOE (Mixture of Experts) Kernels
// =============================================================================

// Router logits computation: input @ gate.T
// input: [batch, dim]
// gate:  [num_experts, dim]
// output:[batch, num_experts] (FP32)
template <typename InType, typename WType>
__global__ void moe_router_logits_kernel(
    const InType* __restrict__ input,
    const WType* __restrict__ gate,
    float* __restrict__ output,
    int batch, int dim, int num_experts) {

    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int totalWarps = (gridDim.x * blockDim.x) / 32;

    int totalWork = batch * num_experts;
    for (int workIdx = warpId; workIdx < totalWork; workIdx += totalWarps) {
        int b = workIdx / num_experts;
        int e = workIdx % num_experts;

        const InType* in_ptr = input + (size_t)b * dim;
        const WType* g_ptr = gate + (size_t)e * dim;

        float sum = 0.0f;
        for (int d = lane; d < dim; d += 32) {
            sum += moe_to_float(in_ptr[d]) * moe_to_float(g_ptr[d]);
        }
        sum = moe_warp_reduce_sum(sum);

        if (lane == 0) {
            output[workIdx] = sum;
        }
    }
}

// Top-K selection with fused softmax
// logits:  [batch, num_experts]
// indices: [batch, top_k] (stored as float)
// weights: [batch, top_k]
__global__ void moe_topk_kernel(
    const float* __restrict__ logits,
    float* __restrict__ indices,
    float* __restrict__ weights,
    int batch, int num_experts, int top_k) {

    int token = blockIdx.x;
    if (token >= batch) return;

    int lane = threadIdx.x; // 32 threads in warp
    const float* logit_row = logits + (size_t)token * num_experts;
    float* idx_row = indices + (size_t)token * top_k;
    float* weight_row = weights + (size_t)token * top_k;

    __shared__ int s_selected[32];
    __shared__ float s_logits[32];

    for (int k = 0; k < top_k && k < 32; k++) {
        float local_max = -1e30f;
        int local_idx = -1;

        for (int e = lane; e < num_experts; e += 32) {
            bool already_selected = false;
            for (int prev = 0; prev < k; prev++) {
                if (s_selected[prev] == e) {
                    already_selected = true;
                    break;
                }
            }
            if (already_selected) continue;

            float val = logit_row[e];
            if (val > local_max) {
                local_max = val;
                local_idx = e;
            }
        }

        // Warp reduction to find argmax across the 32 threads
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_val = __shfl_down_sync(0xffffffff, local_max, offset);
            int other_idx = __shfl_down_sync(0xffffffff, local_idx, offset);
            if (other_val > local_max || (other_val == local_max && other_idx >= 0 && (local_idx < 0 || other_idx < local_idx))) {
                local_max = other_val;
                local_idx = other_idx;
            }
        }

        if (lane == 0) {
            s_selected[k] = local_idx;
            s_logits[k] = local_max;
        }
        __syncwarp();
    }

    // Lane 0 computes softmax over the top_k chosen logits and writes outputs
    if (lane == 0) {
        float max_logit = -1e30f;
        for (int k = 0; k < top_k && k < 32; k++) {
            if (s_logits[k] > max_logit) {
                max_logit = s_logits[k];
            }
        }

        float sum_exp = 0.0f;
        for (int k = 0; k < top_k && k < 32; k++) {
            float e = expf(s_logits[k] - max_logit);
            s_logits[k] = e;
            sum_exp += e;
        }

        float inv_sum = 1.0f / (sum_exp + 1e-9f);
        for (int k = 0; k < top_k && k < 32; k++) {
            idx_row[k] = (float)s_selected[k];
            weight_row[k] = s_logits[k] * inv_sum;
        }
    }
}

// Expert forward pass (Sparse linear projection with weighted combination)
// input:          [batch, dim]
// expert_weights: [num_experts * out_dim, dim]
// indices:        [batch, top_k]
// weights:        [batch, top_k]
// output:         [batch, out_dim]
template <typename InType, typename WType, typename OutType>
__global__ void moe_expert_forward_kernel(
    const InType* __restrict__ input,
    const WType* __restrict__ expert_weights,
    const float* __restrict__ indices,
    const float* __restrict__ weights,
    OutType* __restrict__ output,
    int batch, int dim, int out_dim, int top_k) {

    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int totalWarps = (gridDim.x * blockDim.x) / 32;

    int totalWork = batch * out_dim;
    for (int workIdx = warpId; workIdx < totalWork; workIdx += totalWarps) {
        int b = workIdx / out_dim;
        int d = workIdx % out_dim;

        const InType* in_ptr = input + (size_t)b * dim;
        const float* idx_row = indices + (size_t)b * top_k;
        const float* weight_row = weights + (size_t)b * top_k;

        float sum = 0.0f;
        for (int k = 0; k < top_k; k++) {
            float w = weight_row[k];
            if (w == 0.0f) continue;
            int expert = (int)idx_row[k];
            if (expert < 0) continue;

            const WType* w_ptr = expert_weights + ((size_t)expert * out_dim + d) * dim;
            float expert_dot = 0.0f;
            for (int i = lane; i < dim; i += 32) {
                expert_dot += moe_to_float(in_ptr[i]) * moe_to_float(w_ptr[i]);
            }
            expert_dot = moe_warp_reduce_sum(expert_dot);
            sum += w * expert_dot;
        }

        if (lane == 0) {
            output[workIdx] = moe_from_float<OutType>(sum);
        }
    }
}

// Fused Gate + Up + SwiGLU for multiple experts
// input:        [batch, dim]
// gate_experts: [num_experts * hidden_dim, dim]
// up_experts:   [num_experts * hidden_dim, dim]
// indices:      [batch, top_k]
// weights:      [batch, top_k]
// output:       [batch, hidden_dim]
template <typename InType, typename WType, typename OutType>
__global__ void moe_gate_up_swiglu_kernel(
    const InType* __restrict__ input,
    const WType* __restrict__ gate_experts,
    const WType* __restrict__ up_experts,
    const float* __restrict__ indices,
    const float* __restrict__ weights,
    OutType* __restrict__ output,
    int batch, int dim, int hidden_dim, int top_k) {

    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int totalWarps = (gridDim.x * blockDim.x) / 32;

    int totalWork = batch * hidden_dim;
    for (int workIdx = warpId; workIdx < totalWork; workIdx += totalWarps) {
        int b = workIdx / hidden_dim;
        int h = workIdx % hidden_dim;

        const InType* in_ptr = input + (size_t)b * dim;
        const float* idx_row = indices + (size_t)b * top_k;
        const float* weight_row = weights + (size_t)b * top_k;

        float sum_activated = 0.0f;
        for (int k = 0; k < top_k; k++) {
            float w = weight_row[k];
            if (w == 0.0f) continue;
            int expert = (int)idx_row[k];
            if (expert < 0) continue;

            size_t expert_row = (size_t)expert * hidden_dim + h;
            const WType* gate_ptr = gate_experts + expert_row * dim;
            const WType* up_ptr = up_experts + expert_row * dim;

            float gate_dot = 0.0f;
            float up_dot = 0.0f;
            for (int i = lane; i < dim; i += 32) {
                float in_v = moe_to_float(in_ptr[i]);
                gate_dot += in_v * moe_to_float(gate_ptr[i]);
                up_dot += in_v * moe_to_float(up_ptr[i]);
            }
            gate_dot = moe_warp_reduce_sum(gate_dot);
            up_dot = moe_warp_reduce_sum(up_dot);

            // SwiGLU: up * (gate * sigmoid(gate))
            float g_clamped = fminf(fmaxf(gate_dot, -15.0f), 15.0f);
            float silu_gate = gate_dot / (1.0f + expf(-g_clamped));
            sum_activated += w * (silu_gate * up_dot);
        }

        if (lane == 0) {
            output[workIdx] = moe_from_float<OutType>(sum_activated);
        }
    }
}

// =============================================================================
// C Export Wrappers
// =============================================================================

extern "C" {

void cudaMOERouterLogits(cudaStream_t stream, const void* input, const void* gate, float* output,
                         int batch, int dim, int num_experts, int isF16) {
    int totalWork = batch * num_experts;
    int threadsPerBlock = 256;
    int warpsPerBlock = threadsPerBlock / 32;
    int numBlocks = (totalWork + warpsPerBlock - 1) / warpsPerBlock;
    if (numBlocks > 1024) numBlocks = 1024;
    if (numBlocks < 1) numBlocks = 1;

    if (isF16) {
        moe_router_logits_kernel<__half, __half><<<numBlocks, threadsPerBlock, 0, stream>>>(
            (const __half*)input, (const __half*)gate, output, batch, dim, num_experts);
    } else {
        moe_router_logits_kernel<float, float><<<numBlocks, threadsPerBlock, 0, stream>>>(
            (const float*)input, (const float*)gate, output, batch, dim, num_experts);
    }
}

void cudaMOETopKSelection(cudaStream_t stream, const float* logits, int top_k,
                          float* indices, float* weights, int batch, int num_experts) {
    moe_topk_kernel<<<batch, 32, 0, stream>>>(logits, indices, weights, batch, num_experts, top_k);
}

void cudaMOEExpertForward(cudaStream_t stream, const void* input, const void* expert_weights,
                          const float* indices, const float* expert_weights_w, void* output,
                          int batch, int dim, int hidden_dim, int num_experts, int top_k, int isF16) {
    int totalWork = batch * hidden_dim;
    int threadsPerBlock = 256;
    int warpsPerBlock = threadsPerBlock / 32;
    int numBlocks = (totalWork + warpsPerBlock - 1) / warpsPerBlock;
    if (numBlocks > 1024) numBlocks = 1024;
    if (numBlocks < 1) numBlocks = 1;

    if (isF16) {
        moe_expert_forward_kernel<__half, __half, __half><<<numBlocks, threadsPerBlock, 0, stream>>>(
            (const __half*)input, (const __half*)expert_weights, indices, expert_weights_w,
            (__half*)output, batch, dim, hidden_dim, top_k);
    } else {
        moe_expert_forward_kernel<float, float, float><<<numBlocks, threadsPerBlock, 0, stream>>>(
            (const float*)input, (const float*)expert_weights, indices, expert_weights_w,
            (float*)output, batch, dim, hidden_dim, top_k);
    }
}

void cudaMOEExpertGateUpSwiGLU(cudaStream_t stream, const void* input,
                               const void* gate_experts, const void* up_experts,
                               const float* indices, const float* weights, void* output,
                               int batch, int dim, int hidden_dim,
                               int num_experts, int top_k, int isF16) {
    int totalWork = batch * hidden_dim;
    int threadsPerBlock = 256;
    int warpsPerBlock = threadsPerBlock / 32;
    int numBlocks = (totalWork + warpsPerBlock - 1) / warpsPerBlock;
    if (numBlocks > 1024) numBlocks = 1024;
    if (numBlocks < 1) numBlocks = 1;

    if (isF16) {
        moe_gate_up_swiglu_kernel<__half, __half, __half><<<numBlocks, threadsPerBlock, 0, stream>>>(
            (const __half*)input, (const __half*)gate_experts, (const __half*)up_experts,
            indices, weights, (__half*)output, batch, dim, hidden_dim, top_k);
    } else {
        moe_gate_up_swiglu_kernel<float, float, float><<<numBlocks, threadsPerBlock, 0, stream>>>(
            (const float*)input, (const float*)gate_experts, (const float*)up_experts,
            indices, weights, (float*)output, batch, dim, hidden_dim, top_k);
    }
}

} // extern "C"
