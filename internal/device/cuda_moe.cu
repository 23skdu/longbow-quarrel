#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

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

    // Find top-k using simple selection (O(num_experts * top_k))
    // For production, use more efficient algorithm
    for (int k = 0; k < top_k; k++) {
        float max_val = -FLT_MAX;
        int max_idx = 0;

        for (int e = 0; e < num_experts; e++) {
            // Check if already selected
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

// Expert forward pass (sparse Mixture of Experts)
__global__ void moe_expert_forward_kernel(
    float* input,          // [batch, dim]
    float* expert_weights,  // [hidden_dim, dim, num_experts] or similar
    int* indices,          // [batch, top_k]
    float* expert_weights_w, // [batch, top_k] weights
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

            // Simplified: assume expert_weights is [num_experts * hidden_dim * dim]
            // In practice, use proper tensor layout based on model
            size_t offset = ((size_t)expert * hidden_dim + 0) * dim + d;
            float expert_out = input[b * dim + d]; // Identity for now

            // Real implementation would compute: output = sum_k weight_k * expert_k(input)
            sum += weight * expert_out;
        }

        output[idx] = sum;
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
        sum += input[b * dim + d] * gate[e * dim + d]; // Transposed gate
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

        // Gate: gate = input @ gate_experts[expert]
        float gate_val = 0.0f;
        for (int d = 0; d < dim; d++) {
            gate_val += input[b * dim + d] * gate_experts[((size_t)expert * hidden_dim + h) * dim + d];
        }
        gate_val = gate_val / (1.0f + expf(-gate_val)); // SiLU

        // Up: up = input @ up_experts[expert]
        float up_val = 0.0f;
        for (int d = 0; d < dim; d++) {
            up_val += input[b * dim + d] * up_experts[((size_t)expert * hidden_dim + h) * dim + d];
        }

        // SwiGLU: gate * up
        sum += weight * gate_val * up_val;
    }

    output[b * hidden_dim + h] = sum;
}

// Expert down projection
__global__ void moe_down_kernel(
    float* input,      // [batch, hidden_dim] - expert outputs
    float* down,       // [dim, hidden_dim, num_experts]
    int* indices,      // [batch, top_k]
    float* weights,     // [batch, top_k]
    float* output,      // [batch, dim]
    int batch, int dim, int hidden_dim, int num_experts, int top_k) {

    int b = blockIdx.x;
    int d = threadIdx.x;

    if (b >= batch || d >= dim) return;

    float sum = 0.0f;

    for (int k = 0; k < top_k; k++) {
        int expert = indices[b * top_k + k];
        float weight = weights[b * top_k + k];

        float expert_out = input[b * hidden_dim + (d % hidden_dim)];

        // Simplified: assume diagonal down projection for now
        sum += weight * expert_out;
    }

    output[b * dim + d] = sum;
}

extern "C" {

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

} // extern "C"
