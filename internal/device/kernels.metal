#include <metal_stdlib>
using namespace metal;

// ============ FP16 Kernels ============

kernel void add_kernel_f16(device const half *a [[ buffer(0) ]],
                           device const half *b [[ buffer(1) ]],
                           device half *result [[ buffer(2) ]],
                           uint index [[ thread_position_in_grid ]]) {
    result[index] = a[index] + b[index];
}

kernel void scale_kernel_f16(device const half *a [[ buffer(0) ]],
                             constant half &val [[ buffer(1) ]],
                             device half *result [[ buffer(2) ]],
                             uint index [[ thread_position_in_grid ]]) {
    result[index] = a[index] * val;
}

// Embedding Lookup
// Copy row 'idx' from weights to result
kernel void embedding_kernel_f16(device const half *weights [[ buffer(0) ]],
                                 device half *result [[ buffer(1) ]],
                                 constant int &row_idx [[ buffer(2) ]],
                                 constant int &cols [[ buffer(3) ]],
                                 uint index [[ thread_position_in_grid ]]) {
    int offset = row_idx * cols + index;
    result[index] = weights[offset];
}

// RMSNorm
// out = x * (1 / sqrt(mean(x^2) + eps)) * weight
kernel void rmsnorm_kernel_f16(device const half *input [[ buffer(0) ]],
                               device const half *weight [[ buffer(1) ]], // gamma
                               device half *output [[ buffer(2) ]],
                               constant int &cols [[ buffer(3) ]],
                               constant float &eps [[ buffer(4) ]],
                               uint row_idx [[ threadgroup_position_in_grid ]],
                               uint tid [[ thread_index_in_threadgroup ]],
                               uint tg_size [[ threads_per_threadgroup ]]) {
    
    threadgroup float shared_sq_sum[256];
    
    int offset = row_idx * cols;
    
    // Phase 1: Sum of squares
    float local_sq_sum = 0.0;
    for (int i = tid; i < cols; i += tg_size) {
        float val = float(input[offset + i]);
        local_sq_sum += val * val;
    }
    shared_sq_sum[tid] = local_sq_sum;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sq_sum[tid] += shared_sq_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float mean_sq = shared_sq_sum[0] / float(cols);
    float inv_rms = 1.0 / sqrt(mean_sq + eps);
    
    // Phase 2: Create output
    for (int i = tid; i < cols; i += tg_size) {
        float val = float(input[offset + i]);
        output[offset + i] = half(val * inv_rms * float(weight[i]));
    }
}

// RoPE
kernel void rope_kernel_f16(device half *data [[ buffer(0) ]],
                            constant int &head_dim [[ buffer(1) ]],
                            constant int &num_heads [[ buffer(2) ]],
                            constant int &seq_len [[ buffer(3) ]],
                            constant int &pos_offset [[ buffer(4) ]],
                            uint3 gid [[ thread_position_in_grid ]]) {
    uint i = gid.x; // feature pair index (0 to head_dim/2 - 1)
    uint h = gid.y; // head index
    uint b_s = gid.z; // (batch * seq) index
    
    uint seq_idx = b_s % seq_len;
    uint offset = b_s * (num_heads * head_dim) + h * head_dim;
    
    // Effective position
    float pos = float(pos_offset + seq_idx);
    float theta = pos * pow(10000.0, -2.0 * (float)i / (float)head_dim);
    
    float cos_theta = cos(theta);
    float sin_theta = sin(theta);
    
    half x1 = data[offset + i];
    half x2 = data[offset + i + head_dim/2];
    
    data[offset + i] = half((float)x1 * cos_theta - (float)x2 * sin_theta);
    data[offset + i + head_dim/2] = half((float)x1 * sin_theta + (float)x2 * cos_theta);
}

// SwiGLU
// input_val: Up projection
// input_gate: Gate projection (to be silu'd)
kernel void swiglu_kernel_f16(device const half *input_val [[ buffer(0) ]],
                              device const half *input_gate [[ buffer(1) ]],
                              device half *output [[ buffer(2) ]],
                              constant int &inter_size [[ buffer(3) ]],
                              uint2 gid [[ thread_position_in_grid ]]) {
    uint i = gid.x; // index in inter_size
    uint n = gid.y; // index in N
    
    int offset = n * inter_size + i;
    
    float val = (float)input_val[offset];   
    float gate = (float)input_gate[offset]; 
    
    // silu(gate) = gate * sigmoid(gate)
    float silu_gate = gate / (1.0f + exp(-gate));
    output[offset] = half(val * silu_gate);
}

// Softmax
kernel void softmax_kernel_f16(device const half *input [[ buffer(0) ]],
                               device half *output [[ buffer(1) ]],
                               constant int &cols [[ buffer(2) ]],
                               uint row_idx [[ threadgroup_position_in_grid ]],
                               uint tid [[ thread_index_in_threadgroup ]],
                               uint tg_size [[ threads_per_threadgroup ]]) {
    
    threadgroup float shared_max[256];
    threadgroup float shared_sum[256];
    
    int offset = row_idx * cols;
    
    // Find max
    float local_max = -1e38;
    for (int i = tid; i < cols; i += tg_size) {
        float v = float(input[offset + i]);
        if (v > local_max) local_max = v;
    }
    shared_max[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_val = shared_max[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Exp sum
    float local_sum = 0.0;
    for (int i = tid; i < cols; i += tg_size) {
        float val = exp(float(input[offset + i]) - max_val);
        output[offset + i] = half(val); // store intermediate exp
        local_sum += val;
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0 / shared_sum[0];
    
    // Normalize
    for (int i = tid; i < cols; i += tg_size) {
        output[offset + i] = half(float(output[offset + i]) * inv_sum);
    }
}

// KV Cache Store
kernel void kv_store_f16(device const half *k [[ buffer(0) ]],
                         device const half *v [[ buffer(1) ]],
                         device half *k_cache [[ buffer(2) ]],
                         device half *v_cache [[ buffer(3) ]],
                         constant int &pos [[ buffer(4) ]],
                         constant int &num_heads [[ buffer(5) ]],
                         constant int &head_dim [[ buffer(6) ]],
                         uint head_idx [[ thread_position_in_grid ]]) {
    int cache_offset = (pos * num_heads + head_idx) * head_dim;
    int input_offset = head_idx * head_dim;
    for (int i = 0; i < head_dim; i++) {
        k_cache[cache_offset + i] = k[input_offset + i];
        v_cache[cache_offset + i] = v[input_offset + i];
    }
}

// Fused GQA Attention for Decoding (Single Token)
kernel void attention_f16(device const half *q [[ buffer(0) ]],
                          device const half *k_cache [[ buffer(1) ]],
                          device const half *v_cache [[ buffer(2) ]],
                          device half *out [[ buffer(3) ]],
                          constant int &pos [[ buffer(4) ]],
                          constant int &num_heads [[ buffer(5) ]],
                          constant int &kv_heads [[ buffer(6) ]],
                          constant int &head_dim [[ buffer(7) ]],
                          uint head_idx [[ thread_position_in_grid ]]) {
    int heads_per_kv = num_heads / kv_heads;
    int kv_head_idx = head_idx / heads_per_kv;
    float scale = 1.0f / sqrt((float)head_dim);
    
    float max_val = -1e10;
    
    // Pass 1: Max for Softmax
    for (int t = 0; t <= pos; t++) {
        float sum = 0;
        int q_off = head_idx * head_dim;
        int k_off = (t * kv_heads + kv_head_idx) * head_dim;
        for (int i = 0; i < head_dim; i++) {
            sum += (float)q[q_off + i] * (float)k_cache[k_off + i];
        }
        sum *= scale;
        if (sum > max_val) max_val = sum;
    }
    
    // Pass 2: Exp Sum
    float sum_exp = 0;
    for (int t = 0; t <= pos; t++) {
        float sum = 0;
        int q_off = head_idx * head_dim;
        int k_off = (t * kv_heads + kv_head_idx) * head_dim;
        for (int i = 0; i < head_dim; i++) {
            sum += (float)q[q_off + i] * (float)k_cache[k_off + i];
        }
        sum *= scale;
        sum_exp += exp(sum - max_val);
    }
    
    // Pass 3: Weighted Sum
    for (int i = 0; i < head_dim; i++) {
        float weighted_sum = 0;
        for (int t = 0; t <= pos; t++) {
            float sum = 0;
            int q_off = head_idx * head_dim;
            int k_off = (t * kv_heads + kv_head_idx) * head_dim;
            for (int k_i = 0; k_i < head_dim; k_i++) {
                sum += (float)q[q_off + k_i] * (float)k_cache[k_off + k_i];
            }
            sum *= scale;
            float attn_weight = exp(sum - max_val) / sum_exp;
            
            int v_off = (t * kv_heads + kv_head_idx) * head_dim;
            weighted_sum += attn_weight * (float)v_cache[v_off + i];
        }
        out[head_idx * head_dim + i] = (half)weighted_sum;
    }
}

// Fused RMSNorm + Linear (xW^T) Kernel
// Computes: normalized = RMSNorm(input, norm_weight) then output = normalized * weight^T
// This eliminates the intermediate normalized buffer
kernel void rmsnorm_linear_f16(device const half *input [[ buffer(0) ]],
                               device const half *norm_weight [[ buffer(1) ]],
                               device const half *weight [[ buffer(2) ]],  // [out_dim, in_dim]
                               device half *output [[ buffer(3) ]],
                               constant int &in_dim [[ buffer(4) ]],
                               constant int &out_dim [[ buffer(5) ]],
                               constant float &eps [[ buffer(6) ]],
                               uint row_idx [[ threadgroup_position_in_grid ]],
                               uint tid [[ thread_index_in_threadgroup ]],
                               uint tg_size [[ threads_per_threadgroup ]]) {
    
    threadgroup float shared_sq_sum[256];
    threadgroup half shared_normed[576];  // Max dimension for smollm2
    
    int input_offset = row_idx * in_dim;
    
    // Phase 1: Compute RMS
    float local_sq_sum = 0.0;
    for (int i = tid; i < in_dim; i += tg_size) {
        float val = float(input[input_offset + i]);
        local_sq_sum += val * val;
    }
    shared_sq_sum[tid] = local_sq_sum;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction for sum
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sq_sum[tid] += shared_sq_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float inv_rms = 1.0 / sqrt(shared_sq_sum[0] / float(in_dim) + eps);
    
    // Phase 2: Normalize and store in shared memory
    for (int i = tid; i < in_dim; i += tg_size) {
        float val = float(input[input_offset + i]);
        shared_normed[i] = half(val * inv_rms * float(norm_weight[i]));
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Phase 3: MatMul (normalized * weight^T)
    // Each thread computes one output element
    int output_offset = row_idx * out_dim;
    for (int i = tid; i < out_dim; i += tg_size) {
        float sum = 0.0;
        // weight is [out_dim, in_dim], row-major
        int weight_offset = i * in_dim;
        for (int j = 0; j < in_dim; j++) {
            sum += float(shared_normed[j]) * float(weight[weight_offset + j]);
        }
        output[output_offset + i] = half(sum);
    }
}
