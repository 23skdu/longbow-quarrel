#include <metal_stdlib>
using namespace metal;

static inline float simd_sum(float val) {
    for (uint offset = 16; offset > 0; offset /= 2) val += simd_shuffle_down(val, offset);
    return simd_broadcast(val, 0);
}

// Fixed Linear: 32 threads per row (1 SIMD group) for safe reduction
kernel void linear_f16(device const half *weight [[ buffer(0) ]],
                     device const half *input [[ buffer(1) ]],
                     device half *output [[ buffer(2) ]],
                     constant int &dim_in [[ buffer(3) ]],
                     constant int &dim_out [[ buffer(4) ]],
                     uint2 tid [[ thread_position_in_threadgroup ]],
                     uint2 qid [[ thread_position_in_grid ]]) {
    uint row = qid.y; if (row >= (uint)dim_out) return; uint lane = tid.x;
    device const half4 *w4 = (device const half4 *)(weight + row * dim_in);
    device const half4 *i4 = (device const half4 *)input;
    int n4 = dim_in / 4;
    float sum = 0; for (int i = (int)lane; i < n4; i += 32) {
        float4 v_w = (float4)w4[i]; float4 v_i = (float4)i4[i];
        sum += dot(v_w.xy, v_i.xy) + dot(v_w.zw, v_i.zw);
    }
    sum = simd_sum(sum); if (lane == 0) output[row] = (half)sum;
}

// Fixed RMSNorm: Single Threadgroup per Row, Out-of-Place
kernel void rmsnorm_f16(device const half *x [[ buffer(0) ]],
                      device half *out [[ buffer(1) ]],
                      device const half *w [[ buffer(2) ]],
                      constant float &eps [[ buffer(3) ]],
                      constant int &cols [[ buffer(4) ]],
                      uint tid [[ thread_index_in_threadgroup ]],
                      uint qid [[ thread_position_in_grid ]]) {
    // assumes 1 threadgroup per row.
    threadgroup float s[1024]; 
    float val = (tid < (uint)cols) ? (float)x[qid] : 0.0f;
    s[tid] = val * val; 
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tid == 0) { 
        float t = 0; for (uint i = 0; i < (uint)cols; i++) t += s[i]; 
        s[0] = 1.0f / sqrt(t / (float)cols + eps); 
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < (uint)cols) out[qid] = (half)((float)x[qid] * s[0] * (float)w[qid]);
}

kernel void rope_f16(device half *x [[ buffer(0) ]],
                    constant int &pos [[ buffer(1) ]],
                    constant int &headDim [[ buffer(2) ]],
                    constant float &ropeTheta [[ buffer(3) ]],
                    uint qid [[ thread_position_in_grid ]]) {
    int h = (int)(qid / (headDim / 2)), i = (int)(qid % (headDim / 2)), off = h * headDim;
    float th = (float)pos * pow(ropeTheta, -2.0f * (float)i / (float)headDim);
    float ct = cos(th), st = sin(th);
    float x1 = (float)x[off + i], x2 = (float)x[off + i + headDim/2];
    x[off + i] = (half)(x1 * ct - x2 * st);
    x[off + i + headDim/2] = (half)(x1 * st + x2 * ct);
}

kernel void att_scores_f16(device const half *q [[ buffer(0) ]],
                         device const half *k_cache [[ buffer(1) ]],
                         device float *scores [[ buffer(2) ]],
                         constant int &pos [[ buffer(3) ]],
                         constant int &num_heads [[ buffer(4) ]],
                         constant int &kv_heads [[ buffer(5) ]],
                         constant int &headDim [[ buffer(6) ]],
                         constant int &stride [[ buffer(7) ]],
                         uint qid [[ thread_position_in_grid ]]) {
    uint h = qid / 32, lane = qid % 32; if (h >= (uint)num_heads) return;
    uint kvh = h / (num_heads / kv_heads), kv_dim = kv_heads * headDim;
    float scale = 1.0f / sqrt((float)headDim); device const half *mq = q + h * headDim;
    for (int t = 0; t <= pos; t++) {
        float d = 0; device const half *mk = k_cache + t * kv_dim + kvh * headDim;
        for (int i = (int)lane; i < headDim; i += 32) d += (float)mq[i] * (float)mk[i];
        d = simd_sum(d); if (lane == 0) scores[h * stride + t] = d * scale;
    }
}

kernel void softmax_f16(device float *scores [[ buffer(0) ]],
                      constant int &pos [[ buffer(1) ]],
                      constant int &stride [[ buffer(2) ]],
                      uint qid [[ thread_position_in_grid ]]) {
    device float *s = scores + qid * stride;
    float mv = -1e20f; for (int i = 0; i <= pos; i++) if (s[i] > mv) mv = s[i];
    float se = 0; for (int i = 0; i <= pos; i++) { float e = exp(s[i] - mv); s[i] = e; se += e; }
    for (int i = 0; i <= pos; i++) s[i] /= se;
}

kernel void att_values_f16(device const float *scores [[ buffer(0) ]],
                         device const half *v_cache [[ buffer(1) ]],
                         device half *output [[ buffer(2) ]],
                         constant int &pos [[ buffer(3) ]],
                         constant int &num_heads [[ buffer(4) ]],
                         constant int &kv_heads [[ buffer(5) ]],
                         constant int &headDim [[ buffer(6) ]],
                         constant int &stride [[ buffer(7) ]],
                         uint qid [[ thread_position_in_grid ]]) {
    uint h = qid / headDim, idx = qid % headDim; if (h >= (uint)num_heads) return;
    uint kvh = h / (num_heads / kv_heads), kv_dim = kv_heads * headDim;
    float r = 0; for (int t = 0; t <= pos; t++) r += scores[h * stride + t] * (float)v_cache[t * kv_dim + kvh * headDim + idx];
    output[qid] = (half)r;
}

kernel void swiglu_f16(device half *gate [[ buffer(0) ]], device const half *up [[ buffer(1) ]], uint qid [[ thread_position_in_grid ]]) {
    float g = (float)gate[qid]; gate[qid] = (half)((float)up[qid] * (g / (1.0f + exp(-g))));
}

kernel void embedding_f16(device const half *weight [[ buffer(0) ]], device half *output [[ buffer(1) ]], constant int &idx [[ buffer(2) ]], constant int &cols [[ buffer(3) ]], uint qid [[ thread_position_in_grid ]]) {
    output[qid] = weight[idx * (uint)cols + qid];
}

kernel void add_f16(device half *x [[ buffer(0) ]], device const half *y [[ buffer(1) ]], uint qid [[ thread_position_in_grid ]]) {
    x[qid] += y[qid];
}

kernel void copy_f16(device const half *src [[ buffer(0) ]], device half *dst [[ buffer(1) ]], uint qid [[ thread_position_in_grid ]]) {
    dst[qid] = src[qid];
}
