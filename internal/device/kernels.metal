#include <metal_stdlib>
using namespace metal;

static inline float simd_sum(float val) {
    for (uint offset = 16; offset > 0; offset /= 2) val += simd_shuffle_down(val, offset);
    return simd_broadcast(val, 0);
}

static inline half safe_half(float x) {
    if (isnan(x)) return (half)0.0f;
    if (x > 65504.0f) return (half)65504.0f;
    if (x < -65504.0f) return (half)-65504.0f;
    return (half)x;
}

// Manual FP16 to FP32 conversion with proper subnormal handling
float fp16_to_fp32(ushort f) {
    uint sign = (uint(f) >> 15) & 0x1;
    uint exp = (uint(f) >> 10) & 0x1f;
    uint mant = uint(f) & 0x3ff;
    
    if (exp == 0) {
        if (mant == 0) {
            return as_type<float>(sign << 31);
        }
        // Subnormal: normalize it
        uint shift = 0;
        uint temp = mant;
        while (temp < 0x400) {
            temp <<= 1;
            shift++;
        }
        mant = (temp & 0x3ff) << 13;
        exp = 127 - 14 - shift;
        return as_type<float>((sign << 31) | (exp << 23) | mant);
    } else if (exp == 31) {
        if (mant == 0) {
            return as_type<float>((sign << 31) | 0x7f800000);
        }
        return as_type<float>((sign << 31) | 0x7f800000 | (mant << 13));
    }
    
    uint newExp = exp - 15 + 127;
    return as_type<float>((sign << 31) | (newExp << 23) | (mant << 13));
}

// Native casting is used instead of manual conversion

kernel void scale_f16(device const half *x [[ buffer(0) ]], device half *out [[ buffer(1) ]], constant float &scale [[ buffer(2) ]], uint qid [[ thread_position_in_grid ]]) {
    out[qid] = (half)((float)x[qid] * scale);
}

kernel void swiglu_f16(device const half *gate [[ buffer(0) ]], 
                     device const half *up [[ buffer(1) ]], 
                     device half *out [[ buffer(2) ]],
                     uint qid [[ thread_position_in_grid ]]) {
    float g = (float)gate[qid]; 
    float u = (float)up[qid];
    
    // SwiGLU: up * silu(gate) where silu(x) = x * sigmoid(x)
    float silu_g = g / (1.0f + exp(-g));
    float val = u * silu_g;
    
    // Note: SmolLM2 produces large intermediate values (50-60 range) which
    // causes activation explosion. Proper fix requires FP32 accumulation.
    // For now, allow values up to FP16 safe range.
    out[qid] = safe_half(val);
}

kernel void rmsnorm_f16(device const half *x [[ buffer(0) ]],
                      device half *out [[ buffer(1) ]],
                      device const half *w [[ buffer(2) ]],
                      constant float &eps [[ buffer(3) ]],
                      constant int &cols [[ buffer(4) ]],
                      uint tid [[ thread_index_in_threadgroup ]],
                      uint2 qid [[ thread_position_in_grid ]]) {
    threadgroup float s[1024]; 
    float sum = 0.0f;
    int row_offset = qid.y * cols;
    for (int i = tid; i < cols; i += 1024) {
        float val = (float)x[row_offset + i];
        sum += val * val;
    }
    s[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { 
        float t = 0; 
        int active_threads = (cols < 1024) ? cols : 1024;
        for (int i = 0; i < active_threads; i++) t += s[i]; 
        s[0] = 1.0f / sqrt(t / (float)cols + eps); 
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = s[0];
    for (int i = tid; i < cols; i += 1024) {
        int idx = row_offset + i;
        out[idx] = safe_half((float)x[idx] * scale * (float)w[i]);
    }
}

kernel void linear_f16(device const half *weight [[ buffer(0) ]],
                     device const half *input [[ buffer(1) ]],
                     device half *output [[ buffer(2) ]],
                     constant int &dim_in [[ buffer(3) ]],
                     constant int &dim_out [[ buffer(4) ]],
                     uint3 tid [[ thread_position_in_threadgroup ]],
                     uint3 qid [[ thread_position_in_grid ]]) {
    uint row = qid.y; uint batch = qid.z;
    if (row >= (uint)dim_out) return; uint lane_id = tid.x;
    device const half4 *w4 = (device const half4 *)(weight + row * dim_in);
    device const half4 *i4 = (device const half4 *)(input + batch * dim_in);
    int n4 = dim_in / 4;
    float sum = 0; for (int i = (int)lane_id; i < n4; i += 32) {
        float4 v_w = float4(w4[i]); float4 v_i = float4(i4[i]);
        sum += dot(v_w.xy, v_i.xy) + dot(v_w.zw, v_i.zw);
    }
    sum = simd_sum(sum); 
    if (lane_id == 0) output[batch * dim_out + row] = safe_half(sum);
}

kernel void linear_q4k_f16(device const uchar *weight [[ buffer(0) ]],
                         device const half *input [[ buffer(1) ]],
                         device half *output [[ buffer(2) ]],
                         constant int &dim_in [[ buffer(3) ]],
                         constant int &dim_out [[ buffer(4) ]],
                         uint3 tid [[ thread_position_in_threadgroup ]],
                         uint3 qid [[ thread_position_in_grid ]]) {
    uint row = qid.y; uint batch = qid.z;
    if (row >= (uint)dim_out) return; uint lane_id = tid.x;
    int num_blocks = dim_in / 256;
    device const uchar *row_ptr = weight + row * num_blocks * 144;
    device const half *in_ptr = input + batch * dim_in;
    float sum = 0;
    for (int i = (int)lane_id; i < num_blocks; i += 32) {
        device const uchar *block = row_ptr + i * 144;
        ushort d_bits = *(device const ushort*)(block);
        ushort dmin_bits = *(device const ushort*)(block + 2);
        float d = (float)as_type<half>(d_bits);
        float dmin = (float)as_type<half>(dmin_bits);
        
        device const uchar *scales = block + 4;
        device const uchar *qs = block + 16;
        uchar sc[8], m[8];
        for (int j = 0; j < 8; j++) {
            if (j < 4) {
                sc[j] = scales[j] & 63;
                m[j] = scales[j + 4] & 63;
            } else {
                sc[j] = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4);
                m[j] = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4);
            }
        }
        for (int j = 0; j < 8; j++) {
            float d_val = d * (float)sc[j], m_val = dmin * (float)m[j];
            int sub_offset = j * 32, qs_offset = j * 16;
            for (int k = 0; k < 16; k++) {
                uchar b = qs[qs_offset + k];
                float w0 = d_val * (float)(b & 0xF) - m_val;
                float w1 = d_val * (float)(b >> 4) - m_val;
                int idx0 = i * 256 + sub_offset + k;
                int idx1 = idx0 + 16;
                sum += w0 * (float)in_ptr[idx0] + w1 * (float)in_ptr[idx1];
            }
        }
    }
    sum = simd_sum(sum); 
    if (lane_id == 0) output[batch * dim_out + row] = safe_half(sum);
}

kernel void linear_q3k_f16(device const uchar *weight [[ buffer(0) ]],
                         device const half *input [[ buffer(1) ]],
                         device half *output [[ buffer(2) ]],
                         constant int &dim_in [[ buffer(3) ]],
                         constant int &dim_out [[ buffer(4) ]],
                         uint3 tid [[ thread_position_in_threadgroup ]],
                         uint3 qid [[ thread_position_in_grid ]]) {
    uint row = qid.y; uint batch = qid.z;
    if (row >= (uint)dim_out) return; uint lane_id = tid.x;
    int num_blocks = dim_in / 256;
    device const uchar *row_ptr = weight + row * num_blocks * 110;
    device const half *in_ptr = input + batch * dim_in;
    float sum = 0;
    for (int i = (int)lane_id; i < num_blocks; i += 32) {
        device const uchar *block = row_ptr + i * 110;
        device const uchar *hmask = block;
        device const uchar *qs = block + 32;
        device const uchar *scales = block + 96;
        float d = (float)*(device const half*)(block + 108);
        uchar sc[16];
        for (int j = 0; j < 4; j++) {
            uchar s0 = scales[j], s1 = scales[j+4], s2 = scales[j+8];
            sc[j] = s0 & 63; sc[j+4] = s1 & 63; sc[j+8] = s2 & 63;
            sc[j+12] = (s0 >> 6) | ((s1 >> 6) << 2) | ((s2 >> 6) << 4);
        }
        for (int l = 0; l < 16; l++) {
            float s = d * ((float)sc[l] - 32.0f);
            int sub_offset = l * 16;
            for (int k = 0; k < 16; k++) {
                int idxIB = sub_offset + k;
                uchar q2 = (qs[idxIB / 4] >> ((idxIB % 4) * 2)) & 3;
                uchar h = (hmask[idxIB / 8] >> (idxIB % 8)) & 1;
                float w = s * ((float)((h << 2) | q2) - 4.0f);
                sum += w * (float)in_ptr[i * 256 + idxIB];
            }
        }
    }
    sum = simd_sum(sum); 
    if (lane_id == 0) output[batch * dim_out + row] = (half)sum;
}

kernel void linear_q6k_f16(device const uchar *weight [[ buffer(0) ]],
                         device const half *input [[ buffer(1) ]],
                         device half *output [[ buffer(2) ]],
                         constant int &dim_in [[ buffer(3) ]],
                         constant int &dim_out [[ buffer(4) ]],
                         uint3 tid [[ thread_position_in_threadgroup ]],
                         uint3 qid [[ thread_position_in_grid ]]) {
    uint row = qid.y; uint batch = qid.z;
    if (row >= (uint)dim_out) return; uint lane_id = tid.x;
    int num_blocks = dim_in / 256;
    device const uchar *row_ptr = weight + row * num_blocks * 210;
    device const half *in_ptr = input + batch * dim_in;
    float sum = 0;
    
    for (int i = (int)lane_id; i < num_blocks; i += 32) {
        device const uchar *block = row_ptr + i * 210;
        device const uchar *ql = block;
        device const uchar *qh = block + 128;
        device const char  *sc = (device const char *)(block + 192);
        float d = (float)*(device const half*)(block + 208);
        
        for (int l = 0; l < 16; l++) {
            float s = d * (float)sc[l];
            int sub_off = l * 16;
            for (int k = 0; k < 16; k += 2) {
                int idx = sub_off + k;
                uchar b = ql[idx / 2];
                uchar h0 = (qh[idx / 4] >> ((idx % 4) * 2)) & 3;
                uchar h1 = (qh[(idx+1) / 4] >> (((idx+1) % 4) * 2)) & 3;
                
                float w0 = s * ((float)((int8_t)((h0 << 4) | (b & 0xF))) - 32.0f);
                float w1 = s * ((float)((int8_t)((h1 << 4) | (b >> 4))) - 32.0f);
                
                sum += w0 * (float)in_ptr[i * 256 + idx] + w1 * (float)in_ptr[i * 256 + idx + 1];
            }
        }
    }
    sum = simd_sum(sum); 
    if (lane_id == 0) output[batch * dim_out + row] = safe_half(sum);
}

kernel void rope_f16(device half *x [[ buffer(0) ]],
                    constant int &pos [[ buffer(1) ]],
                    constant int &headDim [[ buffer(2) ]],
                    constant float &ropeTheta [[ buffer(3) ]],
                    uint qid [[ thread_position_in_grid ]]) {
    int h = (int)(qid / (headDim / 2));
    int i = (int)(qid % (headDim / 2));
    int off = h * headDim;

    float th = (float)pos * pow(ropeTheta, -2.0f * (float)i / (float)headDim);
    float ct = cos(th), st = sin(th);
    
    // Mistral/Llama Pairing: [i] and [i + headDim/2]
    int idx1 = off + i;
    int idx2 = off + i + (headDim / 2);
    
    float x1 = (float)x[idx1]; 
    float x2 = (float)x[idx2];
    
    x[idx1] = safe_half(x1 * ct - x2 * st);
    x[idx2] = safe_half(x1 * st + x2 * ct);
}

kernel void embedding_f16(device const half *weight [[ buffer(0) ]], device half *output [[ buffer(1) ]], constant int &idx [[ buffer(2) ]], constant int &cols [[ buffer(3) ]], uint qid [[ thread_position_in_grid ]]) {
    output[qid] = weight[idx * (uint)cols + qid];
}

kernel void softmax_f16(device float *scores [[ buffer(0) ]],
                      constant int &pos [[ buffer(1) ]],
                      constant int &stride [[ buffer(2) ]],
                      uint tid [[ thread_position_in_threadgroup ]],
                      uint qid [[ threadgroup_position_in_grid ]]) {
    device float *s = scores + qid * (uint)stride;
    float mv = -10000.0f;
    for (int i = tid; i <= pos; i += 32) if (s[i] > mv) mv = s[i];
    mv = simd_max(mv);
    
    float se = 0;
    for (int i = tid; i <= pos; i += 32) {
        float e = exp(s[i] - mv);
        s[i] = e;
        se += e;
    }
    se = simd_sum(se);
    for (int i = tid; i <= pos; i += 32) s[i] /= se;
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
// kernel void att_scores_f16 ...
// ...
    // scale = 1.0 / sqrt(head_dim)
    // DEBUG: Force scale 1.0 because signals are small?
    float scale = 1.0f / sqrt((float)headDim);
 device const half *mq = q + h * headDim;
    for (int t = 0; t <= pos; t++) {
        float d = 0; device const half *mk = k_cache + t * kv_dim + kvh * headDim;
        for (int i = (int)lane; i < headDim; i += 32) d += (float)mq[i] * (float)mk[i];
        d = simd_sum(d); if (lane == 0) scores [h * stride + t] = d * scale;
    }
}

kernel void att_fused_f16(device const half *q [[ buffer(0) ]],
                        device const half *k_cache [[ buffer(1) ]],
                        device const half *v_cache [[ buffer(2) ]],
                        device half *output [[ buffer(3) ]],
                        constant int &pos [[ buffer(4) ]],
                        constant int &num_heads [[ buffer(5) ]],
                        constant int &kv_heads [[ buffer(6) ]],
                        constant int &headDim [[ buffer(7) ]],
                        uint3 tid [[ thread_position_in_threadgroup ]],
                        uint3 group_id [[ threadgroup_position_in_grid ]]) {
    uint h = group_id.x; if (h >= (uint)num_heads) return;
    uint kvh = h / (num_heads / kv_heads);
    uint kv_dim = kv_heads * headDim;
    float scale = 1.0f / sqrt((float)headDim);

    threadgroup float s_mem[1024]; // Max fused length
    device const half *mq = q + h * headDim;
    
    // Parallel score calculation
    for (int t = tid.x; t < 1024; t += 1024) {
        if (t <= pos) {
            float d = 0; device const half *mk = k_cache + t * kv_dim + kvh * headDim;
            for (int i = 0; i < headDim; i++) d += (float)mq[i] * (float)mk[i];
            s_mem[t] = d * scale;
        } else {
            s_mem[t] = -10000.0f; // -inf for softmax
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Softmax
    threadgroup float tg_mv;
    float l_mv = -10000.0f;
    for (int i = tid.x; i <= pos; i += 1024) if (s_mem[i] > l_mv) l_mv = s_mem[i];
    l_mv = simd_max(l_mv);
    
    threadgroup float scratch[32];
    if ((tid.x & 31) == 0) scratch[tid.x / 32] = l_mv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid.x < 32) {
        float m = (tid.x < 32) ? scratch[tid.x] : -10000.0f;
        m = simd_max(m);
        if (tid.x == 0) tg_mv = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mv = tg_mv;
    
    threadgroup float tg_se;
    float l_se = 0;
    for (int i = tid.x; i <= pos; i += 1024) {
        float e = exp(s_mem[i] - mv);
        s_mem[i] = e;
        l_se += e;
    }
    l_se = simd_sum(l_se);
    if ((tid.x & 31) == 0) scratch[tid.x / 32] = l_se;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid.x < 32) {
        float s = (tid.x < 32) ? scratch[tid.x] : 0.0f;
        s = simd_sum(s);
        if (tid.x == 0) tg_se = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float se = tg_se;
    
    // Result Accumulation
    if (tid.x < (uint)headDim) {
        float r = 0;
        for (int i = 0; i <= pos; i++) {
            r += (s_mem[i] / se) * (float)v_cache[i * kv_dim + kvh * headDim + tid.x];
        }
        output[h * headDim + tid.x] = safe_half(r);
    }
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
    output[qid] = safe_half(r);
}

kernel void add_f16(device const half *a [[ buffer(0) ]], device const half *b [[ buffer(1) ]], device half *out [[ buffer(2) ]], uint qid [[ thread_position_in_grid ]]) {
    out[qid] = safe_half((float)a[qid] + (float)b[qid]);
}

kernel void copy_f16(device const half *src [[ buffer(0) ]], device half *dst [[ buffer(1) ]], uint qid [[ thread_position_in_grid ]]) {
    dst[qid] = src[qid];
}

kernel void copy_f16_to_f32(device const half *src [[ buffer(0) ]], device float *dst [[ buffer(1) ]], uint qid [[ thread_position_in_grid ]]) {
    dst[qid] = (float)src[qid];
}

kernel void copy_f32_to_f16(device const float *src [[ buffer(0) ]], device half *dst [[ buffer(1) ]], uint qid [[ thread_position_in_grid ]]) {
    dst[qid] = safe_half(src[qid]);
}

// ==========================================
// FP32 Kernels for Higher Precision
// ==========================================

kernel void add_f32(device const float *a [[ buffer(0) ]], device const float *b [[ buffer(1) ]], device float *out [[ buffer(2) ]], uint qid [[ thread_position_in_grid ]]) {
    out[qid] = a[qid] + b[qid];
}

kernel void rmsnorm_f32(device const float *x [[ buffer(0) ]],
                      device float *out [[ buffer(1) ]],
                      device const half *w [[ buffer(2) ]],
                      constant float &eps [[ buffer(3) ]],
                      constant int &cols [[ buffer(4) ]],
                      uint tid [[ thread_index_in_threadgroup ]],
                      uint2 qid [[ thread_position_in_grid ]]) {
    threadgroup float s[1024]; 
    float sum = 0.0f;
    int row_offset = qid.y * cols;
    for (int i = tid; i < cols; i += 1024) {
        float val = x[row_offset + i];
        sum += val * val;
    }
    s[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { 
        float t = 0; 
        int active_threads = (cols < 1024) ? cols : 1024;
        for (int i = 0; i < active_threads; i++) t += s[i]; 
        s[0] = 1.0f / sqrt(t / (float)cols + eps); 
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = s[0];
    for (int i = tid; i < cols; i += 1024) {
        int idx = row_offset + i;
        out[idx] = x[idx] * scale * (float)w[i];
    }
}

kernel void rmsnorm_f32_to_f16(device const float *x [[ buffer(0) ]],
                      device half *out [[ buffer(1) ]],
                      device const half *w [[ buffer(2) ]],
                      constant float &eps [[ buffer(3) ]],
                      constant int &cols [[ buffer(4) ]],
                      uint tid [[ thread_index_in_threadgroup ]],
                      uint2 qid [[ thread_position_in_grid ]]) {
    threadgroup float s[1024]; 
    float sum = 0.0f;
    int row_offset = qid.y * cols;
    for (int i = tid; i < cols; i += 1024) {
        float val = x[row_offset + i];
        sum += val * val;
    }
    s[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { 
        float t = 0; 
        int active_threads = (cols < 1024) ? cols : 1024;
        for (int i = 0; i < active_threads; i++) t += s[i]; 
        s[0] = 1.0f / sqrt(t / (float)cols + eps); 
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = s[0];
    for (int i = tid; i < cols; i += 1024) {
        int idx = row_offset + i;
        out[idx] = safe_half(x[idx] * scale * (float)w[i]);
    }
}

kernel void add_f32_f16(device const float *a [[ buffer(0) ]], device const half *b [[ buffer(1) ]], device float *out [[ buffer(2) ]], uint qid [[ thread_position_in_grid ]]) {
    out[qid] = a[qid] + (float)b[qid];
}

kernel void linear_q4k_f32(device const uchar *weight [[ buffer(0) ]],
                         device const float *input [[ buffer(1) ]],
                         device float *output [[ buffer(2) ]],
                         constant int &dim_in [[ buffer(3) ]],
                         constant int &dim_out [[ buffer(4) ]],
                         uint3 tid [[ thread_position_in_threadgroup ]],
                         uint3 qid [[ thread_position_in_grid ]]) {
    uint row = qid.y; uint batch = qid.z;
    if (row >= (uint)dim_out) return; uint lane_id = tid.x;
    int num_blocks = dim_in / 256;
    device const uchar *row_ptr = weight + row * num_blocks * 144;
    device const float *in_ptr = input + batch * dim_in;
    float sum = 0;
    for (int i = (int)lane_id; i < num_blocks; i += 32) {
        device const uchar *block = row_ptr + i * 144;
        ushort d_bits = *(device const ushort*)(block);
        ushort dmin_bits = *(device const ushort*)(block + 2);
        float d = (float)as_type<half>(d_bits);
        float dmin = (float)as_type<half>(dmin_bits);
        
        device const uchar *scales = block + 4;
        device const uchar *qs = block + 16;
        uchar sc[8], m[8];
        for (int j = 0; j < 8; j++) {
            if (j < 4) {
                sc[j] = scales[j] & 63;
                m[j] = scales[j + 4] & 63;
            } else {
                sc[j] = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4);
                m[j] = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4);
            }
        }
        for (int j = 0; j < 8; j++) {
            float d_val = d * (float)sc[j], m_val = dmin * (float)m[j];
            int sub_offset = j * 32, qs_offset = j * 16;
            for (int k = 0; k < 16; k++) {
                uchar b = qs[qs_offset + k];
                float w0 = d_val * (float)(b & 0xF) - m_val;
                float w1 = d_val * (float)(b >> 4) - m_val;
                int idx0 = i * 256 + sub_offset + k;
                int idx1 = idx0 + 16;
                sum += w0 * in_ptr[idx0] + w1 * in_ptr[idx1];
            }
        }
    }
    sum = simd_sum(sum);
    if (lane_id == 0) output[batch * dim_out + row] = sum;
}

kernel void swiglu_f32(device const float *gate [[ buffer(0) ]], 
                     device const float *up [[ buffer(1) ]], 
                     device float *out [[ buffer(2) ]],
                     uint qid [[ thread_position_in_grid ]]) {
    float g = gate[qid]; 
    float val = up[qid] * (g / (1.0f + exp(-g)));
    out[qid] = val;
}

kernel void linear_f16_f32(device const half *weight [[ buffer(0) ]],
                         device const float *input [[ buffer(1) ]],
                         device float *output [[ buffer(2) ]],
                         constant int &dim_in [[ buffer(3) ]],
                         constant int &dim_out [[ buffer(4) ]],
                         uint3 tid [[ thread_position_in_threadgroup ]],
                         uint3 qid [[ thread_position_in_grid ]]) {
    uint row = qid.y; uint batch = qid.z;
    if (row >= (uint)dim_out) return; uint lane_id = tid.x;
    
    device const half4 *w4 = (device const half4 *)(weight + row * dim_in);
    device const float4 *i4 = (device const float4 *)(input + batch * dim_in);
    int n4 = dim_in / 4;
    
    float sum = 0;
    for (int i = (int)lane_id; i < n4; i += 32) {
        float4 v_w = float4(w4[i]);
        float4 v_i = i4[i];
        sum += dot(v_w.xy, v_i.xy) + dot(v_w.zw, v_i.zw);
    }
    sum = simd_sum(sum); 
    if (lane_id == 0) output[batch * dim_out + row] = sum;
}

// ============================================================================
// FP32 FFN Kernels for Small Models (SmolLM2, TinyLlama)
// These kernels maintain FP32 precision through FFN to prevent activation explosion
// ============================================================================

// Linear: FP16 weights, FP16 input → FP32 output (for Gate/Up projections)
kernel void linear_f16_to_f32(device const half *weight [[ buffer(0) ]],
                              device const half *input [[ buffer(1) ]],
                              device float *output [[ buffer(2) ]],
                              constant int &dim_in [[ buffer(3) ]],
                              constant int &dim_out [[ buffer(4) ]],
                              uint3 tid [[ thread_position_in_threadgroup ]],
                              uint3 qid [[ thread_position_in_grid ]]) {
    uint row = qid.y; uint batch = qid.z;
    if (row >= (uint)dim_out) return; uint lane_id = tid.x;
    
    device const half4 *w4 = (device const half4 *)(weight + row * dim_in);
    device const half4 *i4 = (device const half4 *)(input + batch * dim_in);
    int n4 = dim_in / 4;
    
    float sum = 0;
    for (int i = (int)lane_id; i < n4; i += 32) {
        float4 v_w = float4(w4[i]);
        float4 v_i = float4(i4[i]);
        sum += dot(v_w.xy, v_i.xy) + dot(v_w.zw, v_i.zw);
    }
    sum = simd_sum(sum); 
    if (lane_id == 0) output[batch * dim_out + row] = sum;
}

// Linear: FP16 weights, FP32 input → FP16 output (for Down projection)
kernel void linear_f32_to_f16(device const half *weight [[ buffer(0) ]],
                              device const float *input [[ buffer(1) ]],
                              device half *output [[ buffer(2) ]],
                              constant int &dim_in [[ buffer(3) ]],
                              constant int &dim_out [[ buffer(4) ]],
                              uint3 tid [[ thread_position_in_threadgroup ]],
                              uint3 qid [[ thread_position_in_grid ]]) {
    uint row = qid.y; uint batch = qid.z;
    if (row >= (uint)dim_out) return; uint lane_id = tid.x;
    
    device const half4 *w4 = (device const half4 *)(weight + row * dim_in);
    device const float4 *i4 = (device const float4 *)(input + batch * dim_in);
    int n4 = dim_in / 4;
    
    float sum = 0;
    for (int i = (int)lane_id; i < n4; i += 32) {
        float4 v_w = float4(w4[i]);
        float4 v_i = i4[i];
        sum += dot(v_w.xy, v_i.xy) + dot(v_w.zw, v_i.zw);
    }
    sum = simd_sum(sum); 
    if (lane_id == 0) output[batch * dim_out + row] = safe_half(sum);
}
