#include <metal_stdlib>
using namespace metal;

static inline float simd_sum(float val) {
    for (uint offset = 16; offset > 0; offset /= 2) val += simd_shuffle_down(val, offset);
    return simd_broadcast(val, 0);
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

half safe_half(float x) {
    return (half)clamp(x, -65504.0f, 65504.0f);
}

// Native casting is used instead of manual conversion

kernel void scale_f16(device const half *x [[ buffer(0) ]], constant ushort &scale [[ buffer(1) ]], device half *out [[ buffer(2) ]], uint qid [[ thread_position_in_grid ]]) {
    half s = as_type<half>(scale);
    out[qid] = (half)((float)x[qid] * (float)s);
}

kernel void swiglu_f16(device const half *gate [[ buffer(0) ]], 
                     device const half *up [[ buffer(1) ]], 
                     device half *out [[ buffer(2) ]],
                     uint qid [[ thread_position_in_grid ]]) {
    float g = (float)gate[qid]; 
    float val = (float)up[qid] * (g / (1.0f + exp(-g)));
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
        out[idx] = (half)((float)x[idx] * scale * (float)w[i]);
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
        float4 v_w = (float4)w4[i]; float4 v_i = (float4)i4[i];
        sum += dot(v_w.xy, v_i.xy) + dot(v_w.zw, v_i.zw);
    }
    sum = simd_sum(sum); 
    if (lane_id == 0) output[batch * dim_out + row] = (half)sum;
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
        float d = fp16_to_fp32(d_bits);
        float dmin = fp16_to_fp32(dmin_bits);
        
        // Debug: Print first block's scale for first row
        if (row == 0 && i == 0 && batch == 0 && lane_id == 0) {
            // Can't printf from Metal kernel, but we can write to a debug buffer
            // For now, just ensure values are reasonable
        }
        
        device const uchar *scales = block + 4;
        device const uchar *qs = block + 16;
        uchar sc[8], m[8];
        // Extract scales and mins using llama.cpp's get_scale_min_k4 logic
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
                int idx = i * 256 + sub_offset + k * 2;
                sum += w0 * (float)in_ptr[idx] + w1 * (float)in_ptr[idx + 1];
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

kernel void embedding_f16(device const half *weight [[ buffer(0) ]], device half *output [[ buffer(1) ]], constant int &idx [[ buffer(2) ]], constant int &cols [[ buffer(3) ]], uint qid [[ thread_position_in_grid ]]) {
    output[qid] = weight[idx * (uint)cols + qid];
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

kernel void add_f16(device const half *a [[ buffer(0) ]], device const half *b [[ buffer(1) ]], device half *out [[ buffer(2) ]], uint qid [[ thread_position_in_grid ]]) {
    out[qid] = a[qid] + b[qid];
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
        float d = fp16_to_fp32(d_bits);
        float dmin = fp16_to_fp32(dmin_bits);
        
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
                int idx = i * 256 + sub_offset + k * 2;
                sum += w0 * in_ptr[idx] + w1 * in_ptr[idx + 1];
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
