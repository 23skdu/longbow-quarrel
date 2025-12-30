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
        out[idx] = safe_half((float)x[idx] * scale * w[i]);
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
                         constant float &scale [[ buffer(5) ]],
                         uint3 tid [[ thread_position_in_threadgroup ]],
                         uint3 qid [[ thread_position_in_grid ]]) {
    uint row = qid.y; uint batch = qid.z;
    if (row >= (uint)dim_out) return; uint lane_id = tid.x;
    int num_blocks = (dim_in + 255) / 256;
    device const uchar *row_ptr = weight + row * num_blocks * 144;
    device const half *in_ptr = input + batch * dim_in;
    float sum = 0;
    for (int i = (int)lane_id; i < num_blocks; i += 32) {
        device const uchar *block = row_ptr + i * 144;
        
        // Direct Half Cast
        float d = (float)*(device const half*)(block);
        
        // Direct Half Cast for dmin
        float dmin = (float)*(device const half*)(block + 2);
        
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
            float d_val = d * scale * (float)sc[j], m_val = dmin * scale * (float)m[j];
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

kernel void rmsnorm_linear_q4k_f16(device const float *input [[ buffer(0) ]],
                                 device const float *norm_weight [[ buffer(1) ]],
                                 device const uchar *weight [[ buffer(2) ]],
                                 device half *output [[ buffer(3) ]],
                                 constant int &offRes [[ buffer(4) ]],
                                 constant float &eps [[ buffer(5) ]],
                                 constant int &dim_in [[ buffer(6) ]],
                                 constant int &dim_out [[ buffer(7) ]],
                                 constant float &scale [[ buffer(8) ]],
                                 uint3 tid [[ thread_position_in_threadgroup ]],
                                 uint3 qid [[ thread_position_in_grid ]]) {
    uint row = qid.y; uint batch = qid.z;
    if (row >= (uint)dim_out) return; uint lane_id = tid.x;

    // 1. Shared memory for RMSNorm scale
    threadgroup float s_rms[1024];
    float sum_sq = 0.0f;
    device const float *in_ptr = input + batch * dim_in;
    for (int i = tid.x; i < dim_in; i += 1024) {
        float val = (float)in_ptr[i];
        sum_sq += val * val;
    }
    s_rms[tid.x] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tid.x == 0) {
        float t = 0;
        int active_threads = (dim_in < 1024) ? dim_in : 1024;
        for (int i = 0; i < active_threads; i++) t += s_rms[i];
        s_rms[0] = 1.0f / sqrt(t / (float)dim_in + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rms_scale = s_rms[0];

    // 2. Perform Linear Q4K with on-the-fly normalized input
    int num_blocks = (dim_in + 255) / 256;
    device const uchar *row_ptr = weight + row * num_blocks * 144;
    float sum = 0;
    
    for (int i = (int)tid.x; i < num_blocks; i += 1024) {
        device const uchar *block = row_ptr + i * 144;
        float d = (float)*(device const half*)(block);
        float dmin = (float)*(device const half*)(block + 2);
        
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
            float d_val = d * scale * (float)sc[j], m_val = dmin * scale * (float)m[j];
            int sub_offset = j * 32, qs_offset = j * 16;
            for (int k = 0; k < 16; k++) {
                uchar b = qs[qs_offset + k];
                float w0 = d_val * (float)(b & 0xF) - m_val;
                float w1 = d_val * (float)(b >> 4) - m_val;
                int idx0 = i * 256 + sub_offset + k;
                int idx1 = idx0 + 16;
                
                // Normalized input (now reading float weight)
                float in0 = (float)in_ptr[idx0] * rms_scale * norm_weight[idx0];
                float in1 = (float)in_ptr[idx1] * rms_scale * norm_weight[idx1];
                
                sum += w0 * in0 + w1 * in1;
            }
        }
    }
    sum = simd_sum(sum);
    if ((tid.x % 32) == 0) {
        s_rms[tid.x / 32] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tid.x == 0) {
        float final_sum = 0;
        for (int i = 0; i < 32; i++) final_sum += s_rms[i];
        output[batch * dim_out + row + offRes/2] = safe_half(final_sum);
    }
}

kernel void swiglu_linear_q4k_f16(device const half *gate_input [[ buffer(0) ]],
                                device const half *up_input [[ buffer(1) ]],
                                device const uchar *weight [[ buffer(2) ]],
                                device half *output [[ buffer(3) ]],
                                constant int &offRes [[ buffer(4) ]],
                                constant int &dim_in [[ buffer(5) ]],
                                constant int &dim_out [[ buffer(6) ]],
                                constant float &scale [[ buffer(7) ]],
                                uint3 tid [[ thread_position_in_threadgroup ]],
                                uint3 qid [[ thread_position_in_grid ]]) {
    uint row = qid.y; uint batch = qid.z;
    if (row >= (uint)dim_out) return; uint lane_id = tid.x;

    int num_blocks = (dim_in + 255) / 256;
    device const uchar *row_ptr = weight + row * num_blocks * 144;
    device const half *gate_ptr = gate_input + batch * dim_in;
    device const half *up_ptr = up_input + batch * dim_in;
    float sum = 0;
    
    for (int i = (int)lane_id; i < num_blocks; i += 32) {
        device const uchar *block = row_ptr + i * 144;
        float d = (float)*(device const half*)(block);
        float dmin = (float)*(device const half*)(block + 2);
        
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
            float d_val = d * scale * (float)sc[j], m_val = dmin * scale * (float)m[j];
            int sub_offset = j * 32, qs_offset = j * 16;
            for (int k = 0; k < 16; k++) {
                uchar b = qs[qs_offset + k];
                float w0 = d_val * (float)(b & 0xF) - m_val;
                float w1 = d_val * (float)(b >> 4) - m_val;
                int idx0 = i * 256 + sub_offset + k;
                int idx1 = idx0 + 16;
                
                // On-the-fly SwiGLU for idx0
                float g0 = (float)gate_ptr[idx0];
                float u0 = (float)up_ptr[idx0];
                float in0 = u0 * (g0 / (1.0f + exp(-g0)));
                
                // On-the-fly SwiGLU for idx1
                float g1 = (float)gate_ptr[idx1];
                float u1 = (float)up_ptr[idx1];
                float in1 = u1 * (g1 / (1.0f + exp(-g1)));
                
                sum += w0 * in0 + w1 * in1;
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
                         constant float &scale [[ buffer(5) ]],
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
            float s = d * scale * (float)sc[l];
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
                    constant int &numHeads [[ buffer(4) ]],
                    uint2 gid [[ thread_position_in_grid ]]) {
    // gid.x = pair index within specific token (0 .. numHeads * headDim/2)
    // gid.y = token index (0 .. batch_size-1)
    
    int h = gid.x / (headDim / 2);
    int i = gid.x % (headDim / 2);
    int p = pos + gid.y; // Absolute position in sequence
    
    float freq = (float)p * pow(ropeTheta, -2.0f * (float)i / (float)headDim);
    float ct = cos(freq);
    float st = sin(freq);
    
    // Layout: [Tokens, Heads, HeadDim]
    device half *token_ptr = x + gid.y * numHeads * headDim;
    device half *q_ptr = token_ptr + h * headDim;
    
    // Standard Llama/Mistral uses Half-Half rotation (0 with 64, 1 with 65...)
    int idx0 = i;
    int idx1 = i + headDim/2;
    
    float x0 = (float)q_ptr[idx0];
    float x1 = (float)q_ptr[idx1];
    
    q_ptr[idx0] = (half)(x0 * ct - x1 * st);
    q_ptr[idx1] = (half)(x0 * st + x1 * ct);
}

kernel void embedding_f16(device const half *weight [[ buffer(0) ]], device half *output [[ buffer(1) ]], constant int &idx [[ buffer(2) ]], constant int &cols [[ buffer(3) ]], uint qid [[ thread_position_in_grid ]]) {
    output[qid] = weight[idx * (uint)cols + qid];
}

// Embedding Lookup: Q4_K weights → FP16 output
kernel void embedding_q4k_f16(device const uchar *weight [[ buffer(0) ]],
                             device half *output [[ buffer(1) ]],
                             constant int &idx [[ buffer(2) ]],
                             constant int &cols [[ buffer(3) ]],
                             constant float &scale [[ buffer(4) ]],
                             uint tid [[ thread_index_in_threadgroup ]]) {
    int num_blocks = (cols + 255) / 256;
    device const uchar *row_ptr = weight + (uint)idx * num_blocks * 144;
    
    // Each thread handles a portion of the 256-elements blocks
    // If we want high performance, we'd distribute blocks across threads.
    // For now, simpler: each thread handles one or more blocks if needed.
    // But cols is usually small-ish (4096).
    // 4096 / 256 = 16 blocks.
    
    for (int i = tid; i < num_blocks; i += 1024) {
        device const uchar *block = row_ptr + i * 144;
        ushort d_bits = *(device const ushort*)(block);
        ushort dmin_bits = *(device const ushort*)(block + 2);
        
        // Manual FP16->FP32 conversion with subnormal handling
        uint d_sign = (d_bits & 0x8000u) << 16;
        uint d_exp_raw = (d_bits & 0x7C00u) >> 10;
        uint d_mant = d_bits & 0x03FFu;
        
        float d;
        if (d_exp_raw == 0) {
            if (d_mant == 0) {
                d = 0.0f;
            } else {
                uint shift = 0;
                uint test_mant = d_mant;
                while ((test_mant & 0x0400) == 0) {
                    test_mant <<= 1;
                    shift++;
                }
                uint normalized_mant = (test_mant & 0x03FF) << 13;
                uint d_exp = (113 - shift) << 23;
                d = as_type<float>(d_sign | d_exp | normalized_mant);
            }
        } else {
            uint d_exp = (d_exp_raw + (127 - 15)) << 23;
            d = as_type<float>(d_sign | d_exp | (d_mant << 13));
        }
        
        // Same for dmin
        uint dmin_sign = (dmin_bits & 0x8000u) << 16;
        uint dmin_exp_raw = (dmin_bits & 0x7C00u) >> 10;
        uint dmin_mant = dmin_bits & 0x03FFu;
        
        float dmin;
        if (dmin_exp_raw == 0) {
            if (dmin_mant == 0) {
                dmin = 0.0f;
            } else {
                uint shift = 0;
                uint test_mant = dmin_mant;
                while ((test_mant & 0x0400) == 0) {
                    test_mant <<= 1;
                    shift++;
                }
                uint normalized_mant = (test_mant & 0x03FF) << 13;
                uint dmin_exp = (113 - shift) << 23;
                dmin = as_type<float>(dmin_sign | dmin_exp | normalized_mant);
            }
        } else {
            uint dmin_exp = (dmin_exp_raw + (127 - 15)) << 23;
            dmin = as_type<float>(dmin_sign | dmin_exp | (dmin_mant << 13));
        }
        
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
            float d_val = d * scale * (float)sc[j], m_val = dmin * scale * (float)m[j];
            int sub_offset = j * 32, qs_offset = j * 16;
            for (int k = 0; k < 16; k++) {
                uchar b = qs[qs_offset + k];
                output[i * 256 + sub_offset + k]      = (half)(d_val * (float)(b & 0xF) - m_val);
                output[i * 256 + sub_offset + k + 16] = (half)(d_val * (float)(b >> 4) - m_val);
            }
        }
    }
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
                         constant int &window_size [[ buffer(8) ]],
                         uint qid [[ thread_position_in_grid ]]) {
    uint h = qid / 32, lane = qid % 32; if (h >= (uint)num_heads) return;
    uint kvh = h / (num_heads / kv_heads), kv_dim = kv_heads * headDim;
    float scale = 1.0f / sqrt((float)headDim);
    device const half *mq = q + h * headDim;

    // SLIDING WINDOW: Determine window bounds
    int start = (window_size > 0 && pos >= window_size) ? (pos - window_size + 1) : 0;

    // Initialize all scores to -inf (or masked value) before the window
    if (lane == 0) {
        for (int t = 0; t < start; t++) {
            scores[h * stride + t] = -10000.0f;
        }
    }

    for (int t = start; t <= pos; t++) {
        float d = 0; 
        // Use rolling buffer index
        int cache_idx = (window_size > 0) ? (t % window_size) : t;
        device const half *mk = k_cache + cache_idx * kv_dim + kvh * headDim;
        for (int i = (int)lane; i < headDim; i += 32) d += (float)mq[i] * (float)mk[i];
        d = simd_sum(d); 
        if (lane == 0) {
            scores[h * stride + t] = d * scale;
        }
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
                        constant int &window_size [[ buffer(8) ]],
                        uint3 tid [[ thread_position_in_threadgroup ]],
                        uint3 nthreads [[ threads_per_threadgroup ]],
                        uint3 group_id [[ threadgroup_position_in_grid ]]) {
    uint h = group_id.x; if (h >= (uint)num_heads) return;
    uint kvh = h / (num_heads / kv_heads);
    uint kv_dim = kv_heads * headDim;
    float scale = 1.0f / sqrt((float)headDim);

    // SLIDING WINDOW: Determine window bounds
    int start = (window_size > 0 && pos >= window_size) ? (pos - window_size + 1) : 0;

    // Limit fused path to 4000 tokens to stay within 32KB
    threadgroup float s_mem[4000]; 
    if (pos >= 4000) return;

    device const half *mq = q + h * headDim;
    uint nt = nthreads.x;
    
    // 1. Parallel score calculation
    for (int t = tid.x; t < 4000; t += nt) {
        if (t >= start && t <= pos) {
            float d = 0; 
            int cache_idx = (window_size > 0) ? (t % window_size) : t;
            device const half *mk = k_cache + cache_idx * kv_dim + kvh * headDim;
            for (int i = 0; i < headDim; i++) d += (float)mq[i] * (float)mk[i];
            s_mem[t] = d * scale;
        } else {
            s_mem[t] = -10000.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Softmax Max Reduction
    threadgroup float tg_mv;
    float l_mv = -10000.0f;
    for (int i = tid.x; i <= pos; i += nt) {
        if (s_mem[i] > l_mv) l_mv = s_mem[i];
    }
    l_mv = simd_max(l_mv);
    
    threadgroup float scratch[32]; 
    if ((tid.x & 31) == 0) scratch[tid.x / 32] = l_mv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tid.x < 32) {
        uint n_subgroups = (nt + 31) / 32;
        float m = (tid.x < n_subgroups) ? scratch[tid.x] : -10000.0f;
        m = simd_max(m);
        if (tid.x == 0) tg_mv = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mv = tg_mv;
    
    // 3. Softmax Sum Reduction
    threadgroup float tg_se;
    float l_se = 0;
    for (int i = tid.x; i <= pos; i += nt) {
        float e = exp(s_mem[i] - mv);
        s_mem[i] = e;
        l_se += e;
    }
    l_se = simd_sum(l_se);
    if ((tid.x & 31) == 0) scratch[tid.x / 32] = l_se;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tid.x < 32) {
        uint n_subgroups = (nt + 31) / 32;
        float s = (tid.x < n_subgroups) ? scratch[tid.x] : 0.0f;
        s = simd_sum(s);
        if (tid.x == 0) tg_se = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float se = tg_se;
    if (se == 0) se = 1e-6f;
    
    // 4. Result Accumulation
    if (tid.x < (uint)headDim) {
        float r = 0;
        for (int t = start; t <= pos; t++) {
            int cache_idx = (window_size > 0) ? (t % window_size) : t;
            r += (s_mem[t] / se) * (float)v_cache[cache_idx * kv_dim + kvh * headDim + tid.x];
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
                         constant int &window_size [[ buffer(8) ]],
                         uint qid [[ thread_position_in_grid ]]) {
    uint h = qid / headDim, idx = qid % headDim; if (h >= (uint)num_heads) return;

    uint kvh = h / (num_heads / kv_heads), kv_dim = kv_heads * headDim;
    float r = 0; 
    
    // SLIDING WINDOW: Determine window bounds
    int start = (window_size > 0 && pos >= window_size) ? (pos - window_size + 1) : 0;

    for (int t = start; t <= pos; t++) {
        int cache_idx = (window_size > 0) ? (t % window_size) : t;
        r += scores[h * stride + t] * (float)v_cache[cache_idx * kv_dim + kvh * headDim + idx];
    }
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
        out[idx] = x[idx] * scale * w[i];
    }
}

kernel void rmsnorm_f32_to_f16_v4(device const float *x [[ buffer(0) ]],
                      device half *out [[ buffer(1) ]],
                      device const float *w [[ buffer(2) ]],
                      constant float &eps [[ buffer(3) ]],
                      constant int &cols [[ buffer(4) ]],
                      uint3 tid [[ thread_position_in_threadgroup ]],
                      uint3 qid [[ thread_position_in_grid ]]) {
    threadgroup float s[1024]; 
    float sum = 0.0f;
    int row_offset = qid.y * cols;
    // Sum squares
    for (int i = tid.x; i < cols; i += 1024) {
        float val = x[row_offset + i];
        sum += val * val;
    }
    s[tid.x] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce
    if (tid.x == 0) { 
        float t = 0; 
        int active_threads = (cols < 1024) ? cols : 1024;
        for (int i = 0; i < active_threads; i++) t += s[i]; 
        s[0] = 1.0f / sqrt(t / (float)cols + eps); 
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float scale = s[0];
    
    // Write Output
    for (int i = tid.x; i < cols; i += 1024) {
        int idx = row_offset + i;
        float val_x = x[idx];
        float val_w = w[i]; 
        // Correct logic
        out[idx] = safe_half(val_x * scale * val_w);
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
                         constant float &scale [[ buffer(5) ]],
                         uint3 tid [[ thread_position_in_threadgroup ]],
                         uint3 qid [[ thread_position_in_grid ]]) {
    uint row = qid.y; uint batch = qid.z;
    if (row >= (uint)dim_out) return; uint lane_id = tid.x;
    int num_blocks = (dim_in + 255) / 256;
    device const uchar *row_ptr = weight + row * num_blocks * 144;
    device const float *in_ptr = input + batch * dim_in;
    float sum = 0;
    for (int i = (int)lane_id; i < num_blocks; i += 32) {
        device const uchar *block = row_ptr + i * 144;
        ushort d_bits = *(device const ushort*)(block);
        ushort dmin_bits = *(device const ushort*)(block + 2);
        
        // Manual FP16->FP32 conversion with subnormal handling (same as F16 kernel)
        uint d_sign = (d_bits & 0x8000u) << 16;
        uint d_exp_raw = (d_bits & 0x7C00u) >> 10;
        uint d_mant = d_bits & 0x03FFu;
        
        float d;
        if (d_exp_raw == 0) {
            if (d_mant == 0) {
                d = 0.0f;
            } else {
                uint shift = 0;
                uint test_mant = d_mant;
                while ((test_mant & 0x0400) == 0) {
                    test_mant <<= 1;
                    shift++;
                }
                // After normalization, bit 10 is set (the implicit leading 1)
                // For FP32, we need to remove this bit and place remaining 10 bits in upper mantissa
                uint normalized_mant = (test_mant & 0x03FF) << 13;  // Bits 0-9 -> bits 13-22
                uint d_exp = (113 - shift) << 23;
                d = as_type<float>(d_sign | d_exp | normalized_mant);
            }
        } else {
            uint d_exp = (d_exp_raw + (127 - 15)) << 23;
            d = as_type<float>(d_sign | d_exp | (d_mant << 13));
        }
        
        // Same for dmin
        uint dmin_sign = (dmin_bits & 0x8000u) << 16;
        uint dmin_exp_raw = (dmin_bits & 0x7C00u) >> 10;
        uint dmin_mant = dmin_bits & 0x03FFu;
        
        float dmin;
        if (dmin_exp_raw == 0) {
            if (dmin_mant == 0) {
                dmin = 0.0f;
            } else {
                uint shift = 0;
                uint test_mant = dmin_mant;
                while ((test_mant & 0x0400) == 0) {
                    test_mant <<= 1;
                    shift++;
                }
                uint normalized_mant = (test_mant & 0x03FF) << 13;
                uint dmin_exp = (113 - shift) << 23;
                dmin = as_type<float>(dmin_sign | dmin_exp | normalized_mant);
            }
        } else {
            uint dmin_exp = (dmin_exp_raw + (127 - 15)) << 23;
            dmin = as_type<float>(dmin_sign | dmin_exp | (dmin_mant << 13));
        }


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
            float d_val = d * scale * (float)sc[j], m_val = dmin * scale * (float)m[j];
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

// Weights: F16, Input: F16, Output: F32
kernel void linear_f16_in_f16_out_f32(device const half *weight [[ buffer(0) ]],
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

kernel void store_kv_f16(device const half *k [[ buffer(0) ]],
                        device const half *v [[ buffer(1) ]],
                        device half *k_cache [[ buffer(2) ]],
                        device half *v_cache [[ buffer(3) ]],
                        constant int &pos [[ buffer(4) ]],
                        constant int &kv_dim [[ buffer(5) ]],
                        constant int &window_size [[ buffer(6) ]],
                        uint tid [[ thread_position_in_grid ]]) {
    if (tid >= (uint)kv_dim) return;
    
    // SLIDING WINDOW: Use rolling buffer indexing
    // If window_size == 0, use full sequence (backward compat)
    int cache_pos = (window_size > 0) ? (pos % window_size) : pos;
    int offset = cache_pos * kv_dim + tid;
    
    k_cache[offset] = k[tid];
    v_cache[offset] = v[tid];
}
