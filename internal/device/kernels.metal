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
                     device const half *up [[ buffer(1)]], 
                     device half *out [[ buffer(2) ]],
                     uint qid [[ thread_position_in_grid ]]) {
    float g = (float)gate[qid]; 
    float u = (float)up[qid];
    
    // SwiGLU: up * silu(gate) where silu(x) = x * sigmoid(x)
    // Clamp gate to prevent extreme sigmoid values that cause NaN
    // This matches the F32 SwiGLU kernel behavior
    float g_clamped = clamp(g, -10.0f, 10.0f);
    float sigmoid_g = g_clamped / (1.0f + exp(-g_clamped));
    float val = u * sigmoid_g;
    
    // Clamp output to prevent activation explosion
    // Use FP16 clamp range (65504) to stay within safe bounds
    out[qid] = clamp(val, -65504.0f, 65504.0f);
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
    
    float sum = 0;
    for (int i = (int)lane_id; i < n4; i += 32) {
        float4 v_w = float4(w4[i]);
        float4 v_i = float4(i4[i]);
        sum += dot(v_w.xy, v_i.xy) + dot(v_w.zw, v_i.zw);
    }
    // Handle remainder
    for (int i = n4 * 4 + (int)lane_id; i < dim_in; i += 32) {
        sum += (float)weight[row * dim_in + i] * (float)input[batch * dim_in + i];
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
        // Convert F16 to F32 for proper calculation
        float d = (float)*(device const half*)(block + 208);
        
        for (int l = 0; l < 16; l++) {
            float s = d * scale * (float)sc[l];
            int group_off = l * 16;
            for (int k = 0; k < 16; k += 2) {
                int idx = group_off + k;
                uchar b = ql[idx / 2];
                uchar h0 = (qh[idx / 4] >> ((idx % 4) * 2)) & 3;
                uchar h1 = (qh[(idx+1) / 4] >> (((idx+1) % 4) * 2)) & 3;
                
                sum += s * (float)((int8_t)((h0 << 4) | (b & 0xF)) - 32) * (float)in_ptr[i * 256 + idx];
                sum += s * (float)((int8_t)((h1 << 4) | (b >> 4)) - 32) * (float)in_ptr[i * 256 + idx + 1];
            }
        }
    }
    sum = simd_sum(sum); 
    if (lane_id == 0) output[batch * dim_out + row] = safe_half(sum);
}

kernel void linear_q6k_f32(device const uchar *weight [[ buffer(0) ]],
                         device const float *input [[ buffer(1) ]],
                         device float *output [[ buffer(2) ]],
                         constant int &dim_in [[ buffer(3) ]],
                         constant int &dim_out [[ buffer(4) ]],
                         constant float &scale [[ buffer(5) ]],
                         uint3 tid [[ thread_position_in_threadgroup ]],
                         uint3 qid [[ thread_position_in_grid ]]) {
    uint row = qid.y; uint batch = qid.z;
    if (row >= (uint)dim_out) return; uint lane_id = tid.x;
    int num_blocks = dim_in / 256;
    device const uchar *row_ptr = weight + row * num_blocks * 210;
    device const float *in_ptr = input + batch * dim_in;
    float sum = 0;
    
    for (int i = (int)lane_id; i < num_blocks; i += 32) {
        device const uchar *block = row_ptr + i * 210;
        device const uchar *ql = block;
        device const uchar *qh = block + 128;
        device const char  *sc = (device const char *)(block + 192);
        ushort d_bits = *(device const ushort*)(block + 208);
        
        // Simple cast
        float d = (float)(*(device const half*)(block + 208));
        if (isinf(d) || isnan(d)) d = 0.0f;
        
        for (int l = 0; l < 16; l++) {
            float s = d * scale * (float)sc[l];
            int group_off = l * 16;
            for (int k = 0; k < 16; k += 2) {
                int idx = group_off + k;
                uchar b = ql[idx / 2];
                uchar h0 = (qh[idx / 4] >> ((idx % 4) * 2)) & 3;
                uchar h1 = (qh[(idx+1) / 4] >> (((idx+1) % 4) * 2)) & 3;
                
                sum += s * (float)((int8_t)((h0 << 4) | (b & 0xF)) - 32) * input[batch * dim_in + i * 256 + idx];
                sum += s * (float)((int8_t)((h1 << 4) | (b >> 4)) - 32) * input[batch * dim_in + i * 256 + idx + 1];
        }
    }
    }
    sum = simd_sum(sum);
    if (lane_id == 0) output[batch * dim_out + row] = sum;
}

kernel void linear_q6k_f32_f16(device const uchar *weight [[ buffer(0) ]],
                          device const float *input [[ buffer(1) ]],
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
    device const float *in_ptr = input + batch * dim_in;
    float sum = 0;
    
    for (int i = (int)lane_id; i < num_blocks; i += 32) {
        device const uchar *block = row_ptr + i * 210;
        device const uchar *ql = block;
        device const uchar *qh = block + 128;
        device const char  *sc = (device const char *)(block + 192);
        ushort d_bits = *(device const ushort*)(block + 208);
        
        float d = (float)(*(device const half*)(block + 208));
        if (isinf(d) || isnan(d)) d = 0.0f;
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
                sum += w0 * in_ptr[i * 256 + idx] + w1 * in_ptr[i * 256 + idx + 1];
            }
        }
    }
    if (lane_id == 0) output[batch * dim_out + row] = safe_half(sum);
}


// Q4_0: Block size 32. 18 bytes.
// Structure: half d; uchar qs[16];
kernel void linear_q4_0_f16(device const uchar *weight [[ buffer(0) ]],
                           device const half *input [[ buffer(1) ]],
                           device half *output [[ buffer(2) ]],
                           constant int &dim_in [[ buffer(3) ]],
                           constant int &dim_out [[ buffer(4) ]],
                           constant float &scale [[ buffer(5) ]],
                           uint3 tid [[ thread_position_in_threadgroup ]],
                           uint3 qid [[ thread_position_in_grid ]]) {
    uint row = qid.y; uint batch = qid.z;
    if (row >= (uint)dim_out) return; uint lane_id = tid.x;
    int num_blocks = dim_in / 32;
    device const uchar *row_ptr = weight + row * num_blocks * 18;
    device const half *in_ptr = input + batch * dim_in;
    float sum = 0;
    
    for (int i = (int)lane_id; i < num_blocks; i += 32) {
        device const uchar *block = row_ptr + i * 18;
        half d = *(device const half*)block;
        device const uchar *qs = block + 2;
        
        for (int j = 0; j < 16; j++) {
            uchar b = qs[j];
            float v0 = ((float)(b & 0xF) - 8.0f);
            float v1 = ((float)(b >> 4) - 8.0f);
            sum += (float)d * v0 * (float)in_ptr[i * 32 + j * 2];
            sum += (float)d * v1 * (float)in_ptr[i * 32 + j * 2 + 1];
        }
    }
    sum = simd_sum(sum);
    if (lane_id == 0) output[batch * dim_out + row] = (half)(sum * scale);
}

kernel void linear_q4_0_f32(device const uchar *weight [[ buffer(0) ]],
                           device const half *input [[ buffer(1) ]],
                           device float *output [[ buffer(2) ]],
                           constant int &dim_in [[ buffer(3) ]],
                           constant int &dim_out [[ buffer(4) ]],
                           constant float &scale [[ buffer(5) ]],
                           uint3 tid [[ thread_position_in_threadgroup ]],
                           uint3 qid [[ thread_position_in_grid ]]) {
    uint row = qid.y; uint batch = qid.z;
    if (row >= (uint)dim_out) return; uint lane_id = tid.x;
    int num_blocks = dim_in / 32;
    device const uchar *row_ptr = weight + row * num_blocks * 18;
    device const half *in_ptr = input + batch * dim_in;
    float sum = 0;
    
    for (int i = (int)lane_id; i < num_blocks; i += 32) {
        device const uchar *block = row_ptr + i * 18;
        half d = *(device const half*)block;
        device const uchar *qs = block + 2;
        
        for (int j = 0; j < 16; j++) {
            uchar b = qs[j];
            float v0 = ((float)(b & 0xF) - 8.0f);
            float v1 = ((float)(b >> 4) - 8.0f);
            sum += (float)d * v0 * (float)in_ptr[i * 32 + j * 2];
            sum += (float)d * v1 * (float)in_ptr[i * 32 + j * 2 + 1];
        }
    }
    sum = simd_sum(sum);
    if (lane_id == 0) output[batch * dim_out + row] = sum * scale;
}

kernel void embedding_q4_0_f16(device const uchar *weight [[ buffer(0) ]],
                              device half *output [[ buffer(1) ]],
                              constant int &idx [[ buffer(2) ]],
                              constant int &dim [[ buffer(3) ]],
                              uint qid [[ thread_position_in_grid ]]) {
    if (qid >= (uint)dim) return;
    int num_blocks = dim / 32;
    // Row offset
    device const uchar *row_ptr = weight + idx * num_blocks * 18;
    
    int i = qid / 32; // block index
    int lane = qid % 32; // element in block
    
    device const uchar *block = row_ptr + i * 18;
    half d = *(device const half*)block;
    device const uchar *qs = block + 2;
    
    // qs stores 32 elements. 2 per byte.
    // lane 0 -> byte 0 low. lane 1 -> byte 0 high.
    int byte_idx = lane / 2;
    uchar b = qs[byte_idx];
    float v;
    if ((lane % 2) == 0) {
        v = ((float)(b & 0xF) - 8.0f);
    } else {
        v = ((float)(b >> 4) - 8.0f);
    }
    output[qid] = (half)(v * (float)d);
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
    int lane = gid.x % (headDim / 2);
    
    if (lane < headDim/2) {
        // Neox Rotation (Half)
        int half_dim = headDim / 2;
        
        int idx0 = lane;
        int idx1 = lane + half_dim;
        
        // Calculate theta
        // theta = (pos + token_idx) * base^(-2*i/d)
        // i = idx0
        float theta_i = (float)(pos + (int)gid.y) * pow(ropeTheta, -2.0f * (float)idx0 / (float)headDim);
        float sin_theta = sin(theta_i);
        float cos_theta = cos(theta_i);

        device half *dx = x + gid.y * numHeads * headDim + h * headDim;
        float x0 = (float)dx[idx0];
        float x1 = (float)dx[idx1];

        dx[idx0] = (half)(x0 * cos_theta - x1 * sin_theta);
        dx[idx1] = (half)(x0 * sin_theta + x1 * cos_theta);
    }
}

kernel void embedding_f16(device const half *weight [[ buffer(0) ]], device half *output [[ buffer(1) ]], constant int &idx [[ buffer(2) ]], constant int &cols [[ buffer(3) ]], uint qid [[ thread_position_in_grid ]]) {
    output[qid] = weight[idx * (uint)cols + qid];
}

// Debug kernel to dump frequencies
kernel void debug_rope_freq(device float *output [[ buffer(0) ]],
                          constant int &headDim [[ buffer(1) ]],
                          constant float &ropeTheta [[ buffer(2) ]],
                          constant int &pos [[ buffer(3) ]],
                          uint tid [[ thread_position_in_grid ]]) {
    if (tid >= (uint)headDim/2) return;
    int i = tid;
    float p = (float)pos;
    float freq = p * pow(ropeTheta, -2.0f * (float)i / (float)headDim);
    output[i] = freq;
}

// Debug kernel to check dot product and simd_sum
kernel void debug_dot(device const half *a [[ buffer(0) ]],
                     device const half *b [[ buffer(1) ]],
                     device float *output [[ buffer(2) ]],
                     constant int &dim [[ buffer(3) ]],
                     uint tid [[ thread_position_in_threadgroup ]]) {
    // Single threadgroup of 32 threads
    float d = 0;
    for (int i = tid; i < dim; i += 32) {
        d += (float)a[i] * (float)b[i];
    }
    d = simd_sum(d);
    if (tid == 0) {
        output[0] = d;
    }
}

// Optimized Embedding Lookup: Q4_K weights → FP16 output with vectorization
kernel void embedding_q4k_f16_optimized(device const uchar *weight [[ buffer(0) ]],
                                        device half *output [[ buffer(1) ]],
                                        constant int &idx [[ buffer(2) ]],
                                        constant int &cols [[ buffer(3) ]],
                                        constant float &scale [[ buffer(4) ]],
                                        uint tid [[ thread_position_in_grid ]]) {
    if (tid >= (uint)cols) return;
    
    int num_blocks = (cols + 255) / 256;
    int block_idx = tid / 256;
    int lane_in_block = tid % 256;
    
    device const uchar *row_ptr = weight + (uint)idx * num_blocks * 144;
    device const uchar *block = row_ptr + block_idx * 144;
    
    // Pre-compute scale factors once per block
    float d = fp16_to_fp32(*(device const ushort*)(block));
    float dmin = fp16_to_fp32(*(device const ushort*)(block + 2));
    
    device const uchar *scales = block + 4;
    device const uchar *qs = block + 16;
    
    // Unpack scales for this lane
    int group = lane_in_block / 32;
    int sub_lane = lane_in_block % 32;
    int qs_idx = sub_lane / 2;
    
    // Extract scales and mins for this group
    uchar sc, m;
    if (group < 4) {
        sc = scales[group] & 63;
        m = scales[group + 4] & 63;
    } else {
        sc = (scales[group+4] & 0xF) | ((scales[group-4] >> 6) << 4);
        m = (scales[group+4] >> 4) | ((scales[group] >> 6) << 4);
    }
    
    float d_val = d * scale * (float)sc;
    float m_val = dmin * scale * (float)m;
    
    // Extract 4-bit quantized value
    uchar b = qs[16 * group + qs_idx];
    uchar q4 = (sub_lane % 2 == 0) ? (b & 0xF) : (b >> 4);
    
    // Final dequantization
    output[tid] = (half)(d_val * (float)q4 - m_val);
}

// Legacy version kept for compatibility
kernel void embedding_q4k_f16(device const uchar *weight [[ buffer(0) ]],
                             device half *output [[ buffer(1) ]],
                             constant int &idx [[ buffer(2) ]],
                             constant int &cols [[ buffer(3) ]],
                             constant float &scale [[ buffer(4) ]],
                             uint tid [[ thread_index_in_threadgroup ]]) {
    int num_blocks = (cols + 255) / 256;
    device const uchar *row_ptr = weight + (uint)idx * num_blocks * 144;
    
    // Each thread handles a portion of 256-elements blocks
    // If we want high performance, we'd distribute blocks across threads.
    // For now, simpler: each thread handles one or more blocks if needed.
    // But cols is usually small-ish (4096).
    // 4096 / 256 = 16 blocks.
    
    for (int i = tid; i < num_blocks; i += 1024) {
        device const uchar *block = row_ptr + i * 144;
        float d = fp16_to_fp32(*(device const ushort*)(block));
        float dmin = fp16_to_fp32(*(device const ushort*)(block + 2));
        
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

kernel void att_scores_f16_v2(device const half *q [[ buffer(0) ]],
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
        for (int t = pos + 1; t < stride; t++) {
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
            // Clamp score to prevent softmax overflow
            // Typical attention scores are in range [-50, 50] for headDim=128
            float score = d * scale;
            scores[h * stride + t] = clamp(score, -100.0f, 100.0f);
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
                        constant int &max_context_len [[ buffer(9) ]],
                        uint3 tid [[ thread_position_in_threadgroup ]],
                        uint3 nthreads [[ threads_per_threadgroup ]],
                        uint3 group_id [[ threadgroup_position_in_grid ]]) {
    uint h = group_id.x; if (h >= (uint)num_heads) return;
    uint kvh = h / (num_heads / kv_heads);
    uint kv_dim = kv_heads * headDim;
    float scale = 1.0f / sqrt((float)headDim);

    // SLIDING WINDOW: Determine window bounds
    int start = (window_size > 0 && pos >= window_size) ? (pos - window_size + 1) : 0;

    // Use dynamic memory allocation based on actual context length
    // For small contexts (<= 8192), use direct storage
    // For larger contexts, use streaming approach
    int effective_len = pos - start + 1;
    int max_smem_tokens = (max_context_len > 0 && max_context_len < 4096) ? max_context_len : 4096;

    device const half *mq = q + h * headDim;
    uint nt = nthreads.x;

    // Determine processing strategy based on context length
    // Increase limit to 8192 tokens for fast pass if device has enough smem (32KB is usually enough for 8192 floats? No, 8192*4 = 32KB. Yes!)
    if (effective_len <= 4096) {
        // Direct processing: all tokens fit in threadgroup memory
        threadgroup float s_mem[4096];
        // Note: s_mem is indexed by relative position (t - start), not absolute t
        // This is because KV cache uses rolling buffer indexing (t % window_size)
        // but s_mem needs contiguous storage for the softmax computation

        // 1. Parallel score calculation
        for (int t = tid.x; t <= pos; t += nt) {
            if (t >= start) {
                float d = 0;
                int cache_idx = (window_size > 0) ? (t % window_size) : t;
                device const half *mk = k_cache + cache_idx * kv_dim + kvh * headDim;
                for (int i = 0; i < headDim; i++) d += (float)mq[i] * (float)mk[i];
                s_mem[t - start] = d * scale;  // Use relative indexing
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. Softmax Max Reduction
        threadgroup float tg_mv;
        if (tid.x == 0) tg_mv = -10000.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        float l_mv = -10000.0f;
        for (int i = tid.x; i < effective_len; i += nt) {
            float val = s_mem[i];
            if (val > l_mv) l_mv = val;
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
        if (tid.x == 0) tg_se = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float l_se = 0;
        for (int i = tid.x; i < effective_len; i += nt) {
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
                r += (s_mem[t - start] / se) * (float)v_cache[cache_idx * kv_dim + kvh * headDim + tid.x];
            }
            output[h * headDim + tid.x] = safe_half(r);
        }
    } else {
        // Streaming approach: process tokens in chunks
        // Use smaller threadgroup memory for softmax state
        threadgroup float chunk_max[32];
        threadgroup float chunk_sum[32];

        float r = 0;
        float global_max = -10000.0f;

        // First pass: compute global max
        for (int t = start; t <= pos; t++) {
            float d = 0;
            int cache_idx = (window_size > 0) ? (t % window_size) : t;
            device const half *mk = k_cache + cache_idx * kv_dim + kvh * headDim;
            for (int i = 0; i < headDim; i++) d += (float)mq[i] * (float)mk[i];
            float score = d * scale;
            if (score > global_max) global_max = score;
        }

        // Second pass: compute exp scores and accumulate with numerical stability
        float global_sum = 0;
        for (int t = start; t <= pos; t++) {
            float d = 0;
            int cache_idx = (window_size > 0) ? (t % window_size) : t;
            device const half *mk = k_cache + cache_idx * kv_dim + kvh * headDim;
            for (int i = 0; i < headDim; i++) d += (float)mq[i] * (float)mk[i];
            float score = d * scale;
            // Clamp score to prevent exp overflow
            score = clamp(score, -100.0f, 100.0f);
            float exp_score = exp(score - global_max);
            global_sum += exp_score;

            // Accumulate value weighted by exp_score
            for (int i = 0; i < headDim; i++) {
                if (tid.x == i) {
                    r += exp_score * (float)v_cache[cache_idx * kv_dim + kvh * headDim + i];
                }
            }
        }
        r = simd_sum(r);
        if (global_sum == 0) global_sum = 1e-6f;

        if (tid.x < (uint)headDim) {
            output[h * headDim + tid.x] = safe_half(r / global_sum);
        }
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
// Utility Kernels
// ==========================================
kernel void fill_zero(device uchar *data [[ buffer(0) ]],
                      constant int &size [[ buffer(1) ]],
                      uint tid [[ thread_position_in_grid ]]) {
    if (tid < (uint)size) {
        data[tid] = 0;
    }
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
        out[idx] = safe_half(x[idx] * scale * (float)w[i]);
    }
}

kernel void rmsnorm_f32_to_f16_v4(device const float *x [[ buffer(0) ]],
                      device half *out [[ buffer(1) ]],
                      device const half *w [[ buffer(2) ]],
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
        float d = fp16_to_fp32(*(device const ushort*)(block));
        float dmin = fp16_to_fp32(*(device const ushort*)(block + 2));


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

kernel void linear_q4k_f32_f16(device const uchar *weight [[ buffer(0) ]],
                          device const float *input [[ buffer(1) ]],
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
    device const float *in_ptr = input + batch * dim_in;
    float sum = 0;
    for (int i = (int)lane_id; i < num_blocks; i += 32) {
        device const uchar *block = row_ptr + i * 144;
        float d = fp16_to_fp32(*(device const ushort*)(block));
        float dmin = fp16_to_fp32(*(device const ushort*)(block + 2));

        float scale_val = d * scale;
        float min_val = dmin * scale;
        device const uchar *scales = block + 4;
        device const uchar *qs = block + 16;
        for (int j = 0; j < 8; j++) {
            uchar sc, m;
            if (j < 4) { sc = scales[j] & 63; m = scales[j + 4] & 63; }
            else { sc = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4); m = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4); }
            float d_val = scale_val * (float)sc;
            float m_val = min_val * (float)m;
            int sub_offset = j * 32, qs_offset = j * 16;
            for (int k = 0; k < 16; k++) {
                uchar b = qs[qs_offset + k];
                sum += (float)in_ptr[i * 256 + sub_offset + k] * (d_val * (float)(b & 0xF) - m_val);
                sum += (float)in_ptr[i * 256 + sub_offset + k + 16] * (d_val * (float)(b >> 4) - m_val);
            }
        }
    }
    sum = simd_sum(sum);
    if (lane_id == 0) output[batch * dim_out + row] = safe_half(sum);
}


kernel void swiglu_f32(device const float *gate [[ buffer(0) ]], 
                     device const float *up [[ buffer(1) ]], 
                     device float *out [[ buffer(2) ]],
                     uint qid [[ thread_position_in_grid ]]) {
    float g = gate[qid];
    float u = up[qid];
    
    // SwiGLU: up * silu(x) where silu(x) = x * sigmoid(x)
    // Clamp gate to prevent extreme sigmoid values
    float g_clamped = clamp(g, -10.0f, 10.0f);
    float sigmoid_g = g_clamped / (1.0f + exp(-g_clamped));
    float val = u * sigmoid_g;
    
    // Clamp output to prevent activation explosion
    // Use larger clamp for FP32 (1e4) to allow meaningful values while preventing overflow
    out[qid] = clamp(val, -1e4f, 1e4f);
}

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
    if (lane_id == 0) {
        // Clamp output to prevent activation explosion
        output[batch * dim_out + row] = clamp(sum, -1e4f, 1e4f);
    }
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

kernel void linear_q6k_f16_f32(device const uchar *weight [[ buffer(0) ]],
                          device const half *input [[ buffer(1) ]],
                          device float *output [[ buffer(2) ]],
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
        ushort d_bits = *(device const ushort*)(block + 208);
        
        float d;
        uint d_sign = (d_bits & 0x8000u) << 16;
        uint d_exp_raw = (d_bits & 0x7C00u) >> 10;
        uint d_mant = d_bits & 0x03FFu;
        if (d_exp_raw == 0) {
            if (d_mant == 0) d = 0.0f;
            else {
                uint shift = 0; uint test_mant = d_mant;
                while ((test_mant & 0x0400) == 0) { test_mant <<= 1; shift++; }
                d = as_type<float>(d_sign | ((113 - shift) << 23) | ((test_mant & 0x03FF) << 13));
            }
        } else {
            d = as_type<float>(d_sign | ((d_exp_raw + 112) << 23) | (d_mant << 13));
        }

        device const half *in_ptr_block = input + batch * dim_in + i * 256;
        
        for (int l = 0; l < 16; l++) {
            float s = d * scale * (float)sc[l];
            int group_off = l * 16;
            for (int k = 0; k < 16; k += 2) {
                int idx = group_off + k;
                uchar b = ql[idx / 2];
                uchar h0 = (qh[idx / 4] >> ((idx % 4) * 2)) & 3;
                uchar h1 = (qh[(idx+1) / 4] >> (((idx+1) % 4) * 2)) & 3;
                
                sum += s * (float)((int8_t)((h0 << 4) | (b & 0xF)) - 32) * input[batch * dim_in + i * 256 + idx];
                sum += s * (float)((int8_t)((h1 << 4) | (b >> 4)) - 32) * input[batch * dim_in + i * 256 + idx + 1];
            }
        }
    }
    
    sum = simd_sum(sum);
    if (lane_id == 0) output[batch * dim_out + row] = sum;
}

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
    int cache_pos = pos % window_size;
    k_cache[cache_pos * kv_dim + tid] = k[tid];
    v_cache[cache_pos * kv_dim + tid] = v[tid];
}

kernel void rmsnorm_linear_q4k_f16(device const half *input [[ buffer(0) ]],
                                   device const half *norm_weight [[ buffer(1) ]],
                                   device const uchar *weight [[ buffer(2) ]],
                                   device half *output [[ buffer(3) ]],
                                   constant float &eps [[ buffer(4) ]],
                                   constant int &dim_in [[ buffer(5) ]],
                                   constant int &dim_out [[ buffer(6) ]],
                                   constant float &scale [[ buffer(7) ]],
                                   constant int &batchSize [[ buffer(8) ]],
                                   uint3 tid [[ thread_position_in_threadgroup ]],
                                   uint3 qid [[ thread_position_in_grid ]]) {
    uint row = qid.y; uint batch = qid.z;
    if (batch >= (uint)batchSize) return;
    if (row >= (uint)dim_out) return; uint lane_id = tid.x;

    // 1. Threadgroup-wide Disjoint RMSNorm
    threadgroup float s[32];
    float sum_sq = 0.0f;
    for (int i = (int)tid.x; i < dim_in; i += 1024) {
        float val = (float)input[batch * dim_in + i];
        sum_sq += val * val;
    }
    sum_sq = simd_sum(sum_sq);
    if ((tid.x & 31) == 0) s[tid.x / 32] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float total_sum_sq = 0;
    if (tid.x < 32) {
        float t = s[tid.x]; // Note: s[0..31] are filled (or 0)
        t = simd_sum(t);
        if (tid.x == 0) s[0] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rms = 1.0f / sqrt(s[0] / (float)dim_in + eps);

    // 2. Linear Q4K transformation (only first 16 threads for this small test/segment)
    int num_blocks = (dim_in + 255) / 256;
    device const uchar *row_ptr = weight + row * num_blocks * 144;
    float sum = 0;

    if (tid.x < 16) {
        for (int i = 0; i < num_blocks; i++) {
            device const uchar *block = row_ptr + i * 144;
            float d = fp16_to_fp32(*(device const ushort*)(block));
            float dmin = fp16_to_fp32(*(device const ushort*)(block + 2));
            device const uchar *scales = block + 4;
            device const uchar *qs = block + 16;
            
            for (int j = 0; j < 8; j++) {
                uchar sc, m;
                if (j < 4) {
                    sc = scales[j] & 63;
                    m = scales[j + 4] & 63;
                } else {
                    sc = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4);
                    m = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4);
                }
                float d_val = d * scale * (float)sc, m_val = dmin * scale * (float)m;
                int sub_offset = j * 32, qs_offset = j * 16;
                int k = (int)tid.x;
                
                uchar b = qs[qs_offset + k];
                float w0 = d_val * (float)(b & 0xF) - m_val;
                float w1 = d_val * (float)(b >> 4) - m_val;
                int idx0 = i * 256 + sub_offset + k;
                int idx1 = idx0 + 16;
                
                if (idx0 < dim_in) sum += w0 * (float)input[batch * dim_in + idx0] * rms * (float)norm_weight[idx0];
                if (idx1 < dim_in) sum += w1 * (float)input[batch * dim_in + idx1] * rms * (float)norm_weight[idx1];
            }
        }
    }

    sum = simd_sum(sum);
    if (tid.x == 0) output[batch * dim_out + row] = safe_half(sum);
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
    float sum = 0;

    const device half *input_row = input + batch * dim_in;

    // Distribute all (num_blocks * 8) sub-blocks across 32 threads
    for (int idx = (int)lane_id; idx < num_blocks * 8; idx += 32) {
        int i = idx / 8;
        int j = idx % 8;
        
        device const uchar *block = row_ptr + i * 144;
        float d = fp16_to_fp32(*(device const ushort*)(block));
        float dmin = fp16_to_fp32(*(device const ushort*)(block + 2));

        device const uchar *scales = block + 4;
        
        float sc, m;
        if (j < 4) {
            sc = (float)(scales[j] & 63);
            m = (float)(scales[j + 4] & 63);
        } else {
            sc = (float)((scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4));
            m = (float)((scales[j+4] >> 4) | ((scales[j] >> 6) << 4));
        }

        float d_val = d * scale * sc;
        float m_val = dmin * scale * m;
        
        int offset = i * 256 + j * 32;
        device const uchar *q = block + 16 + j * 16;

        #pragma unroll
        for (int k = 0; k < 16; k++) {
            uchar b = q[k];
            sum += (d_val * (float)(b & 0xF) - m_val) * (float)input_row[offset + k];
            sum += (d_val * (float)(b >> 4) - m_val) * (float)input_row[offset + k + 16];
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
                          constant float &scale [[ buffer(5) ]],
                          uint3 tid [[ thread_position_in_threadgroup ]],
                          uint3 qid [[ thread_position_in_grid ]]) {
    uint row = qid.y; uint batch = qid.z;
    if (row >= (uint)dim_out) return; uint lane_id = tid.x;
    int num_blocks = (dim_in + 255) / 256;
    device const uchar *row_ptr = weight + row * num_blocks * 110;
    float sum = 0;

    for (int i = (int)lane_id; i < num_blocks; i += 32) {
        device const uchar *block = row_ptr + i * 110;
        float d = fp16_to_fp32(*(device const ushort*)(block + 108));
        device const uchar *hmask = block;
        device const uchar *qs = block + 32;
        device const uchar *scales = block + 96;

        for (int j = 0; j < 8; j++) {
            float s = d * scale * (float)(scales[j] & 63);
            int sub_offset = j * 32;
            for (int k = 0; k < 16; k++) {
                int idx = i * 256 + sub_offset + k;
                if (idx >= dim_in) continue;

                int qs_idx = (j * 32 + k) / 4;
                int qs_shift = ((j * 32 + k) % 4) * 2;
                int q = (qs[qs_idx] >> qs_shift) & 3;

                int hmask_idx = (j * 32 + k) / 8;
                int hmask_shift = (j * 32 + k) % 8;
                int hbit = (hmask[hmask_idx] >> hmask_shift) & 1;

                int val = (hbit << 2) | q;
                float w = s * (float)val - s * 4.0f; // Center around 0
                sum += w * (float)input[batch * dim_in + idx];
            }
        }
    }

    sum = simd_sum(sum);
    if (lane_id == 0) output[batch * dim_out + row] = safe_half(sum);
}

kernel void swiglu_linear_q4k_f16(device const half *gate [[ buffer(0) ]],
                                  device const half *up [[ buffer(1) ]],
                                  device const uchar *weight [[ buffer(2) ]],
                                  device half *output [[ buffer(3) ]],
                                  constant int &dim_in [[ buffer(4) ]],
                                  constant int &dim_out [[ buffer(5) ]],
                                  constant float &scale [[ buffer(6) ]],
                                  uint3 tid [[ thread_position_in_threadgroup ]],
                                  uint3 qid [[ thread_position_in_grid ]]) {
    uint row = qid.y; uint batch = qid.z;
    if (row >= (uint)dim_out) return; uint lane_id = tid.x;

    int num_blocks = (dim_in + 255) / 256;
    device const uchar *row_ptr = weight + row * num_blocks * 144;
    float sum = 0;

    const device half *gate_row = gate + batch * dim_in;
    const device half *up_row = up + batch * dim_in;

    for (int idx = (int)lane_id; idx < num_blocks * 8; idx += 32) {
        int i = idx / 8;
        int j = idx % 8;
        
        device const uchar *block = row_ptr + i * 144;
        float d = fp16_to_fp32(*(device const ushort*)(block));
        float dmin = fp16_to_fp32(*(device const ushort*)(block + 2));

        device const uchar *scales = block + 4;
        
        float sc, m;
        if (j < 4) {
            sc = (float)(scales[j] & 63);
            m = (float)(scales[j + 4] & 63);
        } else {
            sc = (float)((scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4));
            m = (float)((scales[j+4] >> 4) | ((scales[j] >> 6) << 4));
        }

        float d_val = d * scale * sc;
        float m_val = dmin * scale * m;
        
        int offset = i * 256 + j * 32;
        device const uchar *q = block + 16 + j * 16;

        #pragma unroll
        for (int k = 0; k < 16; k++) {
            uchar b = q[k];
            
            // Sub-block 0
            {
                float g = (float)gate_row[offset + k];
                float u = (float)up_row[offset + k];
                float swish = u * (g / (1.0f + exp(-clamp(g, -10.0f, 10.0f))));
                sum += (d_val * (float)(b & 0xF) - m_val) * swish;
            }
            // Sub-block 1
            {
                float g = (float)gate_row[offset + k + 16];
                float u = (float)up_row[offset + k + 16];
                float swish = u * (g / (1.0f + exp(-clamp(g, -10.0f, 10.0f))));
                sum += (d_val * (float)(b >> 4) - m_val) * swish;
            }
        }
    }
    sum = simd_sum(sum);
    if (lane_id == 0) output[batch * dim_out + row] = safe_half(sum);
}

kernel void rmsnorm_linear_f16(device const half *input [[ buffer(0) ]],
                              device const half *norm_weight [[ buffer(1) ]],
                              device const half *weight [[ buffer(2) ]],
                              device half *output [[ buffer(3) ]],
                              constant float &eps [[ buffer(4) ]],
                              constant int &dim_in [[ buffer(5) ]],
                              constant int &dim_out [[ buffer(6) ]],
                              constant int &batchSize [[ buffer(7) ]],
                              uint3 tid [[ thread_position_in_threadgroup ]],
                              uint3 qid [[ thread_position_in_grid ]]) {
    uint row = qid.y; uint batch = qid.z;
    if (batch >= (uint)batchSize) return;
    if (row >= (uint)dim_out) return; uint lane_id = tid.x;

    // 1. RMSNorm
    threadgroup float s[32];
    float sum_sq = 0.0f;
    for (int i = lane_id; i < dim_in; i += 1024) {
        float val = (float)input[batch * dim_in + i];
        sum_sq += val * val;
    }
    s[lane_id] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane_id == 0) {
        float t = 0;
        for (int i = 0; i < 32; i++) t += s[i];
        s[0] = 1.0f / sqrt(t / (float)dim_in + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rms_scale = s[0];

    // 2. Linear
    float sum = 0;
    for (int i = lane_id; i < dim_in; i += 32) {
        float val = (float)input[batch * dim_in + i] * rms_scale * (float)norm_weight[i];
        sum += val * (float)weight[row * dim_in + i];
    }
    sum = simd_sum(sum);
    if (lane_id == 0) {
        output[batch * dim_out + row] = safe_half(sum);
    }
}

kernel void rmsnorm_qkv_f16(device const half *input [[ buffer(0) ]],
                           device const half *norm_weight [[ buffer(1) ]],
                           device const half *q_weight [[ buffer(2) ]],
                           device const half *k_weight [[ buffer(3) ]],
                           device const half *v_weight [[ buffer(4) ]],
                           device half *q_out [[ buffer(5) ]],
                           device half *k_out [[ buffer(6) ]],
                           device half *v_out [[ buffer(7) ]],
                           constant int &dim_in [[ buffer(8) ]],
                           constant int &q_dim [[ buffer(9) ]],
                           constant int &kv_dim [[ buffer(10) ]],
                           constant float &eps [[ buffer(11) ]],
                           constant int &batchSize [[ buffer(12) ]],
                           uint3 tid [[ thread_position_in_threadgroup ]],
                           uint3 qid [[ thread_position_in_grid ]]) {
    uint row = qid.y; uint batch = qid.z;
    if (batch >= (uint)batchSize) return;
    uint lane_id = tid.x;

    // 1. RMSNorm (shared across Q, K, V threads in the same batch)
    // We'll use a simpler approach: each thread computes its own part of RMSNorm
    // or we can use threadgroup memory if we coordinate.
    // Since we want to fuse, let's assume qid.y handles different output rows.
    // Q, K, V outputs have different row counts. This kernel might be tricky to balance.
    
    // Simpler approach for now: One kernel that just does the Norm once per batch and then the projections.
    // But Metal dispatches are 3D. 
    // Let's use qid.y for the MAX of (q_dim, kv_dim).
    
    threadgroup float rms_scale_tg;
    if (lane_id == 0 && row == 0) {
        float sum_sq = 0;
        for (int i = 0; i < dim_in; i++) {
            float val = (float)input[batch * dim_in + i];
            sum_sq += val * val;
        }
        rms_scale_tg = 1.0f / sqrt(sum_sq / (float)dim_in + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rms_scale = rms_scale_tg;

    // Q Proj
    if (row < (uint)q_dim) {
        float sum = 0;
        for (int i = lane_id; i < dim_in; i += 32) {
            float val = (float)input[batch * dim_in + i] * rms_scale * (float)norm_weight[i];
            sum += val * (float)q_weight[row * dim_in + i];
        }
        sum = simd_sum(sum);
        if (lane_id == 0) q_out[batch * q_dim + row] = safe_half(sum);
    }

    // K Proj
    if (row < (uint)kv_dim) {
        float sum = 0;
        for (int i = lane_id; i < dim_in; i += 32) {
            float val = (float)input[batch * dim_in + i] * rms_scale * (float)norm_weight[i];
            sum += val * (float)k_weight[row * dim_in + i];
        }
        sum = simd_sum(sum);
        if (lane_id == 0) k_out[batch * kv_dim + row] = safe_half(sum);
    }

    // V Proj
    if (row < (uint)kv_dim) {
        float sum = 0;
        for (int i = lane_id; i < dim_in; i += 32) {
            float val = (float)input[batch * dim_in + i] * rms_scale * (float)norm_weight[i];
            sum += val * (float)v_weight[row * dim_in + i];
        }
        sum = simd_sum(sum);
        if (lane_id == 0) v_out[batch * kv_dim + row] = safe_half(sum);
    }
}

kernel void store_kv_f16_batch(device const half *k [[ buffer(0) ]],
                              device const half *v [[ buffer(1) ]],
                              device half *k_cache [[ buffer(2) ]],
                              device half *v_cache [[ buffer(3) ]],
                              constant int &pos_offset [[ buffer(4) ]],
                              constant int &kv_dim [[ buffer(5) ]],
                              constant int &window_size [[ buffer(6) ]],
                              uint2 gid [[ thread_position_in_grid ]]) {
    if (gid.x >= (uint)kv_dim) return;
    int p = pos_offset + gid.y;
    int cache_pos = p % window_size;
    k_cache[cache_pos * kv_dim + gid.x] = k[gid.y * kv_dim + gid.x];
    v_cache[cache_pos * kv_dim + gid.x] = v[gid.y * kv_dim + gid.x];
}

kernel void rmsnorm_qkv_q4k_f16(device const half *input [[ buffer(0) ]],
                               device const float *norm_weight [[ buffer(1) ]],
                               device const uchar *q_weight [[ buffer(2) ]],
                               device const uchar *k_weight [[ buffer(3) ]],
                               device const uchar *v_weight [[ buffer(4) ]],
                               device half *q_out [[ buffer(5) ]],
                               device half *k_out [[ buffer(6) ]],
                               device half *v_out [[ buffer(7) ]],
                               constant int &dim_in [[ buffer(8) ]],
                               constant int &q_dim [[ buffer(9) ]],
                               constant int &kv_dim [[ buffer(10) ]],
                               constant float &eps [[ buffer(11) ]],
                               constant float &scale [[ buffer(12) ]],
                               constant int &batchSize [[ buffer(13) ]],
                               uint3 tid [[ thread_position_in_threadgroup ]],
                               uint3 qid [[ thread_position_in_grid ]]) {
    uint row = qid.y; uint batch = qid.z;
    if (batch >= (uint)batchSize) return;
    uint lane_id = tid.x;

    // 1. RMSNorm (simplified per-thread or shared)
    threadgroup float rms_scale_tg;
    if (lane_id == 0 && row == 0) {
        float sum_sq = 0;
        for (int i = 0; i < dim_in; i++) {
            float val = (float)input[batch * dim_in + i];
            sum_sq += val * val;
        }
        rms_scale_tg = 1.0f / sqrt(sum_sq / (float)dim_in + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rms_scale = rms_scale_tg;

    // We'll compute normalized input on the fly to save memory, 
    // but for Q4K it's better to have it in threadgroup if possible.
    // However, dim_in can be 4096. 4096 * 2 bytes = 8KB. 
    // Threadgroup can handle it.
    
    threadgroup half normalized[4096];
    if (row == 0) {
        for (int i = lane_id; i < dim_in; i += 32) {
            normalized[i] = safe_half((float)input[batch * dim_in + i] * rms_scale * norm_weight[i]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Q Proj
    if (row < (uint)q_dim) {
        int num_blocks = (dim_in + 255) / 256;
        device const uchar *row_ptr = q_weight + row * num_blocks * 144;
        float sum = 0;
        for (int i = (int)lane_id; i < num_blocks; i += 32) {
            device const uchar *block = row_ptr + i * 144;
            float d = fp16_to_fp32(*(device const ushort*)(block));
            float dmin = fp16_to_fp32(*(device const ushort*)(block + 2));
            device const uchar *scales = block + 4;
            device const uchar *qs = block + 16;
            for (int j = 0; j < 8; j++) {
                uchar sc, m;
                if (j < 4) { sc = scales[j] & 63; m = scales[j + 4] & 63; }
                else { sc = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4); m = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4); }
                float d_val = d * scale * (float)sc, m_val = dmin * scale * (float)m;
                for (int k = 0; k < 16; k++) {
                    uchar b = qs[j * 16 + k];
                    sum += (float)normalized[i * 256 + j * 32 + k] * (d_val * (float)(b & 0xF) - m_val);
                    sum += (float)normalized[i * 256 + j * 32 + k + 16] * (d_val * (float)(b >> 4) - m_val);
                }
            }
        }
        sum = simd_sum(sum);
        if (lane_id == 0) q_out[batch * q_dim + row] = safe_half(sum);
    }

    // K Proj
    if (row < (uint)kv_dim) {
        int num_blocks = (dim_in + 255) / 256;
        device const uchar *row_ptr = k_weight + row * num_blocks * 144;
        float sum = 0;
        for (int i = (int)lane_id; i < num_blocks; i += 32) {
            device const uchar *block = row_ptr + i * 144;
            float d = fp16_to_fp32(*(device const ushort*)(block));
            float dmin = fp16_to_fp32(*(device const ushort*)(block + 2));
            device const uchar *scales = block + 4;
            device const uchar *qs = block + 16;
            for (int j = 0; j < 8; j++) {
                uchar sc, m;
                if (j < 4) { sc = scales[j] & 63; m = scales[j + 4] & 63; }
                else { sc = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4); m = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4); }
                float d_val = d * scale * (float)sc, m_val = dmin * scale * (float)m;
                for (int k = 0; k < 16; k++) {
                    uchar b = qs[j * 16 + k];
                    sum += (float)normalized[i * 256 + j * 32 + k] * (d_val * (float)(b & 0xF) - m_val);
                    sum += (float)normalized[i * 256 + j * 32 + k + 16] * (d_val * (float)(b >> 4) - m_val);
                }
            }
        }
        sum = simd_sum(sum);
        if (lane_id == 0) k_out[batch * kv_dim + row] = safe_half(sum);
    }

    // V Proj
    if (row < (uint)kv_dim) {
        int num_blocks = (dim_in + 255) / 256;
        device const uchar *row_ptr = v_weight + row * num_blocks * 144;
        float sum = 0;
        for (int i = (int)lane_id; i < num_blocks; i += 32) {
            device const uchar *block = row_ptr + i * 144;
            float d = fp16_to_fp32(*(device const ushort*)(block));
            float dmin = fp16_to_fp32(*(device const ushort*)(block + 2));
            device const uchar *scales = block + 4;
            device const uchar *qs = block + 16;
            for (int j = 0; j < 8; j++) {
                uchar sc, m;
                if (j < 4) { sc = scales[j] & 63; m = scales[j + 4] & 63; }
                else { sc = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4); m = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4); }
                float d_val = d * scale * (float)sc, m_val = dmin * scale * (float)m;
                for (int k = 0; k < 16; k++) {
                    uchar b = qs[j * 16 + k];
                    sum += (float)normalized[i * 256 + j * 32 + k] * (d_val * (float)(b & 0xF) - m_val);
                    sum += (float)normalized[i * 256 + j * 32 + k + 16] * (d_val * (float)(b >> 4) - m_val);
                }
            }
        }
        sum = simd_sum(sum);
        if (lane_id == 0) v_out[batch * kv_dim + row] = safe_half(sum);
    }
}

// Paged Attention Kernel
kernel void att_paged_f16(device const half *q [[ buffer(0) ]],
                        device const half *k_cache [[ buffer(1) ]],
                        device const half *v_cache [[ buffer(2) ]],
                        device half *output [[ buffer(3) ]],
                        constant int &pos [[ buffer(4) ]],
                        constant int &num_heads [[ buffer(5) ]],
                        constant int &kv_heads [[ buffer(6) ]],
                        constant int &headDim [[ buffer(7) ]],
                        constant int &block_size [[ buffer(8) ]],
                        constant int &max_context_len [[ buffer(9) ]],
                        device const int *block_table [[ buffer(10) ]],
                        uint3 tid [[ thread_position_in_threadgroup ]],
                        uint3 nthreads [[ threads_per_threadgroup ]],
                        uint3 group_id [[ threadgroup_position_in_grid ]]) {
    uint h = group_id.x; if (h >= (uint)num_heads) return;
    uint kvh = h / (num_heads / kv_heads);
    float scale = 1.0f / sqrt((float)headDim);

    device const half *mq = q + h * headDim;
    uint nt = nthreads.x;

    // Use shared memory for scores
    // Limit to 4096 tokens for this kernel version
    threadgroup float s_mem[4096];
    
    int effective_len = pos + 1;
    if (effective_len > 4096) effective_len = 4096; // Cap for safety/smem

    // 1. Parallel score calculation
    for (int t = tid.x; t < effective_len; t += nt) {
        int logical_block = t / block_size;
        int block_offset = t % block_size;
        int physical_block = block_table[logical_block];
        
        int kv_dim = kv_heads * headDim;
        int block_stride = block_size * kv_dim;
        
        device const half *mk = k_cache + physical_block * block_stride + block_offset * kv_dim + kvh * headDim;
        
        float d = 0;
        for (int i = 0; i < headDim; i++) d += (float)mq[i] * (float)mk[i];
        s_mem[t] = d * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Softmax Max Reduction
    threadgroup float tg_mv;
    float l_mv = -10000.0f;
    for (int i = tid.x; i < effective_len; i += nt) {
        float val = s_mem[i];
        if (val > l_mv) l_mv = val;
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
    for (int i = tid.x; i < effective_len; i += nt) {
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
        int kv_dim = kv_heads * headDim;
        int block_stride = block_size * kv_dim;

        for (int t = 0; t < effective_len; t++) {
            int logical_block = t / block_size;
            int block_offset = t % block_size;
            int physical_block = block_table[logical_block];
            
            device const half *mv_ptr = v_cache + physical_block * block_stride + block_offset * kv_dim + kvh * headDim + tid.x;
            r += (s_mem[t] / se) * (float)*mv_ptr;
        }
        output[h * headDim + tid.x] = safe_half(r);
    }
}

// ==========================================
// Mamba / SSM Kernels
// ==========================================

// 1D Causal Convolution (Depthwise)
// Arguments:
// - input: [batch, dim] (Current token)
// - weight: [dim, kernel_size] (Conv weights)
// - bias: [dim] (Optional)
// - state: [batch, dim, kernel_size] (Conv state ring buffer)
// - output: [batch, dim]
// - dim: Hidden dimension (d_inner)
// - kernel_size: usually 4
// - step: current time step (to index ring buffer)
kernel void mamba_conv1d_f16(
    device const half *input [[ buffer(0) ]],
    device const half *weight [[ buffer(1) ]], // [dim, kernel_size]
    device const half *bias [[ buffer(2) ]],   // [dim]
    device half *state [[ buffer(3) ]],        // [dim, kernel_size] (assuming batch=1 for now)
    device half *output [[ buffer(4) ]],
    constant int &dim [[ buffer(5) ]],
    constant int &kernel_size [[ buffer(6) ]],
    uint tid [[ thread_position_in_grid ]]) {

    if (tid >= (uint)dim) return;

    // Shift state: This is a causal convolution on a stream.
    // In efficient implementation (like ring buffer), we write new input to state[idx]
    // and conv across the window.
    // For simplicity here: we assume 'state' is the history window.
    // Actually, physically shifting is expensive. Using a ring buffer approach is better.
    // Let's assume input 'state' is already managed by the host or we do a shift.
    // For this kernel, let's do a simple shift for correctness first (optimize later with ring buffer).
    
    // Shift history (inefficient but clear)
    // state[tid, 0] = newest
    // state[tid, k-1] = oldest
    // Weight is [dim, k], usually weight[tid, 0] corresponds to newest?
    // Mamba conv is usually: out[t] = sum(k=0..K-1) weight[k] * input[t-k]
    
    // Shift logic:
    for (int k = kernel_size - 1; k > 0; k--) {
        state[tid * kernel_size + k] = state[tid * kernel_size + (k - 1)];
    }
    state[tid * kernel_size + 0] = input[tid];

    float sum = (float)bias[tid];
    for (int k = 0; k < kernel_size; k++) {
        float w = (float)weight[tid * kernel_size + k];
        float x = (float)state[tid * kernel_size + k];
        sum += w * x;
    }
    
    // Silu activation is typical after conv in Mamba block structure, 
    // but often it's: x -> conv -> silu -> ssm.
    // The conv1d output usually goes to the SSM. 
    // Let's verify Mamba architecture: 
    // Project -> Conv1d -> Silu -> SSM.
    // So this kernel just does Conv1d + Bias.
    
    output[tid] = (half)sum;
}


// Mamba Selective Scan (SSM) Step - Single Token (Inference Mode)
// Equations:
// x_t = input[t] (after conv)
// dt = softplus(dt_bias + dt_proj(x_t))
// A_bar = exp(dt * A)
// B_bar = dt * B
// h_t = A_bar * h_{t-1} + B_bar * x_t (element-wise A, simplified diagonal)
// y_t = C * h_t * D * x_t (Wait, D is usually residual)
// Real Mamba:
//   dt = softplus(Parameter + dt_proj(x))
//   dA = exp(dt * A)
//   dB = dt * B 
//   h = dA * h + dB * x
//   y = C * h + D * x
//
// Arguments:
// - u: [dim] (input x)
// - h: [dim, d_state] (hidden state matrix)
// - A: [dim, d_state] (-exp(A_log) parameter usually, or just A)
// - B: [d_state] (input dependent or fixed? In Mamba B is input dependent [batch, d_state] projected from x)
// - C: [d_state] (projected from x)
// - D: [dim]
// - dt_bias: [dim]
// - dt_weight: [dim] or derived? dt is usually projected from x with rank 1 or small rank.
//   Actually dt is [batch, dim] in Mamba (projected from x).
//   Let's assume the host computes dt, B, C if they are input-dependent projections.
//   Or this kernel computes them if we pass the weights.
//   For this "Core" kernel, let's assume B, C, dt are ALREADY computed/projected by linear layers in Go.
//   So we just update the SSM state h.
kernel void mamba_scan_step_f16(device const half *u [[ buffer(0) ]],
                               device half *h [[ buffer(1) ]],
                               device const half *A [[ buffer(2) ]],
                               device const half *B [[ buffer(3) ]],
                               device const half *C [[ buffer(4) ]],
                               device const half *D [[ buffer(5) ]],
                               device const half *dt [[ buffer(6) ]],
                               device half *y [[ buffer(7) ]],
                               constant int &n_heads [[ buffer(8) ]],
                               constant int &d_state [[ buffer(9) ]],
                               constant int &head_dim [[ buffer(10) ]],
                               uint tid [[ thread_position_in_grid ]]) {
    int i = (int)tid;
    int head_idx = i / head_dim;
    int chan_idx = i % head_dim;
    
    float u_val = (float)u[i];
    float dt_val = (float)dt[head_idx];
    float a_val = (float)A[head_idx];
    float d_val = (float)D[i];
    
    float dA = exp(dt_val * a_val);
    
    float out_val = 0;
    for (int n = 0; n < d_state; n++) {
        int state_idx = head_idx * (d_state * head_dim) + n * head_dim + chan_idx;
        float b_val = (float)B[head_idx * d_state + n];
        float c_val = (float)C[head_idx * d_state + n];
        
        float h_prev = (float)h[state_idx];
        float h_next = dA * h_prev + (dt_val * b_val) * u_val;
        h[state_idx] = (half)h_next;
        
        out_val += h_next * c_val;
    }
    
    y[i] = (half)(out_val + d_val * u_val);
}

kernel void silu_f16(device const half *input [[ buffer(0) ]],
                     device half *output [[ buffer(1) ]],
                     uint id [[ thread_position_in_grid ]]) {
    float x = (float)input[id];
    float sig = 1.0f / (1.0f + exp(-x));
    output[id] = (half)(x * sig);
}

kernel void slice_f16(device const half *input [[ buffer(0) ]],
                     device half *output [[ buffer(1) ]],
                     constant int &start_col [[ buffer(2) ]],
                     constant int &num_cols [[ buffer(3) ]],
                     constant int &total_cols [[ buffer(4) ]],
                     uint id [[ thread_position_in_grid ]]) {
    int row = id / num_cols;
    int col = id % num_cols;
    int in_idx = row * total_cols + start_col + col;
    output[id] = input[in_idx];
}

kernel void mul_f16(device const half *a [[ buffer(0) ]],
                   device const half *b [[ buffer(1) ]],
                   device half *result [[ buffer(2) ]],
                   uint id [[ thread_position_in_grid ]]) {
    result[id] = a[id] * b[id];
}
