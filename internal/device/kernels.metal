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
                      device const float *w [[ buffer(2) ]],
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
    int num_blocks = (dim_in + 255) / 256;
    device const uchar *row_ptr = (device const uchar *)weight + row * num_blocks * 110;
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
        // theta = pos * base^(-2*i/d)
        // i = idx0
        float theta_i = (float)pos * pow(ropeTheta, -2.0f * (float)idx0 / (float)headDim);
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
    if (effective_len <= max_smem_tokens) {
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
    
    // SwiGLU: up * silu(gate) where silu(x) = x * sigmoid(x)
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
    
    // SLIDING WINDOW: Use rolling buffer indexing
    // If window_size == 0, use full sequence (backward compat)
    int cache_pos = (window_size > 0) ? (pos % window_size) : pos;
    int offset = cache_pos * kv_dim + tid;
    
    k_cache[offset] = k[tid];
    v_cache[offset] = v[tid];
}
