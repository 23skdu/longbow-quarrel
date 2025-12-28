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
    uint row = qid.y; if (row >= (uint)dim_out) return; uint lane_id = tid.x;
    device const half4 *w4 = (device const half4 *)(weight + row * dim_in);
    device const half4 *i4 = (device const half4 *)input;
    int n4 = dim_in / 4;
    float sum = 0; for (int i = (int)lane_id; i < n4; i += 32) {
        float4 v_w = (float4)w4[i]; float4 v_i = (float4)i4[i];
        sum += dot(v_w.xy, v_i.xy) + dot(v_w.zw, v_i.zw);
    }
    sum = simd_sum(sum); if (lane_id == 0) output[row] = (half)sum;
}

// Q4_K dot product (32 threads per row)
// Manual half decode to avoid FTZ
float manual_half_to_float(ushort x) {
    uint sign = (x >> 15) & 1;
    uint exp = (x >> 10) & 0x1F;
    uint mant = x & 0x3FF;
    
    if (exp == 0) {
        if (mant == 0) return (sign ? -0.0f : 0.0f);
        // Subnormal: val = mant * 2^-24
        float f = (float)mant * 5.9604645e-8f; // 2^-24
        return (sign ? -f : f);
    } else if (exp == 31) {
        // Inf/NaN - simpler to return exp treated approx or just return MAX
        return (sign ? -65504.0f : 65504.0f); 
    }
    
    // Normal: exp - 15 + 127 = exp + 112
    uint f_bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    return as_type<float>(f_bits);
}

// Safe Half Cast
half safe_half(float x) {
    return (half)clamp(x, -65504.0f, 65504.0f);
}

kernel void linear_q4k_f16(device const uchar *weight [[ buffer(0) ]],
                         device const half *input [[ buffer(1) ]],
                         device half *output [[ buffer(2) ]],
                         constant int &dim_in [[ buffer(3) ]],
                         constant int &dim_out [[ buffer(4) ]],
                         uint2 tid [[ thread_position_in_threadgroup ]],
                         uint2 qid [[ thread_position_in_grid ]]) {
    uint row = qid.y; if (row >= (uint)dim_out) return; uint lane_id = tid.x;
    
    // Q4_K Block Size = 256. Not Q3. Q4_K.
    // Copy-paste error in comments fixed below.
    int num_blocks = dim_in / 256;
    
    device const uchar *row_ptr = weight + row * num_blocks * 144;
    device const half *in_ptr = input;
    
    float sum = 0;
    
    // Loop over super-blocks
    for (int i = (int)lane_id; i < num_blocks; i += 32) {
        device const uchar *block = row_ptr + i * 144;
        
        // Header: d (f16), dmin (f16)
        // Load as ushort to access raw bits
        ushort d_bits = *(device const ushort*)(block);
        ushort dmin_bits = *(device const ushort*)(block + 2);
        
        float d = manual_half_to_float(d_bits);
        float dmin = manual_half_to_float(dmin_bits);

        // Decode 16 scales (12 bytes)
        // The line 'lock + 4;' from the instruction is malformed.
        // Retaining the original 'scales' declaration.
        device const uchar *scales = block + 4;
        device const uchar *qs = block + 16;
        
        // Decode 8 scales/mins (12 bytes -> 8 pairs)
        // Logic from dequant.go / k_quants.c
        uchar sc[8];
        uchar m[8];
        
        for (int j = 0; j < 4; j++) {
            uchar u_j = scales[j];
            uchar u_j4 = scales[j+4];
            uchar u_j8 = scales[j+8];
            
            sc[j]   = (u_j4 & 0xF) | ((u_j & 0x3) << 4);
            m[j]    = (u_j4 >> 4)  | ((u_j & 0xC) << 2);
            sc[j+4] = (u_j8 & 0xF) | ((u_j & 0x30) >> 0);
            m[j+4]  = (u_j8 >> 4)  | ((u_j & 0xC0) >> 2);
        }
        
        // Compute effective D/M
        float D[8]; float M[8];
        for (int j = 0; j < 8; j++) {
            D[j] = d * (float)sc[j];
            M[j] = dmin * (float)m[j];
        }
        
        // Loop over 8 sub-blocks (32 weights each)
        for (int j = 0; j < 8; j++) {
            float d_val = D[j];
            float m_val = M[j];
            int sub_offset = j * 32;
            int qs_offset = j * 16;
            
            // Loop 16 bytes (32 weights)
            for (int k = 0; k < 16; k++) {
                uchar b = qs[qs_offset + k];
                
                // Low nibble
                float v0 = (float)(b & 0xF);
                float w0 = d_val * (v0 - 8.0f) - m_val;
                
                // High nibble
                float v1 = (float)(b >> 4);
                float w1 = d_val * (v1 - 8.0f) - m_val;
                
                // Dot product
                // Input index: i*256 + sub_offset + k*2
                int idx = i * 256 + sub_offset + k * 2;
                sum += w0 * (float)in_ptr[idx];
                sum += w1 * (float)in_ptr[idx + 1];
            }
        }
    }
    
    sum = simd_sum(sum); if (lane_id == 0) output[row] = safe_half(sum);
}

// Fixed RMSNorm: Single Threadgroup per Row, Out-of-Place
// Fixed RMSNorm for Large Dimensions (e.g. 4096)
// Uses 1 Threadgroup per Row. Max 1024 threads.
kernel void rmsnorm_f16(device const half *x [[ buffer(0) ]],
                      device half *out [[ buffer(1) ]],
                      device const half *w [[ buffer(2) ]],
                      constant float &eps [[ buffer(3) ]],
                      constant int &cols [[ buffer(4) ]],
                      uint tid [[ thread_index_in_threadgroup ]],
                      uint qid [[ thread_position_in_grid ]],
                      uint gid [[ threadgroup_position_in_grid ]]) {
    threadgroup float s[1024]; 
    
    // 1. Accumulate Sum of Squares (Strided Loop)
    float sum = 0.0f;
    int row_offset = gid * cols; // Row Start
    
    for (int i = tid; i < cols; i += 1024) {
        float val = (float)x[row_offset + i];
        sum += val * val;
    }
    s[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // 2. Reduction (Simple Thread 0 Loop for safety)
    if (tid == 0) { 
        float t = 0; 
        int active_threads = (cols < 1024) ? cols : 1024;
        for (int i = 0; i < active_threads; i++) t += s[i]; 
        s[0] = 1.0f / sqrt(t / (float)cols + eps); 
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // 3. Normalize and Write (Strided Loop)
    float scale = s[0];
    for (int i = tid; i < cols; i += 1024) {
        int idx = row_offset + i;
        out[idx] = (half)((float)x[idx] * scale * (float)w[i]);
    }
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
    float g = (float)gate[qid]; 
    float val = (float)up[qid] * (g / (1.0f + exp(-g)));
    gate[qid] = safe_half(val);
}

// Hardcode Embed debug
kernel void embedding_f16(device const half *weight [[ buffer(0) ]], device half *output [[ buffer(1) ]], constant int &idx [[ buffer(2) ]], constant int &cols [[ buffer(3) ]], uint qid [[ thread_position_in_grid ]]) {
    output[qid] = weight[idx * (uint)cols + qid];
}

kernel void add_f16(device half *x [[ buffer(0) ]], device const half *y [[ buffer(1) ]], uint qid [[ thread_position_in_grid ]]) {
    float res = (float)x[qid] + (float)y[qid];
    x[qid] = safe_half(res);
}

kernel void copy_f16(device const half *src [[ buffer(0) ]], device half *dst [[ buffer(1) ]], uint qid [[ thread_position_in_grid ]]) {
    dst[qid] = src[qid];
}

// Q3_K dot product (32 threads per row)
kernel void linear_q3k_f16(device const uchar *weight [[ buffer(0) ]],
                         device const half *input [[ buffer(1) ]],
                         device half *output [[ buffer(2) ]],
                         constant int &dim_in [[ buffer(3) ]],
                         constant int &dim_out [[ buffer(4) ]],
                         uint2 tid [[ thread_position_in_threadgroup ]],
                         uint2 qid [[ thread_position_in_grid ]]) {
    uint row = qid.y; if (row >= (uint)dim_out) return; uint lane_id = tid.x;
    
    // Q3_K Block Size = 256. SizeBytes = 110.
    int num_blocks = dim_in / 256;
    
    device const uchar *row_ptr = weight + row * num_blocks * 110;
    device const half *in_ptr = input;
    
    float sum = 0;
    
    // Loop over super-blocks
    for (int i = (int)lane_id; i < num_blocks; i += 32) {
        device const uchar *block = row_ptr + i * 110;
        
        // Layout:
        // hmask: 32 bytes
        // qs: 64 bytes
        // scales: 12 bytes
        // d: 2 bytes (f16)
        
        device const uchar *hmask = block;
        device const uchar *qs = block + 32;
        device const uchar *scales = block + 96;
        float d = (float)*(device const half*)(block + 108);

        // DEBUG: Force d=1.0 for testing if needed
        // d = 1.0f;
        
        // Decode 16 scales (12 bytes)
        uchar sc[16];
        for (int j = 0; j < 4; j++) {
            uchar s0 = scales[j];
            uchar s1 = scales[j+4];
            uchar s2 = scales[j+8];
            
            sc[j]    = s0 & 63;
            sc[j+4]  = s1 & 63;
            sc[j+8]  = s2 & 63;
            sc[j+12] = (s0 >> 6) | ((s1 >> 6) << 2) | ((s2 >> 6) << 4);
        }
        
        // Loop over 16 sub-blocks (16 weights each)
        for (int l = 0; l < 16; l++) {
            float s = d * ((float)sc[l] - 32.0f);
            
            // Loop 16 weights
            // qs byte has 4 weights.
            // hmask byte has 8 weights.
            
            int sub_offset = l * 16;
            for (int k = 0; k < 16; k++) {
                int idxInBlock = sub_offset + k;
                
                // QS: 2 bits
                // Byte = idxInBlock / 4
                // Shift = (idxInBlock % 4) * 2
                uchar qsByte = qs[idxInBlock / 4];
                uchar q2 = (qsByte >> ((idxInBlock % 4) * 2)) & 3;
                
                // HMASK: 1 bit
                // Byte = idxInBlock / 8
                // Shift = idxInBlock % 8
                uchar hmByte = hmask[idxInBlock / 8];
                uchar h = (hmByte >> (idxInBlock % 8)) & 1;
                
                int q = (h << 2) | q2;
                float w = s * ((float)q - 4.0f);
                
                // Input index
                int idx = i * 256 + idxInBlock;
                sum += w * (float)in_ptr[idx];
                // sum += w * (float)in_ptr[idx];
            }
        }
    }
    
    // Original reduction
    sum = simd_sum(sum); if (lane_id == 0) output[row] = (half)sum;
}
