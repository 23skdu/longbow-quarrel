import re

with open('internal/device/kernels.metal', 'r') as f:
    content = f.read()

# Define the naive kernel to replace
naive_kernel = r"kernel void linear_q4k_f16_f32\(device const uchar \*weight \[\[ buffer\(0\) \]\],.*?\}\)"

# Define the new kernel
optimized_kernel = """kernel void linear_q4k_f16_f32(device const uchar *weight [[ buffer(0) ]],
                                         device const half *input [[ buffer(1) ]],
                                         device float *output [[ buffer(2) ]],
                                         constant int &dim_in [[ buffer(3) ]],
                                         constant int &dim_out [[ buffer(4) ]],
                                         constant float &scale [[ buffer(5) ]],
                                         uint3 tid [[ thread_position_in_threadgroup ]],
                                         uint3 qid [[ thread_position_in_grid ]]) {
    uint row = qid.y; uint batch = qid.z;
    if (row >= (uint)dim_out) return;
    
    int num_blocks = (dim_in + 255) / 256;
    float sum = 0;
    
    threadgroup half shared_in[256];
    
    for (int i = 0; i < num_blocks; i++) {
        // Cooperatively load 256 elements per block
        if (tid.y == 0) {
            #pragma unroll
            for (int l = 0; l < 8; l++) {
                int in_idx = i * 256 + l * 32 + tid.x;
                if (in_idx < dim_in) {
                    shared_in[l * 32 + tid.x] = input[batch * dim_in + in_idx];
                } else {
                    shared_in[l * 32 + tid.x] = (half)0.0f;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        device const uchar *block = weight + (row * num_blocks + i) * 144;
        float d = fp16_to_fp32(*(device const ushort*)(block));
        float dmin = fp16_to_fp32(*(device const ushort*)(block + 2));
        
        device const uchar *scales = block + 4;
        device const uchar *qs = block + 16;
        
        // Each thread processes 8 pairs of weights (16 elements total)
        for (int l = 0; l < 8; l++) {
            uchar sc, m;
            if (l < 4) { sc = scales[l] & 63; m = scales[l + 4] & 63; }
            else { sc = (scales[l+4] & 0xF) | ((scales[l-4] >> 6) << 4); m = (scales[l+4] >> 4) | ((scales[l] >> 6) << 4); }
            
            float d_val = d * scale * (float)sc;
            float m_val = dmin * scale * (float)m;
            
            int qs_offset = l * 16;
            for (int k = 0; k < 16; k++) {
                uchar b = qs[qs_offset + k];
                
                // Pair 0
                float v0 = d_val * (float)(b & 0xF) - m_val;
                sum += v0 * (float)shared_in[l * 32 + k * 2];
                
                // Pair 1
                float v1 = d_val * (float)(b >> 4) - m_val;
                sum += v1 * (float)shared_in[l * 32 + k * 2 + 1];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    sum = simd_sum(sum);
    if (tid.x == 0 && tid.y == 0) output[batch * dim_out + row] = sum;
}"""

# Perform replacement
new_content = re.sub(naive_kernel, optimized_kernel, content, flags=re.DOTALL)

with open('internal/device/kernels.metal', 'w') as f:
    f.write(new_content)
