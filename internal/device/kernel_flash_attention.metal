#include <metal_stdlib>
using namespace metal;

#define BLOCK_SIZE 16 // Tokens per KV block in paged cache
#define BC 32         // KV Tile Size
#define BR 32         // Q Tile Size (for prefill, usually 1 for decode)

// Metal_FlashAttention2_F16
// This kernel implements the tiled FlashAttention-2 algorithm with paged KV cache support.
// Optimization: Fused online softmax + tiled matrix multiplication in threadgroup memory.
kernel void flash_attention2_f16(
    device const half *q [[ buffer(0) ]],
    device const half *k_cache [[ buffer(1) ]],
    device const half *v_cache [[ buffer(2) ]],
    device half *output [[ buffer(3) ]],
    constant int &num_heads [[ buffer(4) ]],
    constant int &kv_heads [[ buffer(5) ]],
    constant int &headDim [[ buffer(6) ]],
    constant int &seq_len [[ buffer(7) ]],
    constant int &kv_block_size [[ buffer(8) ]], // logical block size (16)
    device const int *block_table [[ buffer(9) ]],
    constant int &max_blocks_per_seq [[ buffer(10) ]],
    uint3 tg_pos [[ threadgroup_position_in_grid ]],
    uint3 t_pos [[ thread_position_in_threadgroup ]],
    uint3 nt [[ threads_per_threadgroup ]])
{
    uint head_id = tg_pos.x;
    uint batch_id = tg_pos.y;
    uint q_pos = tg_pos.z; // For prefill, we might dispatch multiple Q positions

    if (head_id >= (uint)num_heads) return;

    // Indexing helpers
    uint kv_head_id = head_id / (num_heads / kv_heads);
    float scale = 1.0f / sqrt((float)headDim);

    // Threadgroup Memory: Tiles for Q, K, V
    // For BR=1 (Decoding), we only need 1 query vector.
    threadgroup float s_q[128]; 
    threadgroup float s_o[128]; // Accumulator for O
    
    // Online Softmax State
    float m_i = -1e20f;
    float l_i = 0.0f;

    // Initialize output accumulator
    if (t_pos.x < (uint)headDim) {
        s_o[t_pos.x] = 0.0f;
        // Prefetch Query
        s_q[t_pos.x] = (float)q[batch_id * num_heads * headDim + head_id * headDim + t_pos.x];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Loop over logical blocks of KV cache
    uint num_logical_blocks = (seq_len + kv_block_size - 1) / kv_block_size;
    device const int *seq_block_table = block_table + batch_id * max_blocks_per_seq;

    for (uint b = 0; b < num_logical_blocks; b++) {
        int physical_block_id = seq_block_table[b];
        if (physical_block_id < 0) break;

        // Iterate tokens within this physical block
        uint tokens_in_block = min((uint)kv_block_size, (uint)seq_len - b * kv_block_size);
        
        for (uint t = 0; t < tokens_in_block; t++) {
            // 1. Compute Score S = Q @ K^T
            // Data layout: [physical_block, tokens, kv_heads, headDim]
            uint k_offset = physical_block_id * kv_block_size * kv_heads * headDim + 
                            t * kv_heads * headDim + 
                            kv_head_id * headDim;
            
            float s_ij = 0.0f;
            // Dot product Q[head] * K[head][t]
            // For max performance, we use simd_sum
            if (t_pos.x < (uint)headDim) {
                s_ij = s_q[t_pos.x] * (float)k_cache[k_offset + t_pos.x];
            }
            s_ij = simd_sum(s_ij) * scale;

            // 2. Online Softmax update
            // We only need thread 0 of each simdgroup to update m_i/l_i if we were doing tile-level.
            // But since this is 1 token vs 1 head, we just update.
            float m_prev = m_i;
            m_i = max(m_prev, s_ij);
            float exp_s = exp(s_ij - m_i);
            float alpha = exp(m_prev - m_i);
            l_i = l_i * alpha + exp_s;

            // 3. Update Output Accumulator: O = O * alpha + exp_s * V
            uint v_offset = physical_block_id * kv_block_size * kv_heads * headDim + 
                            t * kv_heads * headDim + 
                            kv_head_id * headDim;
            
            if (t_pos.x < (uint)headDim) {
                float v_val = (float)v_cache[v_offset + t_pos.x];
                s_o[t_pos.x] = s_o[t_pos.x] * alpha + exp_s * v_val;
            }
        }
    }

    // 4. Finalize: O = O / l_i
    if (t_pos.x < (uint)headDim) {
        output[batch_id * num_heads * headDim + head_id * headDim + t_pos.x] = (half)(s_o[t_pos.x] / l_i);
    }
}
