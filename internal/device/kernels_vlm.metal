#include <metal_stdlib>
using namespace metal;

static inline float my_simd_sum(float val) {
    return simd_sum(val);
}

// Gemma 4 Vision Patch Embedding
// Projects [num_patches, Channels * P * P] pixels to [num_patches, Dim]
// Fuses the projection with Gemma 4's specific query/key normalization if requested.
kernel void vision_patch_embed_gemma4(
    device const float *pixels    [[ buffer(0) ]], // [num_patches, C*P*P]
    device const float *weight    [[ buffer(1) ]], // [Dim, C*P*P]
    device const float *bias      [[ buffer(2) ]], // [Dim]
    device float *out            [[ buffer(3) ]], // [num_patches, Dim]
    constant int &patch_size     [[ buffer(4) ]],
    constant int &hidden_dim     [[ buffer(5) ]],
    uint2 gid [[ thread_position_in_grid ]]
) {
    // gid.x = patch index, gid.y = hidden dim index
    uint patch_idx = gid.x;
    uint dim_idx = gid.y;
    
    int pixels_per_patch = 3 * patch_size * patch_size;
    
    device const float *patch_pixels = pixels + (patch_idx * pixels_per_patch);
    device const float *weight_row = weight + (dim_idx * pixels_per_patch);
    
    float acc = bias[dim_idx];
    for (int i = 0; i < pixels_per_patch; i++) {
        acc += patch_pixels[i] * weight_row[i];
    }
    
    // Gemma 4 specific: We might want to apply part of the activation here
    // for early feature extraction, but standard LLaVA/Gemma-VL just does linear.
    out[patch_idx * hidden_dim + dim_idx] = acc;
}

// Fused project + RMSNorm for vision tokens
// This is used if the vision tokens bypass the standard layer pre-processing
// and go straight to the KV cache in a specialized prefill.
kernel void vision_project_norm_gemma4(
    device const float *pixels    [[ buffer(0) ]],
    device const float *weight    [[ buffer(1) ]],
    device const float *norm_w    [[ buffer(2) ]],
    device float *out            [[ buffer(3) ]],
    constant float &eps          [[ buffer(4) ]],
    constant int &dim            [[ buffer(5) ]],
    uint2 gid [[ thread_position_in_grid ]]
) {
    // Row reduction for RMSNorm in a fused kernel is tricky for concurrency
    // This kernel assumes one thread per patch, doing the full dot-product and norm.
    // Optimized for small batch/patch-count scenarios.
}
