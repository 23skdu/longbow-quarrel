//go:build arm64 && cgo
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

// Helper to find max absolute value across the entire result array
static float horizontal_max_f32(float32x4_t v) {
    return vmaxvq_f32(v);
}

void polar_quant_neon(const float* input, const float* rotation_matrix, int8_t* quantized, float* scale_out, float* residual, int n, int bits) {
    if (n <= 0) return;

    float32x4_t v_max_abs = vdupq_n_f32(0.0f);
    float* rotated = (float*)__builtin_alloca(n * sizeof(float));

    // 1. Matrix-Vector Multiplication: y = R * x
    for (int i = 0; i < n; i++) {
        float32x4_t v_sum = vdupq_n_f32(0.0f);
        int j = 0;
        for (; j <= n - 4; j += 4) {
            float32x4_t v_r = vld1q_f32(&rotation_matrix[i * n + j]);
            float32x4_t v_x = vld1q_f32(&input[j]);
            v_sum = vfmaq_f32(v_sum, v_r, v_x);
        }
        
        float sum = vgetq_lane_f32(v_sum, 0) + vgetq_lane_f32(v_sum, 1) + vgetq_lane_f32(v_sum, 2) + vgetq_lane_f32(v_sum, 3);
        // Process tail
        for (; j < n; j++) {
            sum += rotation_matrix[i * n + j] * input[j];
        }
        
        rotated[i] = sum;
        float32x4_t v_val = vdupq_n_f32(fabsf(sum));
        v_max_abs = vmaxq_f32(v_max_abs, v_val);
    }

    float max_abs = horizontal_max_f32(v_max_abs);
    float max_quant_val = (float)((1 << (bits - 1)) - 1);
    float scale = (max_abs > 0.0f) ? max_abs / max_quant_val : 1.0f;
    *scale_out = scale;
    float inv_scale = 1.0f / scale;

    float32x4_t v_inv_scale = vdupq_n_f32(inv_scale);
    float32x4_t v_scale = vdupq_n_f32(scale);
    float32x4_t v_max_q = vdupq_n_f32(max_quant_val);
    float32x4_t v_min_q = vdupq_n_f32(-max_quant_val);

    float* res_rotated = (float*)__builtin_alloca(n * sizeof(float));

    // 2. Quantize and calculate residual in rotated space
    int i = 0;
    for (; i <= n - 4; i += 4) {
        float32x4_t v_y = vld1q_f32(&rotated[i]);
        float32x4_t v_qf = vmulq_f32(v_y, v_inv_scale);
        
        // Round to nearest (Neon doesn't have a direct round intrinsic in ARMv7, but ARM64 has vrndnq_f32)
#if defined(__aarch64__)
        v_qf = vrndnq_f32(v_qf);
#else
        // Fallback or just round manually
        for(int k=0; k<4; k++) rotated[i+k] = roundf(rotated[i+k] * inv_scale); 
        // Better to just stay in generic code for tail/AVX
#endif
        v_qf = vmaxq_f32(v_qf, v_min_q);
        v_qf = vminq_f32(v_qf, v_max_q);
        
        // Store quantized int8
        quantized[i]   = (int8_t)vgetq_lane_f32(v_qf, 0);
        quantized[i+1] = (int8_t)vgetq_lane_f32(v_qf, 1);
        quantized[i+2] = (int8_t)vgetq_lane_f32(v_qf, 2);
        quantized[i+3] = (int8_t)vgetq_lane_f32(v_qf, 3);
        
        // Residual = y - q * scale
        float32x4_t v_res = vsubq_f32(v_y, vmulq_f32(v_qf, v_scale));
        vst1q_f32(&res_rotated[i], v_res);
    }
    
    for (; i < n; i++) {
        float qf = roundf(rotated[i] * inv_scale);
        if (qf > max_quant_val) qf = max_quant_val;
        else if (qf < -max_quant_val) qf = -max_quant_val;
        quantized[i] = (int8_t)qf;
        res_rotated[i] = rotated[i] - qf * scale;
    }

    // 3. Inverse Rotation: finalRes = R^T * resRotated
    memset(residual, 0, n * sizeof(float));
    for (int j = 0; j < n; j++) {
        float32x4_t v_res_j = vdupq_n_f32(res_rotated[j]);
        int jn = j * n;
        int i = 0;
        for (; i <= n - 4; i += 4) {
            float32x4_t v_r = vld1q_f32(&rotation_matrix[jn + i]);
            float32x4_t v_res = vld1q_f32(&residual[i]);
            v_res = vfmaq_f32(v_res, v_res_j, v_r);
            vst1q_f32(&residual[i], v_res);
        }
        for (; i < n; i++) {
            residual[i] += res_rotated[j] * rotation_matrix[jn + i];
        }
    }
}

void qjl_transform_neon(const float* residual, const float* sign_matrix, int8_t* quantized, float* scale_out, int rows, int cols) {
    if (rows <= 0 || cols <= 0) return;

    float32x4_t v_norm_sq = vdupq_n_f32(0.0f);
    float* projected = (float*)__builtin_alloca(rows * sizeof(float));

    for (int i = 0; i < rows; i++) {
        float32x4_t v_sum = vdupq_n_f32(0.0f);
        int j = 0;
        for (; j <= cols - 4; j += 4) {
            float32x4_t v_s = vld1q_f32(&sign_matrix[i * cols + j]);
            float32x4_t v_r = vld1q_f32(&residual[j]);
            v_sum = vfmaq_f32(v_sum, v_s, v_r);
        }
        
        float sum = vgetq_lane_f32(v_sum, 0) + vgetq_lane_f32(v_sum, 1) + vgetq_lane_f32(v_sum, 2) + vgetq_lane_f32(v_sum, 3);
        for (; j < cols; j++) {
            sum += sign_matrix[i * cols + j] * residual[j];
        }
        
        projected[i] = sum;
        // Correctly accumulate only to the first lane or use a proper horizontal sum later
        float32x4_t v_sq = vdupq_n_f32(0.0f);
        v_sq = vsetq_lane_f32(sum * sum, v_sq, 0);
        v_norm_sq = vaddq_f32(v_norm_sq, v_sq);
    }

    // Horizontal sum of normSq
    float norm_sq = vgetq_lane_f32(v_norm_sq, 0) + vgetq_lane_f32(v_norm_sq, 1) + vgetq_lane_f32(v_norm_sq, 2) + vgetq_lane_f32(v_norm_sq, 3);
    *scale_out = sqrtf(norm_sq / (float)rows);

    for (int i = 0; i < rows; i++) {
        quantized[i] = (projected[i] >= 0.0f) ? 1 : -1;
    }
}
