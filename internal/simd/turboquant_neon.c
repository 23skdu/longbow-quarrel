//go:build arm64 && cgo
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

// Helper to find max absolute value across the entire result array
static float horizontal_max_f32(float32x4_t v) {
    return vmaxvq_f32(v);
}

// round_half_away_neon rounds to nearest with ties going away from zero,
// which is what C's roundf(), Go's math.Round and CUDA's roundf() all do.
//
// vrndnq_f32 rounds ties to EVEN, so 2.5 would become 2 while Go and CUDA
// produce 3. That made the quantized value differ from the GPU path for
// inputs that land exactly on a .5 boundary, and it disagreed with the
// scalar tail of this very function, which uses roundf(). Ties are detected
// and corrected explicitly.
// bitand_f32 applies a bitwise AND to two float vectors. The float
// spelling of the intrinsic - vandq_f32 - is not declared by Apple's
// clang, so the integer form is used and the result reinterpreted.
static inline float32x4_t bitand_f32(float32x4_t a, float32x4_t b) {
    return vreinterpretq_f32_u32(
        vandq_u32(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b)));
}

// bitor_f32 applies a bitwise OR to two float vectors. As with the AND,
// the float spelling of the intrinsic is not declared by Apple's clang.
static inline float32x4_t bitor_f32(float32x4_t a, float32x4_t b) {
    return vreinterpretq_f32_u32(
        vorrq_u32(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b)));
}

static inline float32x4_t round_half_away_neon(float32x4_t v) {
    // The masks are bit patterns, so they are built as integers and
    // reinterpreted. vdupq_n_f32(0x7FFFFFFF) would convert the integer
    // 2147483647 to the float 2147483648.0f, whose bit pattern is
    // 0x4F000000 - not the mask at all.
    const float32x4_t abs_mask = vreinterpretq_f32_u32(vdupq_n_u32(0x7FFFFFFF));
    const float32x4_t sign_mask = vreinterpretq_f32_u32(vdupq_n_u32(0x80000000));

    float32x4_t v_even = vrndnq_f32(v);

    // Truncate toward zero to recover the integral part and the fraction.
    float32x4_t v_trunc = vcvtq_f32_s32(vcvtq_s32_f32(v));
    float32x4_t v_frac = vsubq_f32(v, v_trunc);
    float32x4_t v_absfrac = bitand_f32(v_frac, abs_mask);

    // A tie is a fraction of exactly 0.5.
    uint32x4_t v_is_tie = vceqq_f32(v_absfrac, vdupq_n_f32(0.5f));

    // Away from zero is trunc + sign(v).
    float32x4_t v_sign = bitor_f32(bitand_f32(v, sign_mask), vdupq_n_f32(1.0f));
    float32x4_t v_away = vaddq_f32(v_trunc, v_sign);

    return vbslq_f32(v_is_tie, v_away, v_even);
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

        v_qf = round_half_away_neon(v_qf);

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
