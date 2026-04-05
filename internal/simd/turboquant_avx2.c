//go:build amd64 && cgo
#if defined(__x86_64__) || defined(_M_X64)
#pragma GCC target("avx2,fma")
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#ifdef __x86_64__
#pragma GCC target("avx2,fma")
#endif

static float horizontal_max_f32_avx2(__m256 v) {
    __m128 v_low = _mm256_castps256_ps128(v);
    __m128 v_high = _mm256_extractf128_ps(v, 1);
    v_low = _mm_max_ps(v_low, v_high);
    v_low = _mm_max_ps(v_low, _mm_shuffle_ps(v_low, v_low, _MM_SHUFFLE(1, 0, 3, 2)));
    v_low = _mm_max_ps(v_low, _mm_shuffle_ps(v_low, v_low, _MM_SHUFFLE(0, 1, 0, 1)));
    return _mm_cvtss_f32(v_low);
}

static float horizontal_sum_f32_avx2(__m256 v) {
    __m128 v_low = _mm256_castps256_ps128(v);
    __m128 v_high = _mm256_extractf128_ps(v, 1);
    v_low = _mm_add_ps(v_low, v_high);
    v_low = _mm_hadd_ps(v_low, v_low);
    v_low = _mm_hadd_ps(v_low, v_low);
    return _mm_cvtss_f32(v_low);
}

void polar_quant_avx2(const float* input, const float* rotation_matrix, int8_t* quantized, float* scale_out, float* residual, int n, int bits) {
    if (n <= 0) return;

    __m256 v_max_abs = _mm256_setzero_ps();
    float* rotated = (float*)__builtin_alloca(n * sizeof(float));

    // 1. Matrix-Vector Multiplication: y = R * x
    for (int i = 0; i < n; i++) {
        __m256 v_sum = _mm256_setzero_ps();
        int j = 0;
        for (; j <= n - 8; j += 8) {
            __m256 v_r = _mm256_loadu_ps(&rotation_matrix[i * n + j]);
            __m256 v_x = _mm256_loadu_ps(&input[j]);
            v_sum = _mm256_fmadd_ps(v_r, v_x, v_sum);
        }
        
        float sum = horizontal_sum_f32_avx2(v_sum);
        // Process tail
        for (; j < n; j++) {
            sum += rotation_matrix[i * n + j] * input[j];
        }
        
        rotated[i] = sum;
        __m256 v_val = _mm256_set1_ps(fabsf(sum));
        v_max_abs = _mm256_max_ps(v_max_abs, v_val);
    }

    float max_abs = horizontal_max_f32_avx2(v_max_abs);
    float max_quant_val = (float)((1 << (bits - 1)) - 1);
    float scale = (max_abs > 0.0f) ? max_abs / max_quant_val : 1.0f;
    *scale_out = scale;
    float inv_scale = 1.0f / scale;

    __m256 v_inv_scale = _mm256_set1_ps(inv_scale);
    __m256 v_scale = _mm256_set1_ps(scale);
    __m256 v_max_q = _mm256_set1_ps(max_quant_val);
    __m256 v_min_q = _mm256_set1_ps(-max_quant_val);

    float* res_rotated = (float*)__builtin_alloca(n * sizeof(float));

    // 2. Quantize and calculate residual in rotated space
    int i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 v_y = _mm256_loadu_ps(&rotated[i]);
        __m256 v_qf = _mm256_mul_ps(v_y, v_inv_scale);
        
        // Round to nearest int
        v_qf = _mm256_round_ps(v_qf, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        
        v_qf = _mm256_max_ps(v_qf, v_min_q);
        v_qf = _mm256_min_ps(v_qf, v_max_q);
        
        // Store quantized int8
        float temp_q[8];
        _mm256_storeu_ps(temp_q, v_qf);
        for(int k=0; k<8; k++) {
            quantized[i+k] = (int8_t)temp_q[k];
        }
        
        // Residual = y - q * scale
        __m256 v_res = _mm256_sub_ps(v_y, _mm256_mul_ps(v_qf, v_scale));
        _mm256_storeu_ps(&res_rotated[i], v_res);
    }
    
    for (; i < n; i++) {
        float qf = roundf(rotated[i] * inv_scale);
        if (qf > max_quant_val) qf = max_quant_val;
        else if (qf < -max_quant_val) qf = -max_quant_val;
        quantized[i] = (int8_t)qf;
        res_rotated[i] = rotated[i] - qf * scale;
    }

    // 3. Inverse Rotation: finalRes = R^T * resRotated
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += rotation_matrix[j * n + i] * res_rotated[j];
        }
        residual[i] = sum;
    }
}

void qjl_transform_avx2(const float* residual, const float* sign_matrix, int8_t* quantized, float* scale_out, int rows, int cols) {
    if (rows <= 0 || cols <= 0) return;

    __m256 v_norm_sq = _mm256_setzero_ps();
    float* projected = (float*)__builtin_alloca(rows * sizeof(float));

    for (int i = 0; i < rows; i++) {
        __m256 v_sum = _mm256_setzero_ps();
        int j = 0;
        for (; j <= cols - 8; j += 8) {
            __m256 v_s = _mm256_loadu_ps(&sign_matrix[i * cols + j]);
            __m256 v_r = _mm256_loadu_ps(&residual[j]);
            v_sum = _mm256_fmadd_ps(v_s, v_r, v_sum);
        }
        
        float sum = horizontal_sum_f32_avx2(v_sum);
        for (; j < cols; j++) {
            sum += sign_matrix[i * cols + j] * residual[j];
        }
        
        projected[i] = sum;
        v_norm_sq = _mm256_add_ps(v_norm_sq, _mm256_set1_ps(sum * sum));
    }

    float norm_sq = horizontal_sum_f32_avx2(v_norm_sq);
    *scale_out = sqrtf(norm_sq / (float)rows);

    for (int i = 0; i < rows; i++) {
        quantized[i] = (projected[i] >= 0.0f) ? 1 : -1;
    }
}
#endif

