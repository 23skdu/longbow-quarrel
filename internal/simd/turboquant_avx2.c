//go:build amd64 && cgo

#if defined(__x86_64__) || defined(_M_X64)
// The no-avx512* options are required. A `#pragma GCC target` ADDS to the
// command line options rather than replacing them, and the -tags avx512
// build adds -mavx512f (and friends) package-wide via #cgo CFLAGS. Without
// these exclusions this file is compiled with AVX-512 enabled and GCC emits
// EVEX-encoded instructions, which raise SIGILL on a CPU that has AVX2 but
// not AVX-512 -- exactly the CPU that would call this kernel.
#pragma GCC target("avx2,fma,no-avx512f,no-avx512vl,no-avx512bw,no-avx512dq")
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static float horizontal_sum_avx2(__m256 v) {
    __m128 v_low = _mm256_castps256_ps128(v);
    __m128 v_high = _mm256_extractf128_ps(v, 1);
    v_low = _mm_add_ps(v_low, v_high);
    v_low = _mm_hadd_ps(v_low, v_low);
    v_low = _mm_hadd_ps(v_low, v_low);
    return _mm_cvtss_f32(v_low);
}

static float horizontal_max_avx2(__m256 v) {
    __m128 v_low = _mm256_castps256_ps128(v);
    __m128 v_high = _mm256_extractf128_ps(v, 1);
    v_low = _mm_max_ps(v_low, v_high);
    v_low = _mm_max_ps(v_low, _mm_shuffle_ps(v_low, v_low, _MM_SHUFFLE(1, 0, 3, 2)));
    v_low = _mm_max_ps(v_low, _mm_shuffle_ps(v_low, v_low, _MM_SHUFFLE(0, 1, 0, 1)));
    return _mm_cvtss_f32(v_low);
}

// round_half_away_avx2 rounds to nearest with ties going away from zero,
// which is what C's roundf(), Go's math.Round and CUDA's roundf() all do.
//
// _mm256_round_ps with _MM_FROUND_TO_NEAREST_INT rounds ties to EVEN, so
// 2.5 would become 2 while Go and CUDA produce 3. That made the quantized
// value differ from the GPU path for inputs that land exactly on a .5
// boundary, and it disagreed with the scalar tail of this very function,
// which uses roundf(). Ties are detected and corrected explicitly.
static inline __m256 round_half_away_avx2(__m256 v) {
    const __m256i abs_mask = _mm256_set1_epi32(0x7FFFFFFF);
    const __m256i sign_mask = _mm256_set1_epi32((int)0x80000000);

    __m256 v_even = _mm256_round_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    // Truncate toward zero to recover the integral part and the fraction.
    __m256 v_trunc = _mm256_cvtepi32_ps(_mm256_cvttps_epi32(v));
    __m256 v_frac = _mm256_sub_ps(v, v_trunc);
    __m256 v_absfrac = _mm256_and_ps(v_frac, _mm256_castsi256_ps(abs_mask));

    // A tie is a fraction of exactly 0.5.
    __m256 v_is_tie = _mm256_cmp_ps(v_absfrac, _mm256_set1_ps(0.5f), _CMP_EQ_OQ);

    // Away from zero is trunc + sign(v).
    __m256 v_sign = _mm256_or_ps(_mm256_and_ps(v, _mm256_castsi256_ps(sign_mask)),
                                _mm256_set1_ps(1.0f));
    __m256 v_away = _mm256_add_ps(v_trunc, v_sign);

    return _mm256_blendv_ps(v_even, v_away, v_is_tie);
}

void polar_quant_avx2(const float* input, const float* rotation_matrix, int8_t* quantized, float* scale_out, float* residual, int n, int bits) {
    if (n <= 0) return;

    __m256 v_max_abs = _mm256_setzero_ps();
    float* rotated = (float*)malloc(n * sizeof(float));
    if (!rotated) return;

    for (int i = 0; i < n; i++) {
        __m256 v_sum = _mm256_setzero_ps();
        int j = 0;
        for (; j <= n - 8; j += 8) {
            __m256 v_r = _mm256_loadu_ps(&rotation_matrix[i * n + j]);
            __m256 v_x = _mm256_loadu_ps(&input[j]);
            v_sum = _mm256_fmadd_ps(v_r, v_x, v_sum);
        }
        
        float sum = horizontal_sum_avx2(v_sum);
        for (; j < n; j++) {
            sum += rotation_matrix[i * n + j] * input[j];
        }
        
        rotated[i] = sum;
        __m256 v_val = _mm256_set1_ps(fabsf(sum));
        v_max_abs = _mm256_max_ps(v_max_abs, v_val);
    }

    float max_abs = horizontal_max_avx2(v_max_abs);
    float max_quant_val = (float)((1 << (bits - 1)) - 1);
    float scale = (max_abs > 0.0f) ? max_abs / max_quant_val : 1.0f;
    *scale_out = scale;
    float inv_scale = 1.0f / scale;

    __m256 v_inv_scale = _mm256_set1_ps(inv_scale);
    __m256 v_scale = _mm256_set1_ps(scale);
    __m256 v_max_q = _mm256_set1_ps(max_quant_val);
    __m256 v_min_q = _mm256_set1_ps(-max_quant_val);

    float* res_rotated = (float*)malloc(n * sizeof(float));
    if (!res_rotated) {
        free(rotated);
        return;
    }

    int i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 v_y = _mm256_loadu_ps(&rotated[i]);
        __m256 v_qf = _mm256_mul_ps(v_y, v_inv_scale);
        v_qf = round_half_away_avx2(v_qf);
        v_qf = _mm256_max_ps(v_qf, v_min_q);
        v_qf = _mm256_min_ps(v_qf, v_max_q);
        
        float temp_q[8];
        _mm256_storeu_ps(temp_q, v_qf);
        for(int k=0; k<8; k++) {
            quantized[i+k] = (int8_t)temp_q[k];
        }
        
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

    memset(residual, 0, n * sizeof(float));
    for (int j = 0; j < n; j++) {
        __m256 v_res_j = _mm256_set1_ps(res_rotated[j]);
        int jn = j * n;
        int i = 0;
        for (; i <= n - 8; i += 8) {
            __m256 v_r = _mm256_loadu_ps(&rotation_matrix[jn + i]);
            __m256 v_res = _mm256_loadu_ps(&residual[i]);
            v_res = _mm256_fmadd_ps(v_res_j, v_r, v_res);
            _mm256_storeu_ps(&residual[i], v_res);
        }
        for (; i < n; i++) {
            residual[i] += res_rotated[j] * rotation_matrix[jn + i];
        }
    }
    free(rotated);
    free(res_rotated);
}

void qjl_transform_avx2(const float* residual, const float* sign_matrix, int8_t* quantized, float* scale_out, int rows, int cols) {
    if (rows <= 0 || cols <= 0) return;

    float norm_sq = 0.0f;
    float* projected = (float*)malloc(rows * sizeof(float));
    if (!projected) return;

    for (int i = 0; i < rows; i++) {
        __m256 v_sum = _mm256_setzero_ps();
        int j = 0;
        for (; j <= cols - 8; j += 8) {
            __m256 v_s = _mm256_loadu_ps(&sign_matrix[i * cols + j]);
            __m256 v_r = _mm256_loadu_ps(&residual[j]);
            v_sum = _mm256_fmadd_ps(v_s, v_r, v_sum);
        }
        
        float sum = horizontal_sum_avx2(v_sum);
        for (; j < cols; j++) {
            sum += sign_matrix[i * cols + j] * residual[j];
        }
        
        projected[i] = sum;
        norm_sq += sum * sum;
    }

    *scale_out = sqrtf(norm_sq / (float)rows);

    for (int i = 0; i < rows; i++) {
        quantized[i] = (projected[i] >= 0.0f) ? 1 : -1;
    }
    free(projected);
}
#endif