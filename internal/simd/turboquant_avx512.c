//go:build amd64 && cgo
#if defined(__x86_64__) || defined(_M_X64)
#pragma GCC target("avx512f,avx512bw,fma")
#ifdef __x86_64__
#pragma GCC target("avx512f,avx512bw,avx512dq,avx512vl")
#endif
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

void polar_quant_avx512(const float* input, const float* rotation_matrix, int8_t* quantized, float* scale_out, float* residual, int n, int bits) {
    if (n <= 0) return;

    __m512 v_max_abs = _mm512_setzero_ps();
    float* rotated = (float*)__builtin_alloca(n * sizeof(float));

    // 1. Matrix-Vector Multiplication: y = R * x
    for (int i = 0; i < n; i++) {
        __m512 v_sum = _mm512_setzero_ps();
        int j = 0;
        for (; j <= n - 16; j += 16) {
            __m512 v_r = _mm512_loadu_ps(&rotation_matrix[i * n + j]);
            __m512 v_x = _mm512_loadu_ps(&input[j]);
            v_sum = _mm512_fmadd_ps(v_r, v_x, v_sum);
        }
        
        float sum = _mm512_reduce_add_ps(v_sum);
        // Process tail
        for (; j < n; j++) {
            sum += rotation_matrix[i * n + j] * input[j];
        }
        
        rotated[i] = sum;
        __m512 v_val = _mm512_set1_ps(fabsf(sum));
        v_max_abs = _mm512_max_ps(v_max_abs, v_val);
    }

    float max_abs = _mm512_reduce_max_ps(v_max_abs);
    float max_quant_val = (float)((1 << (bits - 1)) - 1);
    float scale = (max_abs > 0.0f) ? max_abs / max_quant_val : 1.0f;
    *scale_out = scale;
    float inv_scale = 1.0f / scale;

    __m512 v_inv_scale = _mm512_set1_ps(inv_scale);
    __m512 v_scale = _mm512_set1_ps(scale);
    __m512 v_max_q = _mm512_set1_ps(max_quant_val);
    __m512 v_min_q = _mm512_set1_ps(-max_quant_val);

    float* res_rotated = (float*)__builtin_alloca(n * sizeof(float));

    // 2. Quantize and calculate residual in rotated space
    int i = 0;
    for (; i <= n - 16; i += 16) {
        __m512 v_y = _mm512_loadu_ps(&rotated[i]);
        __m512 v_qf = _mm512_mul_ps(v_y, v_inv_scale);
        
        v_qf = _mm512_roundscale_ps(v_qf, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        
        v_qf = _mm512_max_ps(v_qf, v_min_q);
        v_qf = _mm512_min_ps(v_qf, v_max_q);
        
        // Store quantized int8
        float temp_q[16];
        _mm512_storeu_ps(temp_q, v_qf);
        for(int k=0; k<16; k++) {
            quantized[i+k] = (int8_t)temp_q[k];
        }
        
        // Residual = y - q * scale
        __m512 v_res = _mm512_sub_ps(v_y, _mm512_mul_ps(v_qf, v_scale));
        _mm512_storeu_ps(&res_rotated[i], v_res);
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
        __m512 v_res_j = _mm512_set1_ps(res_rotated[j]);
        int jn = j * n;
        int i = 0;
        for (; i <= n - 16; i += 16) {
            __m512 v_r = _mm512_loadu_ps(&rotation_matrix[jn + i]);
            __m512 v_res = _mm512_loadu_ps(&residual[i]);
            v_res = _mm512_fmadd_ps(v_res_j, v_r, v_res);
            _mm512_storeu_ps(&residual[i], v_res);
        }
        for (; i < n; i++) {
            residual[i] += res_rotated[j] * rotation_matrix[jn + i];
        }
    }
}

void qjl_transform_avx512(const float* residual, const float* sign_matrix, int8_t* quantized, float* scale_out, int rows, int cols) {
    if (rows <= 0 || cols <= 0) return;

    float norm_sq = 0.0f;
    float* projected = (float*)__builtin_alloca(rows * sizeof(float));

    for (int i = 0; i < rows; i++) {
        __m512 v_sum = _mm512_setzero_ps();
        int j = 0;
        int icols = i * cols;
        for (; j <= cols - 16; j += 16) {
            __m512 v_s = _mm512_loadu_ps(&sign_matrix[icols + j]);
            __m512 v_r = _mm512_loadu_ps(&residual[j]);
            v_sum = _mm512_fmadd_ps(v_s, v_r, v_sum);
        }
        
        float sum = _mm512_reduce_add_ps(v_sum);
        for (; j < cols; j++) {
            sum += sign_matrix[icols + j] * residual[j];
        }
        
        projected[i] = sum;
        norm_sq += sum * sum;
    }

    *scale_out = sqrtf(norm_sq / (float)rows);

    for (int i = 0; i < rows; i++) {
        quantized[i] = (projected[i] >= 0.0f) ? 1 : -1;
    }
}
#endif

