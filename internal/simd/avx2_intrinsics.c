#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

// AVX2 Softmax implementation
// Process 8 floats at a time
void softmax_avx2(float* x, int n) {
    if (n <= 0) return;
    
    // Find max for numerical stability
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    
    // Compute exp(x[i] - max) and sum
    float sum = 0.0f;
    int i = 0;
    
    // Process 8 elements at a time
    __m256 v_max = _mm256_set1_ps(max_val);
    __m256 v_sum = _mm256_setzero_ps();
    
    for (; i <= n - 8; i += 8) {
        __m256 v = _mm256_loadu_ps(&x[i]);
        v = _mm256_sub_ps(v, v_max);
        
        // Approximate exp using fast method
        // exp(x) = 2^(x / ln(2))
        // Using polynomial approximation for 2^x
        __m256 x_div_ln2 = _mm256_mul_ps(v, _mm256_set1_ps(1.44269504f));
        __m256 fx = _mm256_floor_ps(x_div_ln2);
        __m256 y = _mm256_sub_ps(x_div_ln2, fx);
        
        // Polynomial: 1 + y * (0.693147 + y * (0.240153 + y * 0.055828))
        __m256 c1 = _mm256_set1_ps(0.693147f);
        __m256 c2 = _mm256_set1_ps(0.240153f);
        __m256 c3 = _mm256_set1_ps(0.055828f);
        
        __m256 result = _mm256_set1_ps(1.0f);
        result = _mm256_add_ps(result, _mm256_mul_ps(y, c1));
        y = _mm256_mul_ps(y, y);
        result = _mm256_add_ps(result, _mm256_mul_ps(y, c2));
        y = _mm256_mul_ps(y, y);
        result = _mm256_add_ps(result, _mm256_mul_ps(y, c3));
        
        // Multiply by 2^fx
        __m256i shift = _mm256_cvtps_epi32(fx);
        __m256i exp = _mm256_slli_epi32(_mm256_set1_epi32(1), shift);
        __m256 scale = _mm256_cvtepi32_ps(exp);
        result = _mm256_mul_ps(result, scale);
        
        _mm256_storeu_ps(&x[i], result);
        v_sum = _mm256_add_ps(v_sum, result);
    }
    
    // Horizontal sum of v_sum
    __m128 vsum_low = _mm256_castps256_ps128(v_sum);
    __m128 vsum_high = _mm256_extractf128_ps(v_sum, 1);
    vsum_low = _mm_add_ps(vsum_low, vsum_high);
    vsum_low = _mm_hadd_ps(vsum_low, vsum_low);
    vsum_low = _mm_hadd_ps(vsum_low, vsum_low);
    sum = _mm_cvtss_f32(vsum_low);
    
    // Process remaining elements
    for (; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    
    // Normalize
    if (sum > 0.0f) {
        float inv_sum = 1.0f / sum;
        __m256 v_inv_sum = _mm256_set1_ps(inv_sum);
        
        i = 0;
        for (; i <= n - 8; i += 8) {
            __m256 v = _mm256_loadu_ps(&x[i]);
            v = _mm256_mul_ps(v, v_inv_sum);
            _mm256_storeu_ps(&x[i], v);
        }
        
        for (; i < n; i++) {
            x[i] *= inv_sum;
        }
    }
}

// AVX2 SwiGLU activation
// SwiGLU(x) = x * sigmoid(x) where sigmoid(x) = 1 / (1 + exp(-x))
void swiglu_avx2(const float* gate, const float* up, float* out, int n) {
    int i = 0;
    
    for (; i <= n - 8; i += 8) {
        __m256 g = _mm256_loadu_ps(&gate[i]);
        __m256 u = _mm256_loadu_ps(&up[i]);
        
        // Clamp to [-10, 10] for numerical stability
        g = _mm256_max_ps(g, _mm256_set1_ps(-10.0f));
        g = _mm256_min_ps(g, _mm256_set1_ps(10.0f));
        
        // Compute sigmoid: 1 / (1 + exp(-x))
        __m256 neg_g = _mm256_sub_ps(_mm256_setzero_ps(), g);
        
        // Fast exp approximation for negative values
        __m256 x_div_ln2 = _mm256_mul_ps(neg_g, _mm256_set1_ps(1.44269504f));
        __m256 fx = _mm256_floor_ps(x_div_ln2);
        __m256 y = _mm256_sub_ps(x_div_ln2, fx);
        
        __m256 c1 = _mm256_set1_ps(0.693147f);
        __m256 c2 = _mm256_set1_ps(0.240153f);
        __m256 c3 = _mm256_set1_ps(0.055828f);
        
        __m256 exp_result = _mm256_set1_ps(1.0f);
        exp_result = _mm256_add_ps(exp_result, _mm256_mul_ps(y, c1));
        y = _mm256_mul_ps(y, y);
        exp_result = _mm256_add_ps(exp_result, _mm256_mul_ps(y, c2));
        y = _mm256_mul_ps(y, y);
        exp_result = _mm256_add_ps(exp_result, _mm256_mul_ps(y, c3));
        
        __m256i shift = _mm256_cvtps_epi32(fx);
        __m256i exp_int = _mm256_slli_epi32(_mm256_set1_epi32(1), shift);
        __m256 scale = _mm256_cvtepi32_ps(exp_int);
        exp_result = _mm256_mul_ps(exp_result, scale);
        
        // sigmoid = 1 / (1 + exp(-x))
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 sigmoid = _mm256_div_ps(one, _mm256_add_ps(one, exp_result));
        
        // SwiGLU: gate * sigmoid(gate) * up
        __m256 result = _mm256_mul_ps(g, sigmoid);
        result = _mm256_mul_ps(result, u);
        
        _mm256_storeu_ps(&out[i], result);
    }
    
    // Process remaining elements
    for (; i < n; i++) {
        float g = gate[i];
        if (g > 10.0f) g = 10.0f;
        if (g < -10.0f) g = -10.0f;
        float sigmoid = 1.0f / (1.0f + expf(-g));
        out[i] = g * sigmoid * up[i];
    }
}

// AVX2 FP16 to FP32 conversion
// Process 16 values at a time (256 bits / 16 bits = 16 values)
void fp16_to_fp32_avx2(const uint16_t* src, float* dst, int n) {
    int i = 0;
    
    for (; i <= n - 16; i += 16) {
        // Load 16 FP16 values
        __m256i v16 = _mm256_loadu_si256((__m256i*)&src[i]);
        
        // Convert to FP32 (process 8 at a time in two halves)
        __m128i v16_low = _mm256_castsi256_si128(v16);
        __m128i v16_high = _mm256_extracti128_si256(v16, 1);
        
        __m256 v32_low = _mm256_cvtph_ps(v16_low);
        __m256 v32_high = _mm256_cvtph_ps(v16_high);
        
        _mm256_storeu_ps(&dst[i], v32_low);
        _mm256_storeu_ps(&dst[i + 8], v32_high);
    }
    
    // Process remaining elements
    for (; i < n; i++) {
        uint16_t h = src[i];
        uint32_t sign = (h >> 15) & 0x1;
        uint32_t exp = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;
        
        uint32_t f32;
        if (exp == 0) {
            if (mant == 0) {
                f32 = sign << 31;
            } else {
                int shift = 0;
                while ((mant & 0x400) == 0) {
                    mant <<= 1;
                    shift++;
                }
                mant = (mant & 0x3FF) << 13;
                uint32_t new_exp = 127 - 14 - shift;
                f32 = (sign << 31) | (new_exp << 23) | mant;
            }
        } else if (exp == 31) {
            if (mant == 0) {
                f32 = (sign << 31) | 0x7F800000;
            } else {
                f32 = (sign << 31) | 0x7F800000 | (mant << 13);
            }
        } else {
            uint32_t new_exp = exp - 15 + 127;
            f32 = (sign << 31) | (new_exp << 23) | (mant << 13);
        }
        
        memcpy(&dst[i], &f32, sizeof(float));
    }
}

// AVX2 FP32 to FP16 conversion
void fp32_to_fp16_avx2(const float* src, uint16_t* dst, int n) {
    int i = 0;
    
    for (; i <= n - 16; i += 16) {
        // Load 16 FP32 values
        __m256 v32_0 = _mm256_loadu_ps(&src[i]);
        __m256 v32_1 = _mm256_loadu_ps(&src[i + 8]);
        
        // Convert to FP16
        __m128i v16_0 = _mm256_cvtps_ph(v32_0, _MM_FROUND_TO_NEAREST_INT);
        __m128i v16_1 = _mm256_cvtps_ph(v32_1, _MM_FROUND_TO_NEAREST_INT);
        
        // Pack into single 256-bit register
        __m256i v16 = _mm256_set_m128i(v16_1, v16_0);
        
        _mm256_storeu_si256((__m256i*)&dst[i], v16);
    }
    
    // Process remaining elements
    for (; i < n; i++) {
        float f = src[i];
        uint32_t bits;
        memcpy(&bits, &f, sizeof(uint32_t));
        
        uint32_t sign = bits >> 31;
        uint32_t exp = (bits >> 23) & 0xFF;
        uint32_t mant = bits & 0x7FFFFF;
        
        uint16_t h;
        if (exp == 0) {
            h = 0;
        } else if (exp == 255) {
            h = (uint16_t)(sign << 15) | 0x7C00 | (uint16_t)(mant >> 9);
        } else {
            int new_exp = (int)exp - 127 + 15;
            if (new_exp >= 31) {
                h = (uint16_t)(sign << 15) | 0x7C00;
            } else if (new_exp <= 0) {
                uint32_t shift = (uint32_t)(1 - new_exp);
                uint32_t m = mant | 0x800000;
                h = (uint16_t)(sign << 15) | (uint16_t)(m >> (9 + shift));
            } else {
                h = (uint16_t)(sign << 15) | (uint16_t)(new_exp << 10) | (uint16_t)(mant >> 13);
            }
        }
        dst[i] = h;
    }
}
