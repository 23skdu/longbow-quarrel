//go:build amd64 && cgo

#pragma GCC target("avx2,f16c,fma,no-avx512f,no-avx512vl,no-avx512bw,no-avx512dq")
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

static inline float hsum_avx2(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    vlow = _mm_hadd_ps(vlow, vlow);
    vlow = _mm_hadd_ps(vlow, vlow);
    return _mm_cvtss_f32(vlow);
}

// Fast exponential approximation using 8-wide AVX2 instructions
static inline __m256 fast_exp_avx2(__m256 x) {
    x = _mm256_min_ps(x, _mm256_set1_ps(88.0f));
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.0f));
    __m256 ln2 = _mm256_set1_ps(1.44269504f);
    __m256 v = _mm256_mul_ps(x, ln2);
    __m256i int_part = _mm256_cvtps_epi32(v);
    __m256 frac = _mm256_sub_ps(v, _mm256_cvtepi32_ps(int_part));
    __m256 c1 = _mm256_set1_ps(0.693147f);
    __m256 c2 = _mm256_set1_ps(0.240153f);
    __m256 c3 = _mm256_set1_ps(0.055828f);
    __m256 result = _mm256_set1_ps(1.0f);
    result = _mm256_add_ps(result, _mm256_mul_ps(frac, c1));
    __m256 frac2 = _mm256_mul_ps(frac, frac);
    result = _mm256_add_ps(result, _mm256_mul_ps(frac2, c2));
    __m256 frac3 = _mm256_mul_ps(frac2, frac);
    result = _mm256_add_ps(result, _mm256_mul_ps(frac3, c3));
    __m256i exp_offset = _mm256_add_epi32(int_part, _mm256_set1_epi32(127));
    exp_offset = _mm256_slli_epi32(exp_offset, 23);
    result = _mm256_mul_ps(result, _mm256_castsi256_ps(exp_offset));
    return result;
}

// AVX2 Softmax implementation
// Process 8 floats at a time
void softmax_avx2(float* x, long n) {
    if (n <= 0) return;
    
    // Find max for numerical stability
    float max_val = x[0];
    for (long i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    if (isnan(max_val) || isinf(max_val)) return;
    
    // Compute exp(x[i] - max) and sum
    float sum = 0.0f;
    long i = 0;
    
    // Process 8 elements at a time
    __m256 v_max = _mm256_set1_ps(max_val);
    __m256 v_sum = _mm256_setzero_ps();
    
    for (; i <= n - 8; i += 8) {
        __m256 v = _mm256_loadu_ps(&x[i]);
        v = _mm256_sub_ps(v, v_max);
        v = fast_exp_avx2(v);
        _mm256_storeu_ps(&x[i], v);
        v_sum = _mm256_add_ps(v_sum, v);
    }
    
    sum = hsum_avx2(v_sum);
    
    // Process remaining elements
    for (; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    
    // Normalize
    if (sum > 0.0f && !isnan(sum) && !isinf(sum)) {
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
void swiglu_avx2(const float* gate, const float* up, float* out, long n) {
    long i = 0;
    
    for (; i <= n - 8; i += 8) {
        __m256 g = _mm256_loadu_ps(&gate[i]);
        __m256 u = _mm256_loadu_ps(&up[i]);
        
        // Clamp to [-10, 10] for numerical stability
        g = _mm256_max_ps(g, _mm256_set1_ps(-10.0f));
        g = _mm256_min_ps(g, _mm256_set1_ps(10.0f));
        
        // Compute sigmoid: 1 / (1 + exp(-x))
        __m256 neg_g = _mm256_sub_ps(_mm256_setzero_ps(), g);
        __m256 exp_neg = fast_exp_avx2(neg_g);
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 sigmoid = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg));
        
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
void fp16_to_fp32_avx2(const uint16_t* src, float* dst, long n) {
    long i = 0;
    
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
void fp32_to_fp16_avx2(const float* src, uint16_t* dst, long n) {
    long i = 0;
    
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

void matmul_avx2(const float* a, const float* b, float* c, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            c[i * n + j] = 0.0f;
        }
        for (int kk = 0; kk < k; kk++) {
            float a_val = a[i * k + kk];
            __m256 va = _mm256_set1_ps(a_val);
            int j = 0;
            for (; j <= n - 8; j += 8) {
                __m256 vc = _mm256_loadu_ps(&c[i * n + j]);
                __m256 vb = _mm256_loadu_ps(&b[kk * n + j]);
                vc = _mm256_fmadd_ps(va, vb, vc);
                _mm256_storeu_ps(&c[i * n + j], vc);
            }
            for (; j < n; j++) {
                c[i * n + j] += a_val * b[kk * n + j];
            }
        }
    }
}

void rope_avx2(float* tensor, const int* posIds, int batch, int heads,
               int seqLen, int headDim, float theta) {
    int half = headDim / 2;
    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < heads; h++) {
            for (int s = 0; s < seqLen; s++) {
                int pos = (posIds && s < seqLen) ? posIds[s] : 0;
                int base = b * heads * seqLen * headDim + h * seqLen * headDim + s * headDim;
                int d = 0;
                for (; d <= half - 8; d += 8) {
                    float cv[8], sv[8];
                    for (int l = 0; l < 8; l++) {
                        float freq = (float)pos / powf(theta, (float)(2 * (d + l)) / (float)headDim);
                        cv[l] = cosf(freq);
                        sv[l] = sinf(freq);
                    }
                    __m256 v_cv = _mm256_loadu_ps(cv);
                    __m256 v_sv = _mm256_loadu_ps(sv);
                    __m256 v_ev = _mm256_loadu_ps(&tensor[base + d]);
                    __m256 v_od = _mm256_loadu_ps(&tensor[base + d + half]);

                    __m256 res_ev = _mm256_fmsub_ps(v_ev, v_cv, _mm256_mul_ps(v_od, v_sv));
                    __m256 res_od = _mm256_fmadd_ps(v_ev, v_sv, _mm256_mul_ps(v_od, v_cv));

                    _mm256_storeu_ps(&tensor[base + d], res_ev);
                    _mm256_storeu_ps(&tensor[base + d + half], res_od);
                }
                for (; d < half; d++) {
                    float freq = (float)pos / powf(theta, (float)(2 * d) / (float)headDim);
                    float cv = cosf(freq), sv = sinf(freq);
                    int ei = base + d, oi = base + d + half;
                    float ev = tensor[ei], od = tensor[oi];
                    tensor[ei] = ev * cv - od * sv;
                    tensor[oi] = ev * sv + od * cv;
                }
            }
        }
    }
}

#define MAX_STATIC_ATTN_WEIGHTS_AVX2 8192
_Thread_local static float tl_attn_weights_avx2[MAX_STATIC_ATTN_WEIGHTS_AVX2];

void fused_attention_avx2(const float* q, const float* k, const float* v,
                          float* output, int batch, int heads, int seqLen,
                          int kvSeqLen, int headDim, float scale) {
    float* weights = (kvSeqLen <= MAX_STATIC_ATTN_WEIGHTS_AVX2) ? tl_attn_weights_avx2 : (float*)malloc(kvSeqLen * sizeof(float));
    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < heads; h++) {
            for (int s = 0; s < seqLen; s++) {
                int offset = b * heads * seqLen * headDim + h * seqLen * headDim + s * headDim;
                float max_val = -INFINITY;
                for (int kv = 0; kv < kvSeqLen; kv++) {
                    int kvOff = b * heads * kvSeqLen * headDim + h * kvSeqLen * headDim + kv * headDim;
                    int d = 0;
                    __m256 v_sum = _mm256_setzero_ps();
                    for (; d <= headDim - 8; d += 8) {
                        __m256 vq = _mm256_loadu_ps(&q[offset + d]);
                        __m256 vk = _mm256_loadu_ps(&k[kvOff + d]);
                        v_sum = _mm256_fmadd_ps(vq, vk, v_sum);
                    }
                    float dot = hsum_avx2(v_sum);
                    for (; d < headDim; d++) dot += q[offset + d] * k[kvOff + d];
                    dot *= scale;
                    if (kv == 0 || dot > max_val) max_val = dot;
                }
                float exp_sum = 0.0f;
                for (int kv = 0; kv < kvSeqLen; kv++) {
                    int kvOff = b * heads * kvSeqLen * headDim + h * kvSeqLen * headDim + kv * headDim;
                    int d = 0;
                    __m256 v_sum = _mm256_setzero_ps();
                    for (; d <= headDim - 8; d += 8) {
                        __m256 vq = _mm256_loadu_ps(&q[offset + d]);
                        __m256 vk = _mm256_loadu_ps(&k[kvOff + d]);
                        v_sum = _mm256_fmadd_ps(vq, vk, v_sum);
                    }
                    float dot = hsum_avx2(v_sum);
                    for (; d < headDim; d++) dot += q[offset + d] * k[kvOff + d];
                    float w = expf(dot * scale - max_val);
                    weights[kv] = w;
                    exp_sum += w;
                }
                float inv_sum = (exp_sum > 0.0f) ? 1.0f / exp_sum : 0.0f;
                for (int kv = 0; kv < kvSeqLen; kv++) {
                    weights[kv] *= inv_sum;
                }
                int d = 0;
                for (; d <= headDim - 8; d += 8) {
                    _mm256_storeu_ps(&output[offset + d], _mm256_setzero_ps());
                }
                for (; d < headDim; d++) {
                    output[offset + d] = 0.0f;
                }
                for (int kv = 0; kv < kvSeqLen; kv++) {
                    int kvOff = b * heads * kvSeqLen * headDim + h * kvSeqLen * headDim + kv * headDim;
                    float w = weights[kv];
                    __m256 vw = _mm256_set1_ps(w);
                    int dd = 0;
                    for (; dd <= headDim - 8; dd += 8) {
                        __m256 vo = _mm256_loadu_ps(&output[offset + dd]);
                        __m256 vv = _mm256_loadu_ps(&v[kvOff + dd]);
                        vo = _mm256_fmadd_ps(vw, vv, vo);
                        _mm256_storeu_ps(&output[offset + dd], vo);
                    }
                    for (; dd < headDim; dd++) {
                        output[offset + dd] += w * v[kvOff + dd];
                    }
                }
            }
        }
    }
    if (weights != tl_attn_weights_avx2) {
        free(weights);
    }
}

#define MAX_STATIC_HIDDEN_AVX2 32768
_Thread_local static float tl_temp_avx2[MAX_STATIC_HIDDEN_AVX2];

void fused_mlp_avx2(const float* input, const float* gateWeight,
                    const float* upWeight, const float* downWeight,
                    float* output, int batch, int dim, int hiddenDim) {
    float* temp = (hiddenDim <= MAX_STATIC_HIDDEN_AVX2) ? tl_temp_avx2 : (float*)malloc(hiddenDim * sizeof(float));
    if (!temp) return;
    for (int b = 0; b < batch; b++) {
        int inOff = b * dim;
        int h = 0;
        for (; h <= hiddenDim - 8; h += 8) {
            __m256 g = _mm256_loadu_ps(&gateWeight[h]);
            g = _mm256_max_ps(g, _mm256_set1_ps(-10.0f));
            g = _mm256_min_ps(g, _mm256_set1_ps(10.0f));
            float g_arr[8];
            _mm256_storeu_ps(g_arr, g);
            float sig_arr[8];
            for (int l = 0; l < 8; l++) {
                sig_arr[l] = 1.0f / (1.0f + expf(-g_arr[l]));
            }
            __m256 sig = _mm256_loadu_ps(sig_arr);
            __m256 up = _mm256_loadu_ps(&upWeight[h]);
            __m256 res = _mm256_mul_ps(up, g);
            res = _mm256_mul_ps(res, sig);
            _mm256_storeu_ps(&temp[h], res);
        }
        for (; h < hiddenDim; h++) {
            float gv = gateWeight[h];
            if (gv > 10.0f) gv = 10.0f;
            if (gv < -10.0f) gv = -10.0f;
            temp[h] = upWeight[h] * gv * (1.0f / (1.0f + expf(-gv)));
        }
        int d = 0;
        for (; d <= dim - 8; d += 8) {
            __m256 out = _mm256_setzero_ps();
            for (int hh = 0; hh < hiddenDim; hh++) {
                __m256 vt = _mm256_set1_ps(temp[hh]);
                __m256 vw = _mm256_loadu_ps(&downWeight[hh * dim + d]);
                out = _mm256_fmadd_ps(vt, vw, out);
            }
            _mm256_storeu_ps(&output[inOff + d], out);
        }
        for (; d < dim; d++) {
            float sum = 0.0f;
            for (int hh = 0; hh < hiddenDim; hh++) sum += temp[hh] * downWeight[hh * dim + d];
            output[inOff + d] = sum;
        }
    }
    if (temp != tl_temp_avx2) free(temp);
}

