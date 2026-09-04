//go:build amd64 && cgo

#pragma GCC target("avx512f,avx512bw,avx512dq,avx512vl,f16c")
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

// Fast exponential approximation
static inline __m512 fast_exp_avx512(__m512 x) {
    x = _mm512_min_ps(x, _mm512_set1_ps(88.0f));
    x = _mm512_max_ps(x, _mm512_set1_ps(-88.0f));
    __m512 ln2 = _mm512_set1_ps(1.44269504f);
    __m512 v = _mm512_mul_ps(x, ln2);
    __m512i int_part = _mm512_cvtps_epi32(v);
    __m512 frac = _mm512_sub_ps(v, _mm512_cvtepi32_ps(int_part));
    __m512 c1 = _mm512_set1_ps(0.693147f);
    __m512 c2 = _mm512_set1_ps(0.240153f);
    __m512 c3 = _mm512_set1_ps(0.055828f);
    __m512 result = _mm512_set1_ps(1.0f);
    result = _mm512_add_ps(result, _mm512_mul_ps(frac, c1));
    __m512 frac2 = _mm512_mul_ps(frac, frac);
    result = _mm512_add_ps(result, _mm512_mul_ps(frac2, c2));
    __m512 frac3 = _mm512_mul_ps(frac2, frac);
    result = _mm512_add_ps(result, _mm512_mul_ps(frac3, c3));
    __m512i exp_offset = _mm512_slli_epi32(int_part, 23);
    result = _mm512_mul_ps(result, _mm512_castsi512_ps(exp_offset));
    return result;
}

void softmax_avx512(float* x, int n) {
    if (n <= 0) return;
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    __m512 v_max = _mm512_set1_ps(max_val);
    __m512 v_sum = _mm512_setzero_ps();
    int i = 0;
    for (; i <= n - 16; i += 16) {
        __m512 v = _mm512_loadu_ps(&x[i]);
        v = _mm512_sub_ps(v, v_max);
        v = fast_exp_avx512(v);
        _mm512_storeu_ps(&x[i], v);
        v_sum = _mm512_add_ps(v_sum, v);
    }
    float sum = _mm512_reduce_add_ps(v_sum);
    for (; i < n; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    if (sum > 0.0f) {
        float inv_sum = 1.0f / sum;
        __m512 v_inv = _mm512_set1_ps(inv_sum);
        i = 0;
        for (; i <= n - 16; i += 16) {
            __m512 v = _mm512_loadu_ps(&x[i]);
            v = _mm512_mul_ps(v, v_inv);
            _mm512_storeu_ps(&x[i], v);
        }
        for (; i < n; i++) x[i] *= inv_sum;
    }
}

void swiglu_avx512(const float* gate, const float* up, float* out, int n) {
    int i = 0;
    for (; i <= n - 16; i += 16) {
        __m512 g = _mm512_loadu_ps(&gate[i]);
        __m512 u = _mm512_loadu_ps(&up[i]);
        g = _mm512_max_ps(g, _mm512_set1_ps(-10.0f));
        g = _mm512_min_ps(g, _mm512_set1_ps(10.0f));
        __m512 neg_g = _mm512_sub_ps(_mm512_setzero_ps(), g);
        __m512 exp_neg = fast_exp_avx512(neg_g);
        __m512 one = _mm512_set1_ps(1.0f);
        __m512 sigmoid = _mm512_div_ps(one, _mm512_add_ps(one, exp_neg));
        __m512 result = _mm512_mul_ps(g, sigmoid);
        result = _mm512_mul_ps(result, u);
        _mm512_storeu_ps(&out[i], result);
    }
    for (; i < n; i++) {
        float gv = gate[i];
        if (gv > 10.0f) gv = 10.0f; if (gv < -10.0f) gv = -10.0f;
        out[i] = gv * (1.0f / (1.0f + expf(-gv))) * up[i];
    }
}

void rmsnorm_avx512(const float* input, const float* weight, float* output,
                    int rows, int cols, float eps) {
    for (int r = 0; r < rows; r++) {
        int offset = r * cols;
        int c = 0;
        __m512 v_sum = _mm512_setzero_ps();
        for (; c <= cols - 16; c += 16) {
            __m512 v = _mm512_loadu_ps(&input[offset + c]);
            v_sum = _mm512_fmadd_ps(v, v, v_sum);
        }
        float sum = _mm512_reduce_add_ps(v_sum);
        for (; c < cols; c++) { float v = input[offset + c]; sum += v * v; }
        float inv_norm = 1.0f / sqrtf(sum / (float)cols + eps);
        c = 0;
        __m512 v_norm = _mm512_set1_ps(inv_norm);
        for (; c <= cols - 16; c += 16) {
            __m512 v = _mm512_loadu_ps(&input[offset + c]);
            __m512 w = _mm512_loadu_ps(&weight[c]);
            v = _mm512_mul_ps(v, v_norm);
            v = _mm512_mul_ps(v, w);
            _mm512_storeu_ps(&output[offset + c], v);
        }
        for (; c < cols; c++) output[offset + c] = input[offset + c] * inv_norm * weight[c];
    }
}

void matmul_avx512(const float* a, const float* b, float* c, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            c[i * n + j] = 0.0f;
        }
        for (int kk = 0; kk < k; kk++) {
            float a_val = a[i * k + kk];
            __m512 va = _mm512_set1_ps(a_val);
            int j = 0;
            for (; j <= n - 16; j += 16) {
                __m512 vc = _mm512_loadu_ps(&c[i * n + j]);
                __m512 vb = _mm512_loadu_ps(&b[kk * n + j]);
                vc = _mm512_fmadd_ps(va, vb, vc);
                _mm512_storeu_ps(&c[i * n + j], vc);
            }
            for (; j < n; j++) {
                c[i * n + j] += a_val * b[kk * n + j];
            }
        }
    }
}

void rope_avx512(float* tensor, const int* posIds, int batch, int heads,
                 int seqLen, int headDim, float theta) {
    for (int b = 0; b < batch; b++)
        for (int h = 0; h < heads; h++)
            for (int s = 0; s < seqLen; s++) {
                int pos = (posIds && s < seqLen) ? posIds[s] : 0;
                int base = b*heads*seqLen*headDim + h*seqLen*headDim + s*headDim;
                int half = headDim / 2;
                for (int d = 0; d < half; d++) {
                    float freq = (float)pos / powf(theta, (float)(2*d) / (float)headDim);
                    float cv = cosf(freq), sv = sinf(freq);
                    int ei = base + d, oi = base + d + half;
                    float ev = tensor[ei], od = tensor[oi];
                    tensor[ei] = ev*cv - od*sv; tensor[oi] = ev*sv + od*cv;
                }
            }
}

#define MAX_STATIC_ATTN_WEIGHTS_AVX512 8192
_Thread_local static float tl_attn_weights_avx512[MAX_STATIC_ATTN_WEIGHTS_AVX512];

void fused_attention_avx512(const float* q, const float* k, const float* v,
                            float* output, int batch, int heads, int seqLen,
                            int kvSeqLen, int headDim, float scale) {
    float* weights = (kvSeqLen <= MAX_STATIC_ATTN_WEIGHTS_AVX512) ? tl_attn_weights_avx512 : (float*)malloc(kvSeqLen * sizeof(float));
    for (int b = 0; b < batch; b++)
        for (int h = 0; h < heads; h++)
            for (int s = 0; s < seqLen; s++) {
                int offset = b*heads*seqLen*headDim + h*seqLen*headDim + s*headDim;
                float max_val = -INFINITY;
                for (int kv = 0; kv < kvSeqLen; kv++) {
                    int kvOff = b*heads*kvSeqLen*headDim + h*kvSeqLen*headDim + kv*headDim;
                    int d = 0;
                    __m512 v_sum = _mm512_setzero_ps();
                    for (; d <= headDim - 16; d += 16) {
                        __m512 vq = _mm512_loadu_ps(&q[offset + d]);
                        __m512 vk = _mm512_loadu_ps(&k[kvOff + d]);
                        v_sum = _mm512_fmadd_ps(vq, vk, v_sum);
                    }
                    float dot = _mm512_reduce_add_ps(v_sum);
                    for (; d < headDim; d++) dot += q[offset + d] * k[kvOff + d];
                    dot *= scale;
                    if (kv == 0 || dot > max_val) max_val = dot;
                }
                float exp_sum = 0.0f;
                for (int kv = 0; kv < kvSeqLen; kv++) {
                    int kvOff = b*heads*kvSeqLen*headDim + h*kvSeqLen*headDim + kv*headDim;
                    int d = 0;
                    __m512 v_sum = _mm512_setzero_ps();
                    for (; d <= headDim - 16; d += 16) {
                        __m512 vq = _mm512_loadu_ps(&q[offset + d]);
                        __m512 vk = _mm512_loadu_ps(&k[kvOff + d]);
                        v_sum = _mm512_fmadd_ps(vq, vk, v_sum);
                    }
                    float dot = _mm512_reduce_add_ps(v_sum);
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
                for (; d <= headDim - 16; d += 16) {
                    _mm512_storeu_ps(&output[offset + d], _mm512_setzero_ps());
                }
                for (; d < headDim; d++) {
                    output[offset + d] = 0.0f;
                }
                for (int kv = 0; kv < kvSeqLen; kv++) {
                    int kvOff = b*heads*kvSeqLen*headDim + h*kvSeqLen*headDim + kv*headDim;
                    float w = weights[kv];
                    __m512 vw = _mm512_set1_ps(w);
                    int dd = 0;
                    for (; dd <= headDim - 16; dd += 16) {
                        __m512 vo = _mm512_loadu_ps(&output[offset + dd]);
                        __m512 vv = _mm512_loadu_ps(&v[kvOff + dd]);
                        vo = _mm512_fmadd_ps(vw, vv, vo);
                        _mm512_storeu_ps(&output[offset + dd], vo);
                    }
                    for (; dd < headDim; dd++) {
                        output[offset + dd] += w * v[kvOff + dd];
                    }
                }
            }
    if (weights != tl_attn_weights_avx512) {
        free(weights);
    }
}

#define MAX_STATIC_HIDDEN_AVX512 32768
_Thread_local static float tl_temp_avx512[MAX_STATIC_HIDDEN_AVX512];

void fused_mlp_avx512(const float* input, const float* gateWeight,
                      const float* upWeight, const float* downWeight,
                      float* output, int batch, int dim, int hiddenDim) {
    float* temp = (hiddenDim <= MAX_STATIC_HIDDEN_AVX512) ? tl_temp_avx512 : (float*)malloc(hiddenDim * sizeof(float));
    if (!temp) return;
    for (int b = 0; b < batch; b++) {
        int inOff = b * dim;
        int h = 0;
        for (; h <= hiddenDim - 16; h += 16) {
            __m512 g = _mm512_loadu_ps(&gateWeight[h]);
            g = _mm512_max_ps(g, _mm512_set1_ps(-10.0f));
            g = _mm512_min_ps(g, _mm512_set1_ps(10.0f));
            __m512 neg_g = _mm512_sub_ps(_mm512_setzero_ps(), g);
            __m512 sig = _mm512_div_ps(_mm512_set1_ps(1.0f),
                          _mm512_add_ps(_mm512_set1_ps(1.0f), fast_exp_avx512(neg_g)));
            __m512 up = _mm512_loadu_ps(&upWeight[h]);
            __m512 res = _mm512_mul_ps(up, g);
            res = _mm512_mul_ps(res, sig);
            _mm512_storeu_ps(&temp[h], res);
        }
        for (; h < hiddenDim; h++) {
            float gv = gateWeight[h];
            if (gv > 10.0f) gv = 10.0f; if (gv < -10.0f) gv = -10.0f;
            temp[h] = upWeight[h] * gv * (1.0f / (1.0f + expf(-gv)));
        }
        int d = 0;
        for (; d <= dim - 16; d += 16) {
            __m512 out = _mm512_setzero_ps();
            for (int hh = 0; hh < hiddenDim; hh++) {
                __m512 vt = _mm512_set1_ps(temp[hh]);
                __m512 vw = _mm512_loadu_ps(&downWeight[hh * dim + d]);
                out = _mm512_fmadd_ps(vt, vw, out);
            }
            _mm512_storeu_ps(&output[inOff + d], out);
        }
        for (; d < dim; d++) {
            float sum = 0.0f;
            for (int hh = 0; hh < hiddenDim; hh++) sum += temp[hh] * downWeight[hh * dim + d];
            output[inOff + d] = sum;
        }
    }
    if (temp != tl_temp_avx512) free(temp);
}

void fp16_to_fp32_avx512(const uint16_t* src, float* dst, int n) {
    int i = 0;
    for (; i <= n - 16; i += 16) {
        __m256i v16 = _mm256_loadu_si256((__m256i*)&src[i]);
        __m128i lo = _mm256_castsi256_si128(v16), hi = _mm256_extractf128_si256(v16, 1);
        _mm256_storeu_ps(&dst[i], _mm256_cvtph_ps(lo));
        _mm256_storeu_ps(&dst[i+8], _mm256_cvtph_ps(hi));
    }
    for (; i < n; i++) {
        uint16_t h = src[i];
        uint32_t s=(h>>15)&1, e=(h>>10)&0x1F, m=h&0x3FF, f32;
        if (e==0) { if(m==0) f32=s<<31; else{int sh=0;while((m&0x400)==0){m<<=1;sh++;}f32=(s<<31)|((127-14-sh)<<23)|((m&0x3FF)<<13);} }
        else if(e==31) f32=(s<<31)|0x7F800000|(m<<13);
        else f32=(s<<31)|((e-15+127)<<23)|(m<<13);
        memcpy(&dst[i], &f32, 4);
    }
}

void fp32_to_fp16_avx512(const float* src, uint16_t* dst, int n) {
    int i = 0;
    for (; i <= n - 16; i += 16) {
        __m128i lo = _mm256_cvtps_ph(_mm256_loadu_ps(&src[i]), _MM_FROUND_TO_NEAREST_INT);
        __m128i hi = _mm256_cvtps_ph(_mm256_loadu_ps(&src[i+8]), _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i*)&dst[i], _mm256_set_m128i(hi, lo));
    }
    for (; i < n; i++) {
        float f = src[i]; uint32_t bits; memcpy(&bits, &f, 4);
        uint32_t s=bits>>31, e=(bits>>23)&0xFF, m=bits&0x7FFFFF;
        uint16_t h;
        if(e==0) h=0;
        else if(e==255) h=(uint16_t)(s<<15)|0x7C00|(uint16_t)(m>>9);
        else { int ne=(int)e-127+15; if(ne>=31) h=(uint16_t)(s<<15)|0x7C00;
            else if(ne<=0){uint32_t sh=(uint32_t)(1-ne), m2=m|0x800000;h=(uint16_t)(s<<15)|(uint16_t)(m2>>(9+sh));}
            else h=(uint16_t)(s<<15)|(uint16_t)(ne<<10)|(uint16_t)(m>>13); }
        dst[i] = h;
    }
}
