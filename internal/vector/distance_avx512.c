//go:build amd64 && cgo

// AVX-512 distance kernels for internal/vector.
//
// Only the build-tagged Go file distance_avx512.go declares and calls
// these; the plain amd64 build never references them (it still compiles
// this file, exactly like internal/simd/kernels_avx512.c, so the symbol
// table stays consistent but no call site exists). The runtime gates
// them behind vecHasAVX512 — these functions raise SIGILL on a CPU
// without AVX-512, the same reason internal/simd needs the check.
//
// The VPOPCNTDQ popcount kernel (_mm512_popcnt_epi64) additionally
// requires AVX-512 VPOPCNTDQ (-mavx512vpopcntdq, Ice Lake+). The pragma
// target line comes first so the intrinsics exist even when the
// command line carries no AVX-512 flag (the plain amd64 build still
// compiles this file into the package, exactly like
// internal/simd/kernels_avx512.c, and only the tagged Go file calls
// it).

#pragma GCC target("avx512f,avx512bw,avx512dq,avx512vl,fma,avx512vpopcntdq")
#include <immintrin.h>
#include <stdint.h>

static inline float vec_hsum16_avx512(__m512 v) {
    return _mm512_reduce_add_ps(v);
}

static inline float vec_l2_f32_avx512_inline(const float* a, const float* b, int n) {
    __m512 s0 = _mm512_setzero_ps();
    __m512 s1 = _mm512_setzero_ps();
    int i = 0;
    for (; i <= n - 32; i += 32) {
        s0 = _mm512_fmadd_ps(_mm512_sub_ps(_mm512_loadu_ps(&a[i]), _mm512_loadu_ps(&b[i])),
                             _mm512_sub_ps(_mm512_loadu_ps(&a[i]), _mm512_loadu_ps(&b[i])), s0);
        s1 = _mm512_fmadd_ps(_mm512_sub_ps(_mm512_loadu_ps(&a[i + 16]), _mm512_loadu_ps(&b[i + 16])),
                             _mm512_sub_ps(_mm512_loadu_ps(&a[i + 16]), _mm512_loadu_ps(&b[i + 16])), s1);
    }
    for (; i <= n - 16; i += 16) {
        __m512 d = _mm512_sub_ps(_mm512_loadu_ps(&a[i]), _mm512_loadu_ps(&b[i]));
        s0 = _mm512_fmadd_ps(d, d, s0);
    }
    float sum = vec_hsum16_avx512(_mm512_add_ps(s0, s1));
    for (; i < n; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

static inline float vec_dot_f32_avx512_inline(const float* a, const float* b, int n) {
    __m512 s0 = _mm512_setzero_ps();
    __m512 s1 = _mm512_setzero_ps();
    int i = 0;
    for (; i <= n - 32; i += 32) {
        s0 = _mm512_fmadd_ps(_mm512_loadu_ps(&a[i]), _mm512_loadu_ps(&b[i]), s0);
        s1 = _mm512_fmadd_ps(_mm512_loadu_ps(&a[i + 16]), _mm512_loadu_ps(&b[i + 16]), s1);
    }
    for (; i <= n - 16; i += 16) {
        s0 = _mm512_fmadd_ps(_mm512_loadu_ps(&a[i]), _mm512_loadu_ps(&b[i]), s0);
    }
    float sum = vec_hsum16_avx512(_mm512_add_ps(s0, s1));
    for (; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

float vec_l2_f32_avx512(const float* a, const float* b, int n) {
    if (n <= 0) return 0.0f;
    return vec_l2_f32_avx512_inline(a, b, n);
}

float vec_dot_f32_avx512(const float* a, const float* b, int n) {
    if (n <= 0) return 0.0f;
    return vec_dot_f32_avx512_inline(a, b, n);
}

void vec_batch_l2_f32_avx512(const float* q, const float* rows, long rowStride, int dim, int m, float* out) {
    for (int j = 0; j < m; j++) {
        out[j] = vec_l2_f32_avx512_inline(q, rows + (size_t)j * (size_t)rowStride, dim);
    }
}

void vec_batch_dot_f32_avx512(const float* q, const float* rows, long rowStride, int dim, int m, float* out) {
    for (int j = 0; j < m; j++) {
        out[j] = vec_dot_f32_avx512_inline(q, rows + (size_t)j * (size_t)rowStride, dim);
    }
}

// ---- complex128 split (SoA) storage ------------------------------------

static inline double vec_hsum8_pd_avx512(__m512d v) {
    return _mm512_reduce_add_pd(v);
}

static inline double vec_dot_c128_row_avx512(const double* qRe, const double* qIm,
                                             const double* re, const double* im, int n) {
    __m512d s0 = _mm512_setzero_pd();
    __m512d s1 = _mm512_setzero_pd();
    int i = 0;
    for (; i <= n - 8; i += 8) {
        s0 = _mm512_fmadd_pd(_mm512_loadu_pd(&qRe[i]), _mm512_loadu_pd(&re[i]), s0);
        s0 = _mm512_fmadd_pd(_mm512_loadu_pd(&qIm[i]), _mm512_loadu_pd(&im[i]), s0);
    }
    for (; i <= n - 4; i += 4) {
        s1 = _mm512_fmadd_pd(_mm512_loadu_pd(&qRe[i]), _mm512_loadu_pd(&re[i]), s1);
        s1 = _mm512_fmadd_pd(_mm512_loadu_pd(&qIm[i]), _mm512_loadu_pd(&im[i]), s1);
    }
    double sum = vec_hsum8_pd_avx512(_mm512_add_pd(s0, s1));
    for (; i < n; i++) {
        sum += qRe[i] * re[i] + qIm[i] * im[i];
    }
    return sum;
}

static inline double vec_l2_c128_row_avx512(const double* qRe, const double* qIm,
                                            const double* re, const double* im, int n) {
    __m512d s0 = _mm512_setzero_pd();
    __m512d s1 = _mm512_setzero_pd();
    int i = 0;
    for (; i <= n - 8; i += 8) {
        s0 = _mm512_fmadd_pd(_mm512_sub_pd(_mm512_loadu_pd(&qRe[i]), _mm512_loadu_pd(&re[i])),
                             _mm512_sub_pd(_mm512_loadu_pd(&qRe[i]), _mm512_loadu_pd(&re[i])), s0);
        s0 = _mm512_fmadd_pd(_mm512_sub_pd(_mm512_loadu_pd(&qIm[i]), _mm512_loadu_pd(&im[i])),
                             _mm512_sub_pd(_mm512_loadu_pd(&qIm[i]), _mm512_loadu_pd(&im[i])), s0);
    }
    for (; i <= n - 4; i += 4) {
        s1 = _mm512_fmadd_pd(_mm512_sub_pd(_mm512_loadu_pd(&qRe[i]), _mm512_loadu_pd(&re[i])),
                             _mm512_sub_pd(_mm512_loadu_pd(&qRe[i]), _mm512_loadu_pd(&re[i])), s1);
        s1 = _mm512_fmadd_pd(_mm512_sub_pd(_mm512_loadu_pd(&qIm[i]), _mm512_loadu_pd(&im[i])),
                             _mm512_sub_pd(_mm512_loadu_pd(&qIm[i]), _mm512_loadu_pd(&im[i])), s1);
    }
    double sum = vec_hsum8_pd_avx512(_mm512_add_pd(s0, s1));
    for (; i < n; i++) {
        double dr = qRe[i] - re[i];
        double di = qIm[i] - im[i];
        sum += dr * dr + di * di;
    }
    return sum;
}

void vec_batch_dot_c128_soa_avx512(const double* qRe, const double* qIm,
                                   const double* re, const double* im,
                                   int dim, int m, double* out) {
    for (int j = 0; j < m; j++) {
        out[j] = vec_dot_c128_row_avx512(qRe, qIm, re + (long)j * dim, im + (long)j * dim, dim);
    }
}

void vec_batch_l2_c128_soa_avx512(const double* qRe, const double* qIm,
                                  const double* re, const double* im,
                                  int dim, int m, double* out) {
    for (int j = 0; j < m; j++) {
        out[j] = vec_l2_c128_row_avx512(qRe, qIm, re + (long)j * dim, im + (long)j * dim, dim);
    }
}

// ---- TurboQuant with VPOPCNTDQ -----------------------------------------
//
// Identical formula and operation order to the AVX-2 kernel
// vec_turbo_dot_avx2; the only difference is that the QJL Hamming
// correction pops 8 packed 64-bit words per VPOPCNTDQ instruction.

float vec_turbo_dot_avx512(const int8_t* aCodes, const int8_t* bCodes,
                           const uint64_t* aWords, const uint64_t* bWords,
                           int dim, int wordsPerRow, int totalBits,
                           float scaleA, float scaleB,
                           float qjlScaleA, float qjlScaleB) {
    int32_t polarSum = 0;
    int i = 0;
    for (; i <= dim - 32; i += 32) {
        // 32 int8 lanes widen to 32 int16 lanes in one AVX-512BW
        // convert; _mm512_madd_epi16 pairs them into 16 int32 lanes.
        __m256i aL = _mm256_loadu_si256((const __m256i*)(aCodes + i));
        __m256i bL = _mm256_loadu_si256((const __m256i*)(bCodes + i));
        __m512i aW = _mm512_cvtepi8_epi16(aL);
        __m512i bW = _mm512_cvtepi8_epi16(bL);
        __m512i prod = _mm512_madd_epi16(aW, bW);
        polarSum += (int32_t)_mm512_reduce_add_epi32(prod);
    }
    for (; i < dim; i++) {
        polarSum += (int32_t)aCodes[i] * (int32_t)bCodes[i];
    }

    float polarScore = (float)polarSum * (scaleA * scaleB);

    if (totalBits > 0 && qjlScaleA > 0.0f && qjlScaleB > 0.0f) {
        int diffBits = 0;
        int w = 0;
        for (; w <= wordsPerRow - 8; w += 8) {
            __m512i xa = _mm512_loadu_si512((const __m512i*)(aWords + w));
            __m512i xb = _mm512_loadu_si512((const __m512i*)(bWords + w));
            __m512i x = _mm512_xor_si512(xa, xb);
            __m512i pc = _mm512_popcnt_epi64(x);
            diffBits += (int)_mm512_reduce_add_epi64(pc);
        }
        for (; w < wordsPerRow; w++) {
            diffBits += __builtin_popcountll(aWords[w] ^ bWords[w]);
        }
        float qjlCorr = (float)(totalBits - 2 * diffBits) / (float)totalBits;
        float qjlScore = qjlCorr * (qjlScaleA * qjlScaleB);
        return polarScore + qjlScore;
    }
    return polarScore;
}

void vec_batch_turbo_dot_avx512(const int8_t* qCodes, const uint64_t* qWords,
                                float qScale, float qQJLScale,
                                const int8_t* codes, const uint64_t* words,
                                const float* scale, const float* qjlScale,
                                int dim, int wordsPerRow, int totalBits, int m, float* out) {
    for (int j = 0; j < m; j++) {
        out[j] = vec_turbo_dot_avx512(qCodes, codes + (long)j * dim,
                                      qWords, words + (long)j * wordsPerRow,
                                      dim, wordsPerRow, totalBits,
                                      qScale, scale[j], qQJLScale, qjlScale[j]);
    }
}

// ---- multi-query shared scan (BatchSearch) ------------------------------
//
// Parity contract matches vec_mq_*_avx2: out[q * m + r], one streaming
// pass over the row block per query group. AVX-512 keeps four 512-bit
// rows in flight per query group of two.

void vec_mq_dot_f32_avx512(const float* qs, int numQ, const float* rows, long rowStride, int dim, int m, float* out) {
    for (int r = 0; r < m; r++) {
        const float* row = rows + (size_t)r * (size_t)rowStride;
        int q = 0;
        for (; q <= numQ - 4; q += 4) {
            const float* q0 = qs + (long)(q + 0) * dim;
            const float* q1 = qs + (long)(q + 1) * dim;
            const float* q2 = qs + (long)(q + 2) * dim;
            const float* q3 = qs + (long)(q + 3) * dim;
            __m512 s0 = _mm512_setzero_ps();
            __m512 s1 = _mm512_setzero_ps();
            __m512 s2 = _mm512_setzero_ps();
            __m512 s3 = _mm512_setzero_ps();
            int i = 0;
            for (; i <= dim - 16; i += 16) {
                __m512 rv = _mm512_loadu_ps(&row[i]);
                s0 = _mm512_fmadd_ps(_mm512_loadu_ps(&q0[i]), rv, s0);
                s1 = _mm512_fmadd_ps(_mm512_loadu_ps(&q1[i]), rv, s1);
                s2 = _mm512_fmadd_ps(_mm512_loadu_ps(&q2[i]), rv, s2);
                s3 = _mm512_fmadd_ps(_mm512_loadu_ps(&q3[i]), rv, s3);
            }
            float acc0 = vec_hsum16_avx512(s0);
            float acc1 = vec_hsum16_avx512(s1);
            float acc2 = vec_hsum16_avx512(s2);
            float acc3 = vec_hsum16_avx512(s3);
            for (; i < dim; i++) {
                float rv = row[i];
                acc0 += q0[i] * rv;
                acc1 += q1[i] * rv;
                acc2 += q2[i] * rv;
                acc3 += q3[i] * rv;
            }
            out[(long)(q + 0) * m + r] = acc0;
            out[(long)(q + 1) * m + r] = acc1;
            out[(long)(q + 2) * m + r] = acc2;
            out[(long)(q + 3) * m + r] = acc3;
        }
        for (; q < numQ; q++) {
            out[(long)q * m + r] = vec_dot_f32_avx512_inline(qs + (long)q * dim, row, dim);
        }
    }
}

void vec_mq_l2_f32_avx512(const float* qs, int numQ, const float* rows, long rowStride, int dim, int m, float* out) {
    for (int r = 0; r < m; r++) {
        const float* row = rows + (size_t)r * (size_t)rowStride;
        int q = 0;
        for (; q <= numQ - 4; q += 4) {
            const float* q0 = qs + (long)(q + 0) * dim;
            const float* q1 = qs + (long)(q + 1) * dim;
            const float* q2 = qs + (long)(q + 2) * dim;
            const float* q3 = qs + (long)(q + 3) * dim;
            __m512 s0 = _mm512_setzero_ps();
            __m512 s1 = _mm512_setzero_ps();
            __m512 s2 = _mm512_setzero_ps();
            __m512 s3 = _mm512_setzero_ps();
            int i = 0;
            for (; i <= dim - 16; i += 16) {
                __m512 rv = _mm512_loadu_ps(&row[i]);
                __m512 d0 = _mm512_sub_ps(_mm512_loadu_ps(&q0[i]), rv);
                __m512 d1 = _mm512_sub_ps(_mm512_loadu_ps(&q1[i]), rv);
                __m512 d2 = _mm512_sub_ps(_mm512_loadu_ps(&q2[i]), rv);
                __m512 d3 = _mm512_sub_ps(_mm512_loadu_ps(&q3[i]), rv);
                s0 = _mm512_fmadd_ps(d0, d0, s0);
                s1 = _mm512_fmadd_ps(d1, d1, s1);
                s2 = _mm512_fmadd_ps(d2, d2, s2);
                s3 = _mm512_fmadd_ps(d3, d3, s3);
            }
            float acc0 = vec_hsum16_avx512(s0);
            float acc1 = vec_hsum16_avx512(s1);
            float acc2 = vec_hsum16_avx512(s2);
            float acc3 = vec_hsum16_avx512(s3);
            for (; i < dim; i++) {
                float d0 = q0[i] - row[i];
                float d1 = q1[i] - row[i];
                float d2 = q2[i] - row[i];
                float d3 = q3[i] - row[i];
                acc0 += d0 * d0;
                acc1 += d1 * d1;
                acc2 += d2 * d2;
                acc3 += d3 * d3;
            }
            out[(long)(q + 0) * m + r] = acc0;
            out[(long)(q + 1) * m + r] = acc1;
            out[(long)(q + 2) * m + r] = acc2;
            out[(long)(q + 3) * m + r] = acc3;
        }
        for (; q < numQ; q++) {
            out[(long)q * m + r] = vec_l2_f32_avx512_inline(qs + (long)q * dim, row, dim);
        }
    }
}

void vec_mq_dot_c128_soa_avx512(const double* qsRe, const double* qsIm, int numQ,
                                const double* re, const double* im, int dim, int m, float* out) {
    for (int r = 0; r < m; r++) {
        const double* rre = re + (long)r * dim;
        const double* rim = im + (long)r * dim;
        for (int q = 0; q < numQ; q++) {
            out[(long)q * m + r] = (float)vec_dot_c128_row_avx512(
                qsRe + (long)q * dim, qsIm + (long)q * dim, rre, rim, dim);
        }
    }
}

void vec_mq_l2_c128_soa_avx512(const double* qsRe, const double* qsIm, int numQ,
                               const double* re, const double* im, int dim, int m, float* out) {
    for (int r = 0; r < m; r++) {
        const double* rre = re + (long)r * dim;
        const double* rim = im + (long)r * dim;
        for (int q = 0; q < numQ; q++) {
            out[(long)q * m + r] = (float)vec_l2_c128_row_avx512(
                qsRe + (long)q * dim, qsIm + (long)q * dim, rre, rim, dim);
        }
    }
}
