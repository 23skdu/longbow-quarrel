//go:build amd64 && cgo

// AVX2 / FMA distance kernels for internal/vector.
//
// The #pragma GCC target lines pin each ISA per file so the same object
// assembles under both the default CFLAGS (nothing special) and the
// AVX-512 tag build, mirroring internal/simd/avx2_intrinsics.c.
//
// All kernels must stay within 1e-4 of the scalar reference in
// distance.go (distance_parity_test.go); reduction order differs, so
// partial accumulators are kept to at most 8 independent lanes, which
// is what the reference does too.

#pragma GCC target("avx2,f16c,fma,no-avx512f,no-avx512vl,no-avx512bw,no-avx512dq,popcnt")
#include <immintrin.h>
#include <stdint.h>

static inline float vec_hsum8_avx2(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

static inline float vec_l2_f32_avx2_inline(const float* a, const float* b, int n) {
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 16; i += 16) {
        __m256 da = _mm256_loadu_ps(&a[i]);
        __m256 db = _mm256_loadu_ps(&b[i]);
        __m256 d = _mm256_sub_ps(da, db);
        s0 = _mm256_fmadd_ps(d, d, s0);
        da = _mm256_loadu_ps(&a[i + 8]);
        db = _mm256_loadu_ps(&b[i + 8]);
        d = _mm256_sub_ps(da, db);
        s1 = _mm256_fmadd_ps(d, d, s1);
    }
    for (; i <= n - 8; i += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(&a[i]), _mm256_loadu_ps(&b[i]));
        s0 = _mm256_fmadd_ps(d, d, s0);
    }
    float sum = vec_hsum8_avx2(_mm256_add_ps(s0, s1));
    for (; i < n; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

static inline float vec_dot_f32_avx2_inline(const float* a, const float* b, int n) {
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 16; i += 16) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(&a[i]), _mm256_loadu_ps(&b[i]), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(&a[i + 8]), _mm256_loadu_ps(&b[i + 8]), s1);
    }
    for (; i <= n - 8; i += 8) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(&a[i]), _mm256_loadu_ps(&b[i]), s0);
    }
    float sum = vec_hsum8_avx2(_mm256_add_ps(s0, s1));
    for (; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

float vec_l2_f32_avx2(const float* a, const float* b, int n) {
    if (n <= 0) return 0.0f;
    return vec_l2_f32_avx2_inline(a, b, n);
}

float vec_dot_f32_avx2(const float* a, const float* b, int n) {
    if (n <= 0) return 0.0f;
    return vec_dot_f32_avx2_inline(a, b, n);
}

// Batch scoring over a contiguous row block: row j of the block is
// rows + j*rowStride, and must carry at least dim readable floats.
// One Go pointer crosses the boundary (cgo pointer rules), which is
// also exactly the layout the mmap-backed store (P0-05) streams.
void vec_batch_l2_f32_avx2(const float* q, const float* rows, long rowStride, int dim, int m, float* out) {
    for (int j = 0; j < m; j++) {
        out[j] = vec_l2_f32_avx2_inline(q, rows + (size_t)j * (size_t)rowStride, dim);
    }
}

void vec_batch_dot_f32_avx2(const float* q, const float* rows, long rowStride, int dim, int m, float* out) {
    for (int j = 0; j < m; j++) {
        out[j] = vec_dot_f32_avx2_inline(q, rows + (size_t)j * (size_t)rowStride, dim);
    }
}

// The gather variants score the rows an index list names.
//
// An inverted list is a list of row numbers, not a block of contiguous
// rows, so the caller used to copy every candidate into a buffer before
// it could be scored. The copy is as expensive as the distance: for a
// 250k index with sixteen probes it moves twelve megabytes per query.
// Reading the rows through the indices does the same work without it.
void vec_batch_l2_f32_gather_avx2(const float* q, const float* rows, const int32_t* indices,
                                  long rowStride, int dim, int m, float* out) {
    for (int j = 0; j < m; j++) {
        const float* row = rows + (size_t)indices[j] * (size_t)rowStride;
        out[j] = vec_l2_f32_avx2_inline(q, row, dim);
    }
}

void vec_batch_dot_f32_gather_avx2(const float* q, const float* rows, const int32_t* indices,
                                   long rowStride, int dim, int m, float* out) {
    for (int j = 0; j < m; j++) {
        const float* row = rows + (size_t)indices[j] * (size_t)rowStride;
        out[j] = vec_dot_f32_avx2_inline(q, row, dim);
    }
}

// ---- complex128 split (SoA) storage, float64 math ----------------------
//
// The public complex128 storage interleaves (re, im) pairs, which stops
// the FMA units from working on 4+ elements at a time. These kernels
// take the split reflection of the same data: row i of a stored set is
// re[i*dim .. (i+1)*dim) and im[i*dim .. (i+1)*dim), so every 256-bit
// register covers four complex numbers' real parts with another
// register for the imaginary parts.

static inline double vec_hsum4_pd_avx2(__m256d v) {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    lo = _mm_add_pd(lo, hi);
    return lo[0] + lo[1];
}

static inline double vec_dot_c128_row_avx2(const double* qRe, const double* qIm,
                                           const double* re, const double* im, int n) {
    __m256d s0 = _mm256_setzero_pd();
    __m256d s1 = _mm256_setzero_pd();
    int i = 0;
    for (; i <= n - 8; i += 8) {
        s0 = _mm256_fmadd_pd(_mm256_loadu_pd(&qRe[i]), _mm256_loadu_pd(&re[i]), s0);
        s0 = _mm256_fmadd_pd(_mm256_loadu_pd(&qIm[i]), _mm256_loadu_pd(&im[i]), s0);
        s1 = _mm256_fmadd_pd(_mm256_loadu_pd(&qRe[i + 4]), _mm256_loadu_pd(&re[i + 4]), s1);
        s1 = _mm256_fmadd_pd(_mm256_loadu_pd(&qIm[i + 4]), _mm256_loadu_pd(&im[i + 4]), s1);
    }
    for (; i <= n - 4; i += 4) {
        s0 = _mm256_fmadd_pd(_mm256_loadu_pd(&qRe[i]), _mm256_loadu_pd(&re[i]), s0);
        s0 = _mm256_fmadd_pd(_mm256_loadu_pd(&qIm[i]), _mm256_loadu_pd(&im[i]), s0);
    }
    (void)s1;
    double sum = vec_hsum4_pd_avx2(_mm256_add_pd(s0, s1));
    for (; i < n; i++) {
        sum += qRe[i] * re[i] + qIm[i] * im[i];
    }
    return sum;
}

static inline double vec_l2_c128_row_avx2(const double* qRe, const double* qIm,
                                          const double* re, const double* im, int n) {
    __m256d s0 = _mm256_setzero_pd();
    __m256d s1 = _mm256_setzero_pd();
    int i = 0;
    for (; i <= n - 8; i += 8) {
        __m256d dr0 = _mm256_sub_pd(_mm256_loadu_pd(&qRe[i]), _mm256_loadu_pd(&re[i]));
        __m256d di0 = _mm256_sub_pd(_mm256_loadu_pd(&qIm[i]), _mm256_loadu_pd(&im[i]));
        s0 = _mm256_fmadd_pd(dr0, dr0, s0);
        s0 = _mm256_fmadd_pd(di0, di0, s0);
        __m256d dr1 = _mm256_sub_pd(_mm256_loadu_pd(&qRe[i + 4]), _mm256_loadu_pd(&re[i + 4]));
        __m256d di1 = _mm256_sub_pd(_mm256_loadu_pd(&qIm[i + 4]), _mm256_loadu_pd(&im[i + 4]));
        s1 = _mm256_fmadd_pd(dr1, dr1, s1);
        s1 = _mm256_fmadd_pd(di1, di1, s1);
    }
    for (; i <= n - 4; i += 4) {
        __m256d dr0 = _mm256_sub_pd(_mm256_loadu_pd(&qRe[i]), _mm256_loadu_pd(&re[i]));
        __m256d di0 = _mm256_sub_pd(_mm256_loadu_pd(&qIm[i]), _mm256_loadu_pd(&im[i]));
        s0 = _mm256_fmadd_pd(dr0, dr0, s0);
        s0 = _mm256_fmadd_pd(di0, di0, s0);
    }
    (void)s1;
    double sum = vec_hsum4_pd_avx2(_mm256_add_pd(s0, s1));
    for (; i < n; i++) {
        double dr = qRe[i] - re[i];
        double di = qIm[i] - im[i];
        sum += dr * dr + di * di;
    }
    return sum;
}

// m rows stored at stride dim in re/im; results in out.
void vec_batch_dot_c128_soa_avx2(const double* qRe, const double* qIm,
                                 const double* re, const double* im,
                                 int dim, int m, double* out) {
    for (int j = 0; j < m; j++) {
        out[j] = vec_dot_c128_row_avx2(qRe, qIm, re + (long)j * dim, im + (long)j * dim, dim);
    }
}

void vec_batch_l2_c128_soa_avx2(const double* qRe, const double* qIm,
                                const double* re, const double* im,
                                int dim, int m, double* out) {
    for (int j = 0; j < m; j++) {
        out[j] = vec_l2_c128_row_avx2(qRe, qIm, re + (long)j * dim, im + (long)j * dim, dim);
    }
}

// ---- TurboQuant asymmetric distance ------------------------------------
//
// dot(A, B) = scaleA*scaleB * sum(codesA * codesB)
//           + (totalBits - 2*hamming)/totalBits * (qjlScaleA*qjlScaleB)
//
// The integer code dot widens both signed operands to 16-bit lanes and
// accumulates with _mm256_madd_epi16; the QJL correction is a 64-bit
// popcount of the XOR of the packed words (the POPCNT ISA is enabled by
// the pragma target above).
//
// The result of the dot product of two TurboQuant vectors written as
// one function so the caller cannot elide the popcount correction:
//
//   score = (float)polar * (scaleA*scaleB) + qjlCorr * (qjlA*qjlB)
//   qjlCorr = (float)(totalBits - 2*diff) / (float)totalBits
//
// wordsPerRow is the packed stride of both word arrays; totalBits is
// the number of valid QJL bits (the product DotTurboQuant derives from
// the byte length), which may be smaller than wordsPerRow*64.
float vec_turbo_dot_avx2(const int8_t* aCodes, const int8_t* bCodes,
                         const uint64_t* aWords, const uint64_t* bWords,
                         int dim, int wordsPerRow, int totalBits,
                         float scaleA, float scaleB,
                         float qjlScaleA, float qjlScaleB) {
    int32_t polarSum = 0;
    int i = 0;
    for (; i <= dim - 32; i += 32) {
        for (int h = 0; h < 2; h++) {
            __m128i aHalf = _mm_loadu_si128((const __m128i*)(aCodes + i + 16 * h));
            __m128i bHalf = _mm_loadu_si128((const __m128i*)(bCodes + i + 16 * h));
            __m256i aW = _mm256_cvtepi8_epi16(aHalf);
            __m256i bW = _mm256_cvtepi8_epi16(bHalf);
            __m256i prod = _mm256_madd_epi16(aW, bW);
            polarSum += (int32_t)_mm256_extract_epi32(prod, 0) + _mm256_extract_epi32(prod, 1) +
                        _mm256_extract_epi32(prod, 2) + _mm256_extract_epi32(prod, 3) +
                        _mm256_extract_epi32(prod, 4) + _mm256_extract_epi32(prod, 5) +
                        _mm256_extract_epi32(prod, 6) + _mm256_extract_epi32(prod, 7);
        }
    }
    for (; i < dim; i++) {
        polarSum += (int32_t)aCodes[i] * (int32_t)bCodes[i];
    }

    float polarScore = (float)polarSum * (scaleA * scaleB);

    if (totalBits > 0 && qjlScaleA > 0.0f && qjlScaleB > 0.0f) {
        int diffBits = 0;
        for (int w = 0; w < wordsPerRow; w++) {
            diffBits += __builtin_popcountll(aWords[w] ^ bWords[w]);
        }
        float qjlCorr = (float)(totalBits - 2 * diffBits) / (float)totalBits;
        float qjlScore = qjlCorr * (qjlScaleA * qjlScaleB);
        return polarScore + qjlScore;
    }
    return polarScore;
}

// Batch form over packed TurboQuant columns: codes and words are
// contiguous over m rows (codes row j at codes + j*dim, words row j at
// words + j*wordsPerRow); scale, qjlScale point at m-length arrays.
void vec_batch_turbo_dot_avx2(const int8_t* qCodes, const uint64_t* qWords,
                              float qScale, float qQJLScale,
                              const int8_t* codes, const uint64_t* words,
                              const float* scale, const float* qjlScale,
                              int dim, int wordsPerRow, int totalBits, int m, float* out) {
    for (int j = 0; j < m; j++) {
        out[j] = vec_turbo_dot_avx2(qCodes, codes + (long)j * dim,
                                    qWords, words + (long)j * wordsPerRow,
                                    dim, wordsPerRow, totalBits,
                                    qScale, scale[j], qQJLScale, qjlScale[j]);
    }
}

// ---- multi-query shared scan (BatchSearch) ------------------------------
//
// BatchSearch scores numQ queries against one streaming pass of the row
// block. The row block is walked once per query group of four; with the
// block resident in L1/L2 that keeps the DRAM traffic at one full sweep
// of the rows while every query keeps a private FMA accumulator chain.
// out[q * m + r].

void vec_mq_dot_f32_avx2(const float* qs, int numQ, const float* rows, long rowStride, int dim, int m, float* out) {
    for (int r = 0; r < m; r++) {
        const float* row = rows + (size_t)r * (size_t)rowStride;
        int q = 0;
        for (; q <= numQ - 4; q += 4) {
            const float* q0 = qs + (long)(q + 0) * dim;
            const float* q1 = qs + (long)(q + 1) * dim;
            const float* q2 = qs + (long)(q + 2) * dim;
            const float* q3 = qs + (long)(q + 3) * dim;
            __m256 s0 = _mm256_setzero_ps();
            __m256 s1 = _mm256_setzero_ps();
            __m256 s2 = _mm256_setzero_ps();
            __m256 s3 = _mm256_setzero_ps();
            int i = 0;
            for (; i <= dim - 8; i += 8) {
                __m256 rv = _mm256_loadu_ps(&row[i]);
                s0 = _mm256_fmadd_ps(_mm256_loadu_ps(&q0[i]), rv, s0);
                s1 = _mm256_fmadd_ps(_mm256_loadu_ps(&q1[i]), rv, s1);
                s2 = _mm256_fmadd_ps(_mm256_loadu_ps(&q2[i]), rv, s2);
                s3 = _mm256_fmadd_ps(_mm256_loadu_ps(&q3[i]), rv, s3);
            }
            float acc0 = vec_hsum8_avx2(s0);
            float acc1 = vec_hsum8_avx2(s1);
            float acc2 = vec_hsum8_avx2(s2);
            float acc3 = vec_hsum8_avx2(s3);
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
            out[(long)q * m + r] = vec_dot_f32_avx2_inline(qs + (long)q * dim, row, dim);
        }
    }
}

void vec_mq_l2_f32_avx2(const float* qs, int numQ, const float* rows, long rowStride, int dim, int m, float* out) {
    for (int r = 0; r < m; r++) {
        const float* row = rows + (size_t)r * (size_t)rowStride;
        int q = 0;
        for (; q <= numQ - 4; q += 4) {
            const float* q0 = qs + (long)(q + 0) * dim;
            const float* q1 = qs + (long)(q + 1) * dim;
            const float* q2 = qs + (long)(q + 2) * dim;
            const float* q3 = qs + (long)(q + 3) * dim;
            __m256 s0 = _mm256_setzero_ps();
            __m256 s1 = _mm256_setzero_ps();
            __m256 s2 = _mm256_setzero_ps();
            __m256 s3 = _mm256_setzero_ps();
            int i = 0;
            for (; i <= dim - 8; i += 8) {
                __m256 rv = _mm256_loadu_ps(&row[i]);
                s0 = _mm256_fmadd_ps(_mm256_sub_ps(_mm256_loadu_ps(&q0[i]), rv), _mm256_sub_ps(_mm256_loadu_ps(&q0[i]), rv), s0);
                s1 = _mm256_fmadd_ps(_mm256_sub_ps(_mm256_loadu_ps(&q1[i]), rv), _mm256_sub_ps(_mm256_loadu_ps(&q1[i]), rv), s1);
                s2 = _mm256_fmadd_ps(_mm256_sub_ps(_mm256_loadu_ps(&q2[i]), rv), _mm256_sub_ps(_mm256_loadu_ps(&q2[i]), rv), s2);
                s3 = _mm256_fmadd_ps(_mm256_sub_ps(_mm256_loadu_ps(&q3[i]), rv), _mm256_sub_ps(_mm256_loadu_ps(&q3[i]), rv), s3);
            }
            float acc0 = vec_hsum8_avx2(s0);
            float acc1 = vec_hsum8_avx2(s1);
            float acc2 = vec_hsum8_avx2(s2);
            float acc3 = vec_hsum8_avx2(s3);
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
            out[(long)q * m + r] = vec_l2_f32_avx2_inline(qs + (long)q * dim, row, dim);
        }
    }
}

// Complex SoA multi-query: re/im point at the first row of the block
// (row r at re + r*dim). Accumulation stays float64; out is float32 so
// the caller feeds heaps without a second conversion pass.
void vec_mq_dot_c128_soa_avx2(const double* qsRe, const double* qsIm, int numQ,
                              const double* re, const double* im, int dim, int m, float* out) {
    for (int r = 0; r < m; r++) {
        const double* rre = re + (long)r * dim;
        const double* rim = im + (long)r * dim;
        for (int q = 0; q < numQ; q++) {
            out[(long)q * m + r] = (float)vec_dot_c128_row_avx2(
                qsRe + (long)q * dim, qsIm + (long)q * dim, rre, rim, dim);
        }
    }
}

void vec_mq_l2_c128_soa_avx2(const double* qsRe, const double* qsIm, int numQ,
                             const double* re, const double* im, int dim, int m, float* out) {
    for (int r = 0; r < m; r++) {
        const double* rre = re + (long)r * dim;
        const double* rim = im + (long)r * dim;
        for (int q = 0; q < numQ; q++) {
            out[(long)q * m + r] = (float)vec_l2_c128_row_avx2(
                qsRe + (long)q * dim, qsIm + (long)q * dim, rre, rim, dim);
        }
    }
}
