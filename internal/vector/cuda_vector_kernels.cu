// cuda_vector_kernels.cu implements the GPU batch kernels of
// internal/vector (P0-01 batch inverted-list scoring, P0-02 complex
// SoA batch path, P0-04 mini-batch k-means). Compiled by nvcc into
// libvector_cuda_kernels.a (see the Makefile target of the same name)
// and linked by vector_cuda.go under the `cuda` build tag.
//
// Device-side contracts (mirroring distance_dispatch.go):
//
//   - Multi-query f32: rows row-major, row j at rows[j*stride], dim
//     floats scored per row; qs carries numQ back-to-back queries; one
//     kernel writes out[q*count + r] for every (query, row) pair. Rows
//     are tiled into shared memory (tileRows chosen at launch), so the
//     batch streams the row set once per tile instead of once per
//     query — that is the shared-scan property P0-01/P0-04 want.
//   - Complex SoA: the same shape over split float32 planes (the fp64
//     CPU reference trades precision for bandwidth on GeForce cards;
//     parity is tested to 1e-4, see vector_cuda_test.go).
//   - IVF list scan: one launch covers every probed list. ids holds
//     the concatenated list ids, offsets their exclusive prefix sums,
//     probeLists/probeBase the per-launch list indices and output base
//     offsets. Scores land at probeBase[p] + row-within-list.
//   - Top-k resolves on the device, best score first (heap order of
//     the CPU reference), ties broken toward the smaller row id to
//     match heapAdd's first-wins semantics.
//   - k-means assignment accumulates atomically into sums/counts; the
//     centroid update stays on the CPU so both paths share arithmetic

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stddef.h>

// ---------------------------------------------------------------------
// Score primitives
// ---------------------------------------------------------------------

__device__ __forceinline__ float vecRowDotF32(const float* __restrict__ a,
                                              const float* __restrict__ b,
                                              int n) {
    float acc = 0.0f;
    int i = 0;
    for (; i <= n - 4; i += 4) {
        acc = fmaf(a[i], b[i], acc);
        acc = fmaf(a[i + 1], b[i + 1], acc);
        acc = fmaf(a[i + 2], b[i + 2], acc);
        acc = fmaf(a[i + 3], b[i + 3], acc);
    }
    for (; i < n; i++) {
        acc = fmaf(a[i], b[i], acc);
    }
    return acc;
}

__device__ __forceinline__ float vecRowL2F32(const float* __restrict__ a,
                                             const float* __restrict__ b,
                                             int n) {
    float acc = 0.0f;
    int i = 0;
    for (; i <= n - 4; i += 4) {
        float d0 = a[i] - b[i];
        float d1 = a[i + 1] - b[i + 1];
        float d2 = a[i + 2] - b[i + 2];
        float d3 = a[i + 3] - b[i + 3];
        acc = fmaf(d0, d0, acc);
        acc = fmaf(d1, d1, acc);
        acc = fmaf(d2, d2, acc);
        acc = fmaf(d3, d3, acc);
    }
    for (; i < n; i++) {
        float d = a[i] - b[i];
        acc = fmaf(d, d, acc);
    }
    return acc;
}

// ---------------------------------------------------------------------
// Multi-query shared-scan, float32 rows (flat P0-01/P0-02 and IVF)
// ---------------------------------------------------------------------

__global__ void vecMqScanF32Kernel(const float* __restrict__ qs, int numQ,
                                   const float* __restrict__ rows, long stride,
                                   int dim, int count, int tileRows, int l2,
                                   float* __restrict__ out) {
    extern __shared__ float smem[];
    const int row0 = blockIdx.x * tileRows;
    if (row0 >= count) {
        return;
    }
    // Cooperative load of the tile rows into shared memory.
    for (int r = 0; r < tileRows; ++r) {
        const int gr = row0 + r;
        if (gr >= count) {
            break;
        }
        const float* src = rows + (size_t)gr * (size_t)stride;
        float* dst = smem + (size_t)r * (size_t)dim;
        for (int j = threadIdx.x; j < dim; j += blockDim.x) {
            dst[j] = src[j];
        }
    }
    __syncthreads();
    const int nrows = (count - row0) < tileRows ? (count - row0) : tileRows;
    const int pairs = tileRows * numQ;
    for (int p = threadIdx.x; p < pairs; p += blockDim.x) {
        const int q = p / tileRows;
        const int r = p % tileRows;
        if (r >= nrows) {
            continue;
        }
        const float* qr = qs + (size_t)q * (size_t)dim;
        const float* rr = smem + (size_t)r * (size_t)dim;
        float acc = l2 ? vecRowL2F32(qr, rr, dim) : vecRowDotF32(qr, rr, dim);
        out[(size_t)q * (size_t)count + (size_t)(row0 + r)] = acc;
    }
}

static int vecLaunchMqScanF32(const float* qs, int numQ, const float* rows,
                              long stride, int dim, int count, int l2,
                              float* out) {
    if (!qs || !rows || !out || numQ <= 0 || count <= 0 || dim <= 0 ||
        stride < dim || (size_t)count * (size_t)stride > (size_t)1 << 38) {
        return 1;
    }
    // Shared-memory budget: the default dynamic carve-out is 48 KB.
    int tileRows = 12288 / dim;
    if (tileRows > 16) {
        tileRows = 16;
    }
    if (tileRows < 1) {
        return 2; // dim too wide for the shared-scan kernel
    }
    dim3 block(256);
    dim3 grid((count + tileRows - 1) / tileRows);
    size_t shared = (size_t)tileRows * (size_t)dim * sizeof(float);
    vecMqScanF32Kernel<<<grid, block, shared>>>(qs, numQ, rows, stride, dim,
                                                count, tileRows, l2, out);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 3;
    }
    return 0;
}

// Complex SoA shared scan: device planes are float32 (half the fp64
// bandwidth; parity vs the fp64 CPU reference within 1e-4). Both planes
// share one dynamic allocation: [tileRows][dim] re then im.
__global__ void vecMqScanCSoAKernel(const float* __restrict__ qsRe,
                                    const float* __restrict__ qsIm, int numQ,
                                    const float* __restrict__ re,
                                    const float* __restrict__ im, int dim,
                                    int count, int tileRows, int l2,
                                    float* __restrict__ out) {
    extern __shared__ float smem[]; // [2][tileRows][dim]: re then im
    const int row0 = blockIdx.x * tileRows;
    if (row0 >= count) {
        return;
    }
    float* simPlane = smem + (size_t)tileRows * (size_t)dim;
    for (int r = 0; r < tileRows; ++r) {
        const int gr = row0 + r;
        if (gr >= count) {
            break;
        }
        float* dstRe = smem + (size_t)r * (size_t)dim;
        float* dstIm = simPlane + (size_t)r * (size_t)dim;
        for (int j = threadIdx.x; j < dim; j += blockDim.x) {
            dstRe[j] = (float)re[(size_t)gr * (size_t)dim + (size_t)j];
            dstIm[j] = (float)im[(size_t)gr * (size_t)dim + (size_t)j];
        }
    }
    __syncthreads();
    const int nrows = (count - row0) < tileRows ? (count - row0) : tileRows;
    const int pairs = tileRows * numQ;
    for (int p = threadIdx.x; p < pairs; p += blockDim.x) {
        const int q = p / tileRows;
        const int r = p % tileRows;
        if (r >= nrows) {
            continue;
        }
        const float* qr = qsRe + (size_t)q * (size_t)dim;
        const float* qi = qsIm + (size_t)q * (size_t)dim;
        const float* rr = smem + (size_t)r * (size_t)dim;
        const float* ii = simPlane + (size_t)r * (size_t)dim;
        float score;
        if (l2) {
            float acc = 0.0f;
            for (int t = 0; t < dim; t++) {
                float d1 = qr[t] - rr[t];
                float d2 = qi[t] - ii[t];
                acc = fmaf(d1, d1, acc);
                acc = fmaf(d2, d2, acc);
            }
            score = acc;
        } else {
            float acc = 0.0f;
            float acci = 0.0f;
            int j = 0;
            for (; j <= dim - 4; j += 4) {
                acc = fmaf(qr[j], rr[j], acc);
                acci = fmaf(qi[j], ii[j], acci);
                acc = fmaf(qr[j + 1], rr[j + 1], acc);
                acci = fmaf(qi[j + 1], ii[j + 1], acci);
                acc = fmaf(qr[j + 2], rr[j + 2], acc);
                acci = fmaf(qi[j + 2], ii[j + 2], acci);
                acc = fmaf(qr[j + 3], rr[j + 3], acc);
                acci = fmaf(qi[j + 3], ii[j + 3], acci);
            }
            for (; j < dim; j++) {
                acc = fmaf(qr[j], rr[j], acc);
                acci = fmaf(qi[j], ii[j], acci);
            }
            score = acc + acci;
        }
        out[(size_t)q * (size_t)count + (size_t)(row0 + r)] = score;
    }
}

static int vecLaunchMqScanCSoA(const float* qsRe, const float* qsIm, int numQ,
                               const float* re, const float* im, int dim,
                               int count, int l2, float* out) {
    if (!qsRe || !qsIm || !re || !im || !out || numQ <= 0 || count <= 0 ||
        dim <= 0) {
        return 1;
    }
    int half = 6144 / dim; // two planes in 48 KB
    if (half > 16) {
        half = 16;
    }
    if (half < 1) {
        return 2;
    }
    dim3 block(256);
    dim3 grid((count + half - 1) / half);
    size_t shared = (size_t)2 * (size_t)half * (size_t)dim * sizeof(float);
    vecMqScanCSoAKernel<<<grid, block, shared>>>(qsRe, qsIm, numQ, re, im, dim,
                                                 count, half, l2, out);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 3;
    }
    return 0;
}

// ---------------------------------------------------------------------
// IVF list scan: one launch for every probed list (P0-01)
// ---------------------------------------------------------------------

__global__ void vecScanListsKernel(const float* __restrict__ rows, long stride,
                                   int dim, const int* __restrict__ ids,
                                   const long* __restrict__ offsets,
                                   const int* __restrict__ probeLists,
                                   const long* __restrict__ probeBase,
                                   int numProbes, const float* __restrict__ q,
                                   float* __restrict__ scores) {
    const int p = blockIdx.y;
    if (p >= numProbes) {
        return;
    }
    const int list = probeLists[p];
    if (list < 0) {
        return;
    }
    const long base = offsets[list];
    const long len = offsets[list + 1] - base;
    const long rowStart = (long)blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (rowStart >= len) {
        return;
    }
    const int vid = ids[base + rowStart];
    float score = 0.0f;
    if (vid >= 0) {
        score = vecRowDotF32(q, rows + (size_t)vid * (size_t)stride, dim);
    }
    scores[probeBase[p] + rowStart] = score;
}

// maxListLen is the longest probed list (computed host-side from the
// downloaded offsets; the device pointers above must NOT be read by
// the launcher — they are not host addressable). Probes with shorter
// lists exit early inside the kernel, so one grid.x covers all.
static int vecLaunchScanLists(const float* rows, long stride, int dim,
                              const int* ids, const long* offsets,
                              const int* probeLists, const long* probeBase,
                              int numProbes, const float* q, float* scores,
                              int maxListLen) {
    if (!rows || !ids || !offsets || !probeLists || !probeBase || !q ||
        !scores || dim <= 0 || numProbes <= 0 || maxListLen <= 0) {
        return 1;
    }
    dim3 block(256);
    dim3 grid((maxListLen + block.x - 1) / block.x, numProbes);
    vecScanListsKernel<<<grid, block>>>(rows, stride, dim, ids, offsets,
                                        probeLists, probeBase, numProbes, q,
                                        scores);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 3;
    }
    return 0;
}

// ---------------------------------------------------------------------
// Device top-k (best first, ties -> smaller id), one block per query
// ---------------------------------------------------------------------

// vecBetter reports whether (v, i) beats the running (best, bestI) with
// heapAdd order: strict improvement, or ties resolved toward the
// smaller id (the CPU heap keeps the first winner). bestI < 0 means
// "no candidate yet".
__device__ __forceinline__ bool vecBetter(int l2, float v, int i, float best,
                                          int bestI) {
    if (i < 0) {
        return false;
    }
    if (bestI < 0) {
        return true;
    }
    if (l2) {
        return v < best || (v == best && i < bestI);
    }
    return v > best || (v == best && i < bestI);
}

__global__ void vecTopKKernel(float* __restrict__ scores, int count, int k,
                              int l2, int* __restrict__ outIds,
                              float* __restrict__ outVals) {
    const int q = blockIdx.x;
    float* sc = scores + (size_t)q * (size_t)count;
    extern __shared__ float sBest[]; // blockDim floats then blockDim ints
    int* sIdx = (int*)(sBest + blockDim.x);
    const float posInf = __int_as_float(0x7f800000);
    const float negInf = __int_as_float(0xff800000);
    const float absent = l2 ? posInf : negInf;
    for (int r = 0; r < k; ++r) {
        float best = absent;
        int bestI = -1;
        for (int i = threadIdx.x; i < count; i += blockDim.x) {
            if (vecBetter(l2, sc[i], i, best, bestI)) {
                best = sc[i];
                bestI = i;
            }
        }
        sBest[threadIdx.x] = best;
        sIdx[threadIdx.x] = bestI;
        __syncthreads();
        for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                float o = sBest[threadIdx.x + s];
                int oi = sIdx[threadIdx.x + s];
                if (vecBetter(l2, o, oi, sBest[threadIdx.x], sIdx[threadIdx.x])) {
                    sBest[threadIdx.x] = o;
                    sIdx[threadIdx.x] = oi;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            outIds[(size_t)q * (size_t)k + (size_t)r] = sIdx[0];
            outVals[(size_t)q * (size_t)k + (size_t)r] = sBest[0];
            if (sIdx[0] >= 0) {
                // Boom the winner out of the candidate set: min scans see
                // +inf, max scans -inf.
                sc[sIdx[0]] = absent;
            }
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------
// k-means (P0-04): device mini-batch assignment pass
// ---------------------------------------------------------------------

__global__ void vecKmeansAssignKernel(const float* __restrict__ rows,
                                      long stride, int dim,
                                      const int* __restrict__ sampleIdx,
                                      int numSamples,
                                      const float* __restrict__ centroids,
                                      int numCentroids, float* __restrict__ sums,
                                      int* __restrict__ counts) {
    const int s = blockIdx.x * (int)blockDim.x + threadIdx.x;
    if (s >= numSamples) {
        return;
    }
    const int vid = sampleIdx[s];
    const float* rr = rows + (size_t)vid * (size_t)stride;
    int best = -1;
    float bestD = 0.0f;
    for (int c = 0; c < numCentroids; ++c) {
        float d = vecRowL2F32(rr, centroids + (size_t)c * (size_t)dim, dim);
        if (best < 0 || d < bestD) {
            best = c;
            bestD = d;
        }
    }
    if (best < 0) {
        return;
    }
    atomicAdd(counts + best, 1);
    float* srow = sums + (size_t)best * (size_t)dim;
    for (int j = 0; j < dim; ++j) {
        atomicAdd(srow + j, rr[j]);
    }
}

__global__ void vecFullAssignKernel(const float* __restrict__ rows, long stride,
                                    int dim, const float* __restrict__ centroids,
                                    int numCentroids, int count,
                                    int* __restrict__ best) {
    const int vid = blockIdx.x * (int)blockDim.x + threadIdx.x;
    if (vid >= count) {
        return;
    }
    const float* rr = rows + (size_t)vid * (size_t)stride;
    int bestC = -1;
    float bestD = 0.0f;
    for (int c = 0; c < numCentroids; ++c) {
        float d = vecRowL2F32(rr, centroids + (size_t)c * (size_t)dim, dim);
        if (bestC < 0 || d < bestD) {
            bestC = c;
            bestD = d;
        }
    }
    best[vid] = bestC;
}

// ---------------------------------------------------------------------
// extern C wrappers (called from vector_cuda.go cgo glue)
// ---------------------------------------------------------------------

extern "C" {

int vectorCudaInit(void) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) {
        return 0;
    }
    if (cudaSetDevice(0) != cudaSuccess) {
        return 0;
    }
    // Shared-scan tiles up to 48 KB dynamic shared memory per block.
    if (cudaFuncSetAttribute(vecMqScanF32Kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             48 * 1024) != cudaSuccess) {
        // Not fatal: smaller tiles still fit the default carve-out.
    }
    if (cudaFuncSetAttribute(vecMqScanCSoAKernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             48 * 1024) != cudaSuccess) {
    }
    return 1;
}

void* vectorMalloc(size_t bytes) {
    void* p = NULL;
    if (bytes == 0) {
        return NULL;
    }
    if (cudaMalloc(&p, bytes) != cudaSuccess) {
        return NULL;
    }
    return p;
}

void vectorFree(void* p) {
    if (p != NULL) {
        cudaFree(p);
    }
}

int vectorUpload(void* dst, const void* src, size_t bytes) {
    if (!dst || !src || bytes == 0) {
        return 1;
    }
    if (cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        return 2;
    }
    return 0;
}

int vectorDownload(void* dst, const void* src, size_t bytes) {
    if (!dst || !src || bytes == 0) {
        return 1;
    }
    if (cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        return 2;
    }
    return 0;
}

int vectorMemset32(void* dst, size_t bytes) {
    if (!dst) {
        return 1;
    }
    if (cudaMemset(dst, 0, bytes) != cudaSuccess) {
        return 2;
    }
    return 0;
}

int vectorSync(void) {
    if (cudaDeviceSynchronize() != cudaSuccess) {
        return 1;
    }
    return 0;
}

int vectorMQScanF32(const float* qs, int numQ, const float* rows, long stride,
                    int dim, int count, int l2, float* out) {
    int rc = vecLaunchMqScanF32(qs, numQ, rows, stride, dim, count, l2, out);
    if (rc != 0) {
        return rc;
    }
    return vectorSync();
}

int vectorMQScanCSoA(const float* qsRe, const float* qsIm, int numQ,
                     const float* re, const float* im, int dim, int count,
                     int l2, float* out) {
    int rc = vecLaunchMqScanCSoA(qsRe, qsIm, numQ, re, im, dim, count, l2, out);
    if (rc != 0) {
        return rc;
    }
    return vectorSync();
}

int vectorScanListsF32(const float* rows, long stride, int dim, const int* ids,
                       const long* offsets, const int* probeLists,
                       const long* probeBase, int numProbes, const float* q,
                       float* scores, int maxListLen) {
    int rc = vecLaunchScanLists(rows, stride, dim, ids, offsets, probeLists,
                                probeBase, numProbes, q, scores, maxListLen);
    if (rc != 0) {
        return rc;
    }
    return vectorSync();
}

int vectorTopKF32(const float* scores, int count, int numQ, int k, int l2,
                  int* ids, float* vals) {
    if (!scores || !ids || !vals || count <= 0 || numQ <= 0 || k <= 0 ||
        k > 128) {
        return 1;
    }
    dim3 block(256);
    dim3 grid(numQ);
    size_t shared = (size_t)block.x * (sizeof(float) + sizeof(int));
    vecTopKKernel<<<grid, block, shared>>>((float*)scores, count, k, l2, ids,
                                           vals);
    if (cudaGetLastError() != cudaSuccess) {
        return 3;
    }
    return vectorSync();
}

int vectorKmeansAssignF32(const float* rows, long stride, int dim,
                          const int* sampleIdx, int numSamples,
                          const float* centroids, int numCentroids,
                          float* sums, int* counts) {
    if (!rows || !sampleIdx || !centroids || !sums || !counts || dim <= 0 ||
        numSamples <= 0 || numCentroids <= 0) {
        return 1;
    }
    if (vectorMemset32(sums, (size_t)numCentroids * (size_t)dim *
                                 sizeof(float)) != 0) {
        return 2;
    }
    if (vectorMemset32(counts, (size_t)numCentroids * sizeof(int)) != 0) {
        return 3;
    }
    dim3 block(256);
    dim3 grid((numSamples + (int)block.x - 1) / (int)block.x);
    vecKmeansAssignKernel<<<grid, block>>>(rows, stride, dim, sampleIdx,
                                           numSamples, centroids, numCentroids,
                                           sums, counts);
    if (cudaGetLastError() != cudaSuccess) {
        return 4;
    }
    return vectorSync();
}

int vectorFullAssignF32(const float* rows, long stride, int dim,
                        const float* centroids, int numCentroids, int count,
                        int* best) {
    if (!rows || !centroids || !best || dim <= 0 || numCentroids <= 0 ||
        count <= 0) {
        return 1;
    }
    dim3 block(256);
    dim3 grid((count + (int)block.x - 1) / (int)block.x);
    vecFullAssignKernel<<<grid, block>>>(rows, stride, dim, centroids,
                                         numCentroids, count, best);
    if (cudaGetLastError() != cudaSuccess) {
        return 3;
    }
    return vectorSync();
}

} // extern "C"
