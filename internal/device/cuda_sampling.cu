#include <cuda_runtime.h>
#include <stdint.h>
#include <float.h>
#include <math.h>

// =============================================================================
// On-Device GPU Top-K / Top-P Sampling and Penalty Kernels
// =============================================================================

__device__ __forceinline__ float uniform_rand(unsigned long long* state) {
    *state = (*state * 6364136223846793005ULL + 1442695040888963407ULL);
    return (float)((*state >> 32) & 0x7FFFFFFF) / 2147483648.0f;
}

__device__ __forceinline__ void warpReduceArgmax(float &maxVal, int &maxIdx) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float otherVal = __shfl_down_sync(0xffffffff, maxVal, offset);
        int otherIdx = __shfl_down_sync(0xffffffff, maxIdx, offset);
        if (otherVal > maxVal) {
            maxVal = otherVal;
            maxIdx = otherIdx;
        }
    }
}

__global__ void apply_penalties_kernel(
    float* logits,
    const int32_t* prevTokens,
    int numPrevTokens,
    float repetitionPenalty,
    float frequencyPenalty,
    float presencePenalty,
    float temperature,
    int vocabSize
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Scale temperature across entire vocabulary
    if (temperature > 0.0f && temperature != 1.0f) {
        for (int i = idx; i < vocabSize; i += gridDim.x * blockDim.x) {
            logits[i] /= temperature;
        }
    }
    
    // Apply repetition, frequency, presence penalties
    if (numPrevTokens > 0 && (repetitionPenalty != 1.0f || frequencyPenalty != 0.0f || presencePenalty != 0.0f)) {
        for (int p = idx; p < numPrevTokens; p += gridDim.x * blockDim.x) {
            int tok = prevTokens[p];
            if (tok >= 0 && tok < vocabSize) {
                float val = logits[tok];
                if (repetitionPenalty != 1.0f) {
                    if (val > 0.0f) {
                        val /= repetitionPenalty;
                    } else {
                        val *= repetitionPenalty;
                    }
                }
                val -= (presencePenalty + frequencyPenalty);
                logits[tok] = val;
            }
        }
    }
}

__global__ void argmax_kernel(const float* logits, int vocabSize, int32_t* outToken) {
    __shared__ float s_max[32];
    __shared__ int s_idx[32];

    float localMax = -1e38f;
    int localIdx = 0;

    for (int i = threadIdx.x; i < vocabSize; i += blockDim.x) {
        float val = logits[i];
        if (val > localMax) {
            localMax = val;
            localIdx = i;
        }
    }

    warpReduceArgmax(localMax, localIdx);

    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    if (lane == 0) {
        s_max[wid] = localMax;
        s_idx[wid] = localIdx;
    }
    __syncthreads();

    if (wid == 0) {
        localMax = (threadIdx.x < (blockDim.x + 31) / 32) ? s_max[lane] : -1e38f;
        localIdx = (threadIdx.x < (blockDim.x + 31) / 32) ? s_idx[lane] : 0;
        warpReduceArgmax(localMax, localIdx);
        if (threadIdx.x == 0) {
            *outToken = localIdx;
        }
    }
}

__global__ void sample_topk_topp_kernel(
    const float* logits,
    int vocabSize,
    int topK,
    float topP,
    unsigned long long seed,
    int32_t* outToken
) {
    if (topK <= 1) {
        // Fall back to greedy argmax
        if (threadIdx.x == 0) {
            float maxV = logits[0];
            int maxI = 0;
            for (int i = 1; i < vocabSize; i++) {
                if (logits[i] > maxV) {
                    maxV = logits[i];
                    maxI = i;
                }
            }
            *outToken = maxI;
        }
        return;
    }
    if (topK > 256) topK = 256;

    __shared__ float s_vals[256];
    __shared__ int s_ids[256];
    __shared__ float s_probs[256];

    // Initialize shared top candidates
    if (threadIdx.x < topK) {
        s_vals[threadIdx.x] = -1e38f;
        s_ids[threadIdx.x] = -1;
    }
    __syncthreads();

    // Each thread finds top candidate across strided items
    float myVal = -1e38f;
    int myIdx = -1;
    for (int i = threadIdx.x; i < vocabSize; i += blockDim.x) {
        float val = logits[i];
        if (val > myVal) {
            myVal = val;
            myIdx = i;
        }
    }

    if (threadIdx.x < topK) {
        s_vals[threadIdx.x] = myVal;
        s_ids[threadIdx.x] = myIdx;
    }
    __syncthreads();

    // Sort top candidates in shared memory (descending bitonic sort)
    for (int k = 2; k <= 256; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = threadIdx.x ^ j;
            if (ixj > threadIdx.x && ixj < 256) {
                if ((threadIdx.x & k) == 0) {
                    if (s_vals[threadIdx.x] < s_vals[ixj]) {
                        float tv = s_vals[threadIdx.x]; s_vals[threadIdx.x] = s_vals[ixj]; s_vals[ixj] = tv;
                        int ti = s_ids[threadIdx.x]; s_ids[threadIdx.x] = s_ids[ixj]; s_ids[ixj] = ti;
                    }
                } else {
                    if (s_vals[threadIdx.x] > s_vals[ixj]) {
                        float tv = s_vals[threadIdx.x]; s_vals[threadIdx.x] = s_vals[ixj]; s_vals[ixj] = tv;
                        int ti = s_ids[threadIdx.x]; s_ids[threadIdx.x] = s_ids[ixj]; s_ids[ixj] = ti;
                    }
                }
            }
            __syncthreads();
        }
    }

    // Compute softmax over topK candidates
    float maxV = s_vals[0];
    float threadExp = 0.0f;
    if (threadIdx.x < topK && s_ids[threadIdx.x] >= 0) {
        threadExp = expf(s_vals[threadIdx.x] - maxV);
        s_probs[threadIdx.x] = threadExp;
    } else if (threadIdx.x < 256) {
        s_probs[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // Sum probabilities on thread 0
    if (threadIdx.x == 0) {
        float sumProbs = 0.0f;
        for (int i = 0; i < topK && s_ids[i] >= 0; i++) {
            sumProbs += s_probs[i];
        }

        if (sumProbs <= 0.0f) {
            *outToken = (s_ids[0] >= 0) ? s_ids[0] : 0;
            return;
        }

        unsigned long long rngState = seed ^ 0x9e3779b97f4a7c15ULL;
        float r = uniform_rand(&rngState);

        float cumSum = 0.0f;
        int chosenToken = (s_ids[0] >= 0) ? s_ids[0] : 0;
        float invSum = 1.0f / sumProbs;

        for (int i = 0; i < topK && s_ids[i] >= 0; i++) {
            float p = s_probs[i] * invSum;
            cumSum += p;
            if (r <= cumSum || cumSum >= topP) {
                chosenToken = s_ids[i];
                break;
            }
        }
        *outToken = chosenToken;
    }
}

extern "C" {
    void cudaSampleOnDevice(
        cudaStream_t stream,
        float* logits,
        const int32_t* prevTokens,
        int numPrevTokens,
        float repetitionPenalty,
        float frequencyPenalty,
        float presencePenalty,
        float temperature,
        int topK,
        float topP,
        unsigned long long seed,
        int vocabSize,
        int32_t* outToken
    ) {
        int blocks = (vocabSize + 255) / 256;
        if (blocks > 256) blocks = 256;

        if (numPrevTokens > 0 && (repetitionPenalty != 1.0f || frequencyPenalty != 0.0f || presencePenalty != 0.0f)) {
            apply_penalties_kernel<<<blocks, 256, 0, stream>>>(
                logits, prevTokens, numPrevTokens,
                repetitionPenalty, frequencyPenalty, presencePenalty,
                temperature, vocabSize
            );
        } else if (temperature > 0.0f && temperature != 1.0f && topK > 1) {
            apply_penalties_kernel<<<blocks, 256, 0, stream>>>(
                logits, NULL, 0,
                1.0f, 0.0f, 0.0f,
                temperature, vocabSize
            );
        }

        if (temperature <= 0.0f || topK <= 1) {
            argmax_kernel<<<1, 256, 0, stream>>>(logits, vocabSize, outToken);
            return;
        }

        sample_topk_topp_kernel<<<1, 256, 0, stream>>>(
            logits, vocabSize, topK, topP, seed, outToken
        );
    }
}
