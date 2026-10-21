# Longbow-Quarrel Performance & Benchmark Results (v0.3.0)

> **Version scope.** This document describes release **v0.3.0**, the latest
> git tag. `main` carries additional unreleased work; see
> [roadmap.md](roadmap.md) for what has landed and what is planned.

## 1. Executive Summary

Longbow-Quarrel v0.3.0 introduces **zero-copy quantized inference**, **SIMD batch dequantization**, **partial GPU layer offloading**, and **SIMD-accelerated MatMul**, yielding substantial gains in memory efficiency and execution throughput:

1. **Heap Memory Reduction**: Slashed Go heap memory from **17.9 GB down to < 20 MB** on 4B parameter models (>99.9% reduction), completely eliminating swap thrashing and kernel OOM kills.
2. **SIMD Matrix-Vector Operations**: Zero-copy dot products for all major quantization formats (`Q8_0`, `Q4_K`, `Q6_K`, `Q2_K`, `Q3_K`, `Q4_0`, `Q5_0`, `Q5_K`, `BF16`) directly on memory-mapped quantized bytes.
3. **SIMD-Accelerated MatMul**: `MatMul` now uses B-transpose + `VecDotF32` inner loop for batch prefill and attention score computation.
4. **TurboQuant SIMD Vectorization**: Vectorized inverse rotation and QJL transforms in AVX-512, AVX2, and ARM NEON.

---

## 2. Memory Consumption Comparison (4B Parameter Q8_0 Model)

Tested with `Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated.Q8_0.gguf` (4.4 GB on disk) on a Linux workstation (AMD/Intel x86_64, 22 GB RAM):

| Metric | Upfront Dequantization (Pre-v0.2.0) | Zero-Copy Inference (v0.2.0) | Impact |
|--------|------------------------------------|-------------------------------|--------|
| **Go Heap Allocation** | 17.9 GB | < 20 MB | **99.9% reduction** |
| **System Free RAM During Inference** | 0 MB (Swap thrashing / OOM killer) | 17.1 GB Free | **Stable system execution** |
| **Time to First Token (TTFT)** | ~45s (Upfront dequant barrier) | **< 100 ms** (Instant start) | **450x faster startup** |
| **Embedding Table Memory** | 1.5 GB | 10 KB (Row-on-demand) | **99.99% reduction** |

---

## 3. SIMD Dequantization & MatVec Benchmarks

Measured on an Intel Core i7-12650H (AVX2, no AVX-512), with:

```bash
go test -run '^$' -bench 'BenchmarkDequantizeQ4K_(256|1024|4096)$' -benchmem ./internal/gguf/
go test -run '^$' -bench 'BenchmarkMatVecMulQ[48]_' -benchmem ./internal/simd/
```

| Benchmark Operation | Elements | Latency | Throughput |
|---------------------|----------|---------|-----------|
| `BenchmarkDequantizeQ4K_256` | 256 | 318 ns | 805 M elem/s |
| `BenchmarkDequantizeQ4K_1024` | 1024 | 1.31 μs | 781 M elem/s |
| `BenchmarkDequantizeQ4K_4096` | 4096 | 4.76 μs | 860 M elem/s |
| `BenchmarkMatVecMulQ8_0_VNNI` (1024×1024) | 1,048,576 | 17.6 μs | 59.6 G elem/s |
| `BenchmarkMatVecMulQ4_K_VNNI` (1024×1024) | 1,048,576 | 20.3 μs | 51.7 G elem/s |

> The benchmark names in the previous revision of this table
> (`BenchmarkDequantizeQ4K_SIMD_*`, `BenchmarkDequantizeQ6K_SIMD_*`) do not
> exist: the benchmarks are named `BenchmarkDequantizeQ4K_*`, and the Q6_K
> path is exercised through `BenchmarkDequantizeQ6K_Comparison`. Every row
> above is reproducible with the commands shown.

---

## 4. SIMD-Accelerated MatMul (v0.3.0)

`MatMul` now pre-transposes B for cache-friendly column access and uses `VecDotF32` for the inner dot product, replacing the previous scalar triple-nested loop.

| Operation | Size | Before (Scalar) | After (VecDotF32) | Speedup |
|-----------|------|-----------------|-------------------|---------|
| `MatMul` | 256×256 × 256×256 | ~12 ms | ~3.1 ms | **~3.9x** |
| `MatMul` | 128×128 × 128×128 | ~1.5 ms | ~0.4 ms | **~3.8x** |

Parallel outer-row goroutines used when `rowsA * colsB >= 1024`.

---

## 5. TurboQuant SIMD Acceleration (AVX-512, AVX2 & NEON)

> **Status: rewritten October 2026.** The table previously published here was
> not reproducible: the vectorized path did not exist. The three Go
> implementations of `PolarQuantSIMD` were pure Go, and the C kernels
> (`polar_quant_avx512`, `polar_quant_avx2`, `polar_quant_neon`,
> `qjl_transform_*`) were never referenced from any Go file. The numbers
> below are the actual measured ones; see
> [roadmap.md](roadmap.md) Step 33.

The C kernels are now dispatched at runtime from
`internal/simd/turboquant_simd_{amd64,arm64,generic}.go`, guarded by CPU
feature detection (`HasAVX512()`, `HasAVX2()`), with a canonical pure-Go
reference in `internal/simd/turboquant.go` as the fallback.

Measured with `go test -bench PolarQuant -benchmem ./internal/simd/` on a
12th Gen Intel Core i7-12650H (AVX2, no AVX-512), $n = 256$, 4-bit:

| Operation | Pure Go reference | Vectorized SIMD | Speedup |
|-----------|-------------------|-----------------|---------|
| **PolarQuant (rotate + quantize + residual)** | 85,848 ns/op | 11,377 ns/op | **7.5x** |

Allocations drop from 4 to 3 per call (3,328 B/op → 1,284 B/op).

Two caveats:

- The AVX-512 kernel is **not** exercised on this host (the CPU lacks
  `avx512f`), so no AVX-512 figure is published. `HasAVX512()` gates the call
  because the kernel is compiled with `#pragma GCC target("avx512f,...")` and
  would raise SIGILL otherwise.
- The accelerated kernels use fused multiply-add, so their last-ULP results
  differ slightly from the pure-Go reference. The quantized output can
  therefore differ by one level where the pre-rounding value lands exactly on
  a `.5` boundary. `internal/simd/turboquant_parity_test.go` and
  `internal/device/turboquant_parity_test.go` pin the semantics that matter
  (the clamp, and the residual being in the original space, matching the CUDA
  `turboquant_encode_kernel`).

---

## 6. Partial GPU Layer Offloading Performance

Tested on NVIDIA GeForce GPU (8 GB VRAM) with a 32-layer 7B model:

| Configuration | GPU VRAM Allocated | Host RAM Allocated | Generation Speed (tok/s) |
|---------------|-------------------|-------------------|--------------------------|
| **Full GPU (32 layers)** | OOM (Requires ~14 GB) | N/A | Fails (VRAM exhausted) |
| **Partial Offload (16 GPU / 16 CPU)** | 7.1 GB | 3.8 GB | 14.2 tok/s |
| **Partial Offload (8 GPU / 24 CPU)** | 3.6 GB | 5.7 GB | 9.8 tok/s |
| **Full CPU (0 GPU / 32 CPU)** | 0 GB | 7.6 GB | 5.1 tok/s |

---

## 7. Recommendations for Production Deployments

1. **Memory Budgeting**: For models with $> 4\text{B}$ parameters on consumer hardware, prefer `Q8_0` or `Q4_K` formats to leverage zero-copy SIMD arithmetic.
2. **GPU Layer Splitting**: Use `-ngl <N>` to allocate up to 80-90% of available GPU VRAM, allowing the CPU SIMD engine to process remaining layers seamlessly.
3. **KV Cache Compression**: Enable TurboQuant (`KVCacheTQ1_0`) for long contexts ($> 8\text{k}$ tokens) to reduce KV memory by ~8x.

---

## 8. Math Backend A/B Evaluation (stdlib `math` vs `emlgo`)

> **Status: measured. The answer is no: `emlgo` is a regression on
> this hardware, and the standard library stays.** The measurements below
> were taken with the harness in 8.1 on a 12th generation Intel Core
> i7-12650H (AVX2 and FMA, no AVX-512) with Go 1.27. The reason is
> visible in the numbers: `emlgo`'s vectorised paths need AVX-512, and
> without it they are slower than the standard library, not faster.

The engine calls `math.*` ~700 times across ~100 files, concentrated in the
sampler (softmax, entropy, repetition penalty), RoPE, RMSNorm / SwiGLU / GELU,
dequantization, and vector distance. `emlgo` provides SIMD batch equivalents
(AVX-512 / AVX2+FMA / NEON / SVE / WASM) that could vectorize these per-token
scalar loops.

### 8.1 Per-operation microbenchmark (measured)

Run both backends and record `ns/op` and `allocs/op`:

```bash
go test -run '^$' -bench . -benchmem ./internal/mathx/ > stdlib.txt
go test -tags emlgo -run '^$' -bench . -benchmem ./internal/mathx/ > emlgo.txt
scripts/bench_gate.sh compare stdlib.txt emlgo.txt
```

| Operation | stdlib `math` ns/op | `emlgo` scalar ns/op | `emlfast` ns/op | `emlgo` speedup | `emlfast` speedup |
|-----------|--------------------:|---------------------:|-----------------:|-----------------:|-------------------:|
| `Exp` | 5.43 | 6.48 | 5.84 | 0.84x | 0.93x |
| `Log` | 6.41 | 8.73 | 4.71 | 0.73x | 1.36x |
| `Log2` | 10.89 | 9.80 | 11.63 | 1.11x | 0.94x |
| `Log10` | 6.43 | 8.82 | 9.01 | 0.73x | 0.71x |
| `Sqrt` | 0.14 | 1.92 | 0.12 | 0.07x | 1.08x |
| `Pow` | 3.69 | 23.14 | 24.89 | 0.16x | 0.15x |
| `Sin` | 6.33 | 10.63 | 6.42 | 0.60x | 0.99x |
| `Cos` | 6.80 | 10.63 | 7.26 | 0.64x | 0.94x |
| `Tanh` | 5.01 | 17.16 | 17.23 | 0.29x | 0.29x |
| `Abs` | 0.15 | 1.45 | 1.54 | 0.10x | 0.09x |
| `Max` / `Min` | 2.05 | 2.01 | 2.05 | 1.02x | 1.00x |
| `Floor` / `Ceil` | 1.00 | 2.20 | 2.20 | 0.45x | 0.45x |
| `Round` | 0.14 | 2.32 | 2.24 | 0.06x | 0.06x |
| `Trunc` | 0.97 | 2.01 | 2.31 | 0.48x | 0.42x |

A speedup below 1.00x is a regression. Every operation the engine calls in
a hot loop is slower on the `emlgo` path, and most of them are slower on
the `emlfast` path too. The two that win — `Log` and `Sqrt` on `emlfast` —
do not make up for `Pow`, which the sampler and the RoPE use, and which is
five to seven times slower.

The batch path is the one `emlgo` exists for, and it is per element:

| Operation | stdlib ns/element | `emlgo` batch ns/element | Speedup |
|-----------|-------------------:|-------------------------:|--------:|
| `Exp` | 5.43 | 6.30 | 0.86x |
| `Log` | 6.41 | 7.73 | 0.83x |
| `Sin` | 6.33 | 7.09 | 0.89x |
| `Cos` | 6.80 | 7.83 | 0.87x |
| `Tanh` | 5.01 | 4.27 | 1.17x |
| `Sqrt` | 0.14 | 2.11 | 0.06x |

The batch path is the faster of the two `emlgo` paths and it still loses
almost everywhere. `emlgo` allocates for the batch call (8 KB and one
allocation per call), which the engine's hot loops cannot afford either.

**Decision: keep the standard library.** The `emlgo` build stays
available behind `-tags emlgo` so the comparison can be repeated on
hardware with AVX-512, but the default build does not use it.

### 8.2 End-to-end inference A/B (not measured)

Build `quarrel` with and without `-tags emlgo` and run
`scripts/benchmark_models.sh` / `profile_tokens.sh` on a fixed model
(Qwen3.5-4B-Q8_0):

| Backend | TTFT (ms) | tok/s | P95 (ms) | P99 (ms) | Peak RSS (MB) |
|---------|----------:|------:|---------:|---------:|--------------:|
| stdlib `math` (`!emlgo`) | — | — | — | — | — |
| `emlgo` scalar (`emlgo`) | — | — | — | — | — |
| `emlgo` fast (`emlgo,emlfast`) | — | — | — | — | — |

Not measured: the end-to-end run needs the Qwen3.5-4B-Q8_0 model, which
is not in the repository. The per-operation numbers above already answer
the question the end-to-end run would answer, because the engine calls
these functions one value at a time.

### 8.3 Decision gate

Adopt `emlgo` only if **all** hold:

1. Parity tests pass (`mathx.X(x) == math.X(x)`, or `≤ 1 ULP` for `emlfast`).
2. No regression in the existing `bench_gate.sh` 1.5× regression gate.
3. A measurable win on the target hardware.

Otherwise keep stdlib `math` and record the negative result here.

**The result: the gate is not met.** The parity tests pass — the values
are the same — but the performance gate fails on every operation the
engine calls in a hot loop, on hardware without AVX-512. The standard
library stays.

---

## 9. AVX-512 & VNNI Hardware Certification (Google Cloud c2-standard-4)

Validated and certified on Google Cloud `c2-standard-4` instance (`tf-testbed-e4fdb8`, Intel Xeon Cascade Lake @ 3.10GHz, 4 vCPUs, flags: `avx512f`, `avx512bw`, `avx512dq`, `avx512cd`, `avx512vl`, `avx512_vnni`, `avx2`, `fma`, `f16c`):

### 9.1 Microbenchmarks (Go 1.27, `-tags avx512`)

> **Reproducing these numbers requires AVX-512 hardware.** The `-tags avx512`
> build selects the AVX-512 kernels at *runtime*, via `simdLevel`; on a CPU
> without `avx512f` it falls back to the scalar implementation, so the same
> command produces very different numbers. The command is:
>
> ```bash
> go test -tags avx512 -run '^$' -bench 'BenchmarkRMSNorm_AVX512|BenchmarkMatmul_AVX512|BenchmarkVecDotQ[48]_' -benchmem ./internal/simd/
> ```
>
> The CI `avx512` job compiles and vets the tag and asserts that the VNNI
> kernels are linked into the binary; the benchmark itself is only meaningful
> on a runner with AVX-512.

| Kernel Operation | Vector Size | Latency (ns/op) | Memory (B/op) | Allocs/op |
|------------------|-------------|----------------:|--------------:|----------:|
| `BenchmarkRMSNorm_AVX512` | 4 × 64 | **103.9 ns** | 0 B | **0** |
| `BenchmarkRMSNorm_AVX512` | 16 × 128 | **453.3 ns** | 0 B | **0** |
| `BenchmarkRMSNorm_AVX512` | 64 × 256 | **3,411 ns** | 0 B | **0** |
| `BenchmarkRMSNorm_AVX512` | 128 × 512 | **12,512 ns** | 0 B | **0** |
| `BenchmarkMatmul_AVX512` | 8 × 16 × 8 | **187.9 ns** | 0 B | **0** |
| `BenchmarkMatmul_AVX512` | 32 × 32 × 32 | **3,727 ns** | 0 B | **0** |
| `BenchmarkMatmul_AVX512` | 64 × 64 × 64 | **18,688 ns** | 0 B | **0** |
| `BenchmarkMatmul_AVX512` | 128 × 128 × 64 | **68,350 ns** | 0 B | **0** |
| `BenchmarkVecDotQ8_0_VNNI` | 1,024 elements | **148.7 ns** | 0 B | **0** |
| `BenchmarkVecDotQ8_0_VNNI` | 4,096 elements | **475.6 ns** | 0 B | **0** |
| `BenchmarkVecDotQ4_K_VNNI` | 1,024 elements | **1,344 ns** | 0 B | **0** |
| `BenchmarkVecDotQ4_K_VNNI` | 4,096 elements | **5,219 ns** | 0 B | **0** |

### 9.2 Critical Hardware Defects Resolved

1. **IEEE 754 Exponent Bias in `fast_exp_avx512`**:
   - `fast_exp_avx512` in `internal/simd/kernels_avx512.c` previously shifted the integer component directly into exponent bits via `_mm512_slli_epi32(int_part, 23)` without adding the IEEE 754 exponent bias (`+127`).
   - For negative inputs, this caused integer sign extension into the float sign bit, yielding negative exponentials and `-Inf` softmax sum.
   - Fixed by adding `_mm512_add_epi32(int_part, _mm512_set1_epi32(127))` before the bit shift, matching `fast_exp_avx2`.
2. **Zero-Allocation CGo Hotpath**:
   - Updated `dot_q8_0_vnni` and `dot_q4_k_vnni` to return `float` directly rather than passing a pointer (`float* result`), completely eliminating CGo heap escapes and achieving strict zero-allocation ($0\text{ B/op}, 0\text{ allocs/op}$) across all VNNI dot product invocations.

---

## 10. Vector search

`internal/vector` is a standalone component: the inference path does not
use it. The numbers below are the ones the roadmap's vector blockers are
measured against, on the host this document was written on: 16 cores, one
NVIDIA GPU, AVX2 without AVX-512.

### 10.1 Reproducing the numbers

```bash
# The whole matrix. The dataset sizes come from the environment, so CI
# can run the small one and the full matrix is reproducible.
VECTOR_BENCH_SIZES=50000,100000,250000 \
  go test -run '^$' -bench 'BenchmarkFlat|BenchmarkIVF|BenchmarkHNSW|BenchmarkDistance' \
  -benchtime 3x ./internal/vector/

# The CPU profile of the IVF search, which is where the time goes.
VECTOR_BENCH_SIZES=250000 go test -run '^$' \
  -bench 'BenchmarkIVFSearch/float32/250000_384' -benchtime 3x \
  -cpuprofile /tmp/ivf.prof ./internal/vector/
go tool pprof -top /tmp/ivf.prof
```

Every search benchmark runs 50 queries per operation, top-k = 10, 16
probes for IVF and at most 50,000 nodes for HNSW. The per-query figures
below are the operation time divided by 50.

The inverted-list scan reads the rows where they are: the rows of a list
are stored next to each other, so the scan of a list is a sequential read
and the prefetcher follows it. The scan used to copy every candidate
into a scratch buffer first, and the copy moved twelve megabytes per
query — as much as the distances.

### 10.2 The distance kernels

The hand-written AVX2 kernels are used by the default build; the AVX-512
build adds the wider ones.

| Kernel | Dimension | Latency | Per element |
|---------|----------:|--------:|------------:|
| `L2Float32` | 128 | **33.2 ns** | 0.26 ns |
| `L2Float32` | 384 | **42.0 ns** | 0.11 ns |
| `DotFloat32` | 384 | **39.0 ns** | 0.10 ns |
| `DotComplex128` (split planes) | 384 | **244 ns** | 0.64 ns |
| `DotTurboQuant` (asymmetric) | 384 | **178 ns** | 0.46 ns |

### 10.3 The searches at 250,000 vectors

| Search | 128 dims | 384 dims | Per query (384) |
|--------|---------:|---------:|---------------:|
| flat, float32 dot | 198 ms | 667 ms | 13.3 ms (75 QPS) |
| flat, float32 L2 | 202 ms | 637 ms | 12.7 ms (78 QPS) |
| flat, complex128 dot | 885 ms | 2,697 ms | 53.9 ms (18 QPS) |
| flat, TurboQuant dot | 100 ms | 226 ms | 4.5 ms (221 QPS) |
| flat batch, float32 dot (50 queries) | 36 ms | 100 ms | 2.0 ms (499 QPS) |
| flat batch, complex128 dot (50 queries) | 131 ms | 366 ms | 7.3 ms (137 QPS) |
| IVF, float32 | 12.0 ms | 59.1 ms | 1.18 ms (846 QPS) |
| IVF, TurboQuant | 32.5 ms | 132 ms | 2.6 ms (378 QPS) |
| IVF build (float32) | 226 ms | 622 ms | — |
| HNSW, float32 (50,000 nodes) | 4.3 ms | 9.0 ms | 0.18 ms (5,563 QPS) |
| HNSW, TurboQuant (50,000 nodes) | 24.4 ms | 45.3 ms | 0.91 ms (1,106 QPS) |

### 10.4 The blockers, measured

| Blocker | Criterion | Measured | Met |
|---------|-----------|----------|:----:|
| **P0-01** | `L2Float32` below 15 % of the CPU | the kernel is 6 % of the IVF *search*; the *build* is dominated by the cgo call overhead (65 % of the profile) | partly |
| **P0-01** | 250k IVF query under 500 µs | **240 µs** at 128 dimensions, **1.18 ms** at 384 (846 QPS) | at 128 |
| **P0-02** | 250k complex128 flat search above 500 QPS | **18 QPS** single, **137 QPS** batched | no |
| **P0-03** | TurboQuant HNSW above 10,000 QPS, 7.8× compression | **1,106 QPS** at 50,000 nodes; the compression is kept | no |
| **P0-04** | IVF construction on 250k×384 under 50 ms | **622 ms** on the CPU, 640 ms with the GPU k-means (the transfer of the 384 MB dataset is 135 ms of it) | no |
| **P0-05** | a million-vector index inside a 256 MB working set | the store is memory mapped and the Go heap stays flat: 100,000 × 384 vectors load into a 150 MB file and the heap does not grow | yes |

The implementations the criteria ask for are in the tree: the AVX2 and
AVX-512 distance kernels (`internal/vector/distance_avx2.c`,
`distance_avx512.c`), the CUDA batch kernels for the inverted-list
scoring and the mini-batch k-means (`cuda_vector_kernels.cu`), the split
real/imaginary planes for the complex type (`complex_soa.go`), the
asymmetric TurboQuant distance with a lookup table
(`turboquant_adc.go`) and the memory-mapped store (`mmapstore.go`).

The GPU assignment kernel was rewritten while these numbers were
measured: it gave every *thread* a row, and the threads of a warp read
thirty-two different rows, so every load touched thirty-two cache lines.
It now gives every *warp* a row and keeps the centroids in shared
memory. The assignment of 250,000 rows to 500 centroids went from 1.1
seconds to 240 ms, and the whole build from 1.5 seconds to 640 ms.

What is missing is the *speed*. The gap that dominates is the memory
bandwidth of the scan: at 250k × 384 with sixteen probes the scan reads
twelve megabytes per query, and that is the floor for a float32 store —
the query cannot go below the time it takes to read the data it has to
look at. The next step is a half-precision store for the scan, which
would halve the traffic. The second gap is that the complex type is
four times the bandwidth of float32; the GPU already has a float32 path.

The measurements above are the ones this document claims, and the
commands above reproduce them.
