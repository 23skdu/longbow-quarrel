# Longbow-Quarrel Performance & Benchmark Results (v0.3.0)

> **Version scope.** This document describes release **v0.3.0**, the latest
> git tag. `main` carries additional unreleased work; see
> [nextsteps.md](nextsteps.md) for what has landed and what is planned.

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

Benchmark executed on Intel Core i7-12650H (AVX2):

| Benchmark Operation | Elements | Latency (μs) | Throughput (M elem/s) |
|---------------------|----------|--------------|-----------------------|
| `BenchmarkDequantizeQ4K_SIMD_256` | 256 | 0.28 μs | 914 M/s |
| `BenchmarkDequantizeQ4K_SIMD_1024` | 1024 | 1.05 μs | 975 M/s |
| `BenchmarkDequantizeQ4K_SIMD_4096` | 4096 | 4.12 μs | 994 M/s |
| `BenchmarkDequantizeQ6K_SIMD_256` | 256 | 0.35 μs | 731 M/s |
| `BenchmarkDequantizeQ6K_SIMD_1024` | 1024 | 1.38 μs | 742 M/s |
| `BenchmarkDequantizeQ6K_SIMD_4096` | 4096 | 5.30 μs | 772 M/s |
| `BenchmarkMatVecMulQ4_K` (1024×1024) | 1,048,576 | 412 μs | 2,545 M/s |
| `BenchmarkMatVecMulQ8_0` (1024×1024) | 1,048,576 | 285 μs | 3,679 M/s |

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

| Operation | Dimension | Pure Go Scalar | Vectorized SIMD | Speedup |
|-----------|-----------|----------------|-----------------|---------|
| **PolarQuant** | $d = 128$ | 144.1 μs | 22.4 μs | **6.4x** |
| **Inverse Rotation ($R^T \cdot \mathbf{r}_y$)** | $d = 128$ | 98.6 μs | 8.1 μs | **12.1x** |
| **QJL Transform** | $m=32, d=128$ | 13.4 μs | 2.1 μs | **6.3x** |
| **Full TurboQuant Encode** | $d = 128$ | 256.1 μs | 32.6 μs | **7.8x** |

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

> **Status: planned, no results yet.** This section is the template the
> `emlgo` swap is measured against. See [nextsteps.md](nextsteps.md) Phase 3
> for the plan and [dependencies.md](dependencies.md) for the module-path
> prerequisite. Do not fill in numbers here until the A/B harness in Step 7.6
> has been run on the target hardware.

The engine calls `math.*` ~700 times across ~100 files, concentrated in the
sampler (softmax, entropy, repetition penalty), RoPE, RMSNorm / SwiGLU / GELU,
dequantization, and vector distance. `emlgo` provides SIMD batch equivalents
(AVX-512 / AVX2+FMA / NEON / SVE / WASM) that could vectorize these per-token
scalar loops.

### 8.1 Per-operation microbenchmark (to be measured)

Run both backends and record `ns/op` and `allocs/op`:

```bash
go test -run '^$' -bench . -benchmem ./internal/mathx/ > stdlib.txt
go test -tags emlgo -run '^$' -bench . -benchmem ./internal/mathx/ > emlgo.txt
scripts/bench_gate.sh compare stdlib.txt emlgo.txt
```

| Operation | stdlib `math` ns/op | `emlgo` scalar ns/op | `emlgo` batch ns/op | Speedup | Allocs/op |
|-----------|--------------------:|---------------------:|--------------------:|--------:|----------:|
| `Exp` | — | — | — | — | — |
| `Log` | — | — | — | — | — |
| `Sqrt` | — | — | — | — | — |
| `Pow` | — | — | — | — | — |
| `Sin` / `Cos` | — | — | — | — | — |
| `Tanh` | — | — | — | — | — |
| Softmax (row) | — | — | — | — | — |

### 8.2 End-to-end inference A/B (to be measured)

Build `quarrel` with and without `-tags emlgo` and run
`scripts/benchmark_models.sh` / `profile_tokens.sh` on a fixed model
(Qwen3.5-4B-Q8_0):

| Backend | TTFT (ms) | tok/s | P95 (ms) | P99 (ms) | Peak RSS (MB) |
|---------|----------:|------:|---------:|---------:|--------------:|
| stdlib `math` (`!emlgo`) | — | — | — | — | — |
| `emlgo` scalar (`emlgo`) | — | — | — | — | — |
| `emlgo` fast (`emlgo,emlfast`) | — | — | — | — | — |

### 8.3 Decision gate

Adopt `emlgo` only if **all** hold:

1. Parity tests pass (`mathx.X(x) == math.X(x)`, or `≤ 1 ULP` for `emlfast`).
2. No regression in the existing `bench_gate.sh` 1.5× regression gate.
3. A measurable win on the target hardware.

Otherwise keep stdlib `math` and record the negative result here.
