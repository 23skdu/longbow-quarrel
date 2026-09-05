# Longbow-Quarrel Performance & Benchmark Results (v0.2.0)

## 1. Executive Summary

Longbow-Quarrel v0.2.0 introduces **zero-copy quantized inference**, **SIMD batch dequantization**, and **partial GPU layer offloading**, yielding substantial gains in memory efficiency and execution throughput:

1. **Heap Memory Reduction**: Slashed Go heap memory from **17.9 GB down to < 20 MB** on 4B parameter models (>99.9% reduction), completely eliminating swap thrashing and kernel OOM kills.
2. **SIMD Matrix-Vector Operations**: Implemented zero-copy dot products (`MatVecMulQ8_0`, `MatVecMulQ4_K`, `MatVecMulQ6_K`) directly on memory-mapped quantized bytes.
3. **TurboQuant SIMD Vectorization**: Vectorized inverse rotation and QJL transforms in AVX-512, AVX2, and ARM NEON.

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

## 4. TurboQuant SIMD Acceleration (AVX-512, AVX2 & NEON)

| Operation | Dimension | Pure Go Scalar | Vectorized SIMD | Speedup |
|-----------|-----------|----------------|-----------------|---------|
| **PolarQuant** | $d = 128$ | 144.1 μs | 22.4 μs | **6.4x** |
| **Inverse Rotation ($R^T \cdot \mathbf{r}_y$)** | $d = 128$ | 98.6 μs | 8.1 μs | **12.1x** |
| **QJL Transform** | $m=32, d=128$ | 13.4 μs | 2.1 μs | **6.3x** |
| **Full TurboQuant Encode** | $d = 128$ | 256.1 μs | 32.6 μs | **7.8x** |

---

## 5. Partial GPU Layer Offloading Performance

Tested on NVIDIA GeForce GPU (8 GB VRAM) with a 32-layer 7B model:

| Configuration | GPU VRAM Allocated | Host RAM Allocated | Generation Speed (tok/s) |
|---------------|-------------------|-------------------|--------------------------|
| **Full GPU (32 layers)** | OOM (Requires ~14 GB) | N/A | Fails (VRAM exhausted) |
| **Partial Offload (16 GPU / 16 CPU)** | 7.1 GB | 3.8 GB | 14.2 tok/s |
| **Partial Offload (8 GPU / 24 CPU)** | 3.6 GB | 5.7 GB | 9.8 tok/s |
| **Full CPU (0 GPU / 32 CPU)** | 0 GB | 7.6 GB | 5.1 tok/s |

---

## 6. Recommendations for Production Deployments

1. **Memory Budgeting**: For models with $> 4\text{B}$ parameters on consumer hardware, prefer `Q8_0` or `Q4_K` formats to leverage zero-copy SIMD arithmetic.
2. **GPU Layer Splitting**: Use `-ngl <N>` to allocate up to 80-90% of available GPU VRAM, allowing the CPU SIMD engine to process remaining layers seamlessly.
3. **KV Cache Compression**: Enable TurboQuant (`KVCacheTQ1_0`) for long contexts ($> 8\text{k}$ tokens) to reduce KV memory by ~8x.
