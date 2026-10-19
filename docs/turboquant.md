# TurboQuant Design & SIMD Implementation Document

## 1. Overview

TurboQuant is an advanced KV cache compression technique that combines:
1. **PolarQuant** — 4-bit or 2-bit quantization of the primary signal on the unit sphere
2. **QJL Transform** — 1-bit quantized Johnson-Lindenstrauss residual correction

This achieves **~8x to 16x memory compression** of the KV cache while maintaining output perplexity and coherence.

---

## 2. Mathematical Formulation

### Encoding:
1. **Input Vector $\mathbf{x} \in \mathbb{R}^d$**: Normalized to $\mathbf{u} = \mathbf{x} / \|\mathbf{x}\|$.
2. **Random Orthogonal Rotation**: $\mathbf{y} = \mathbf{R} \mathbf{u}$, where $\mathbf{R} \in \mathbb{R}^{d \times d}$ is an orthogonal matrix.
3. **Polar Quantization**: Quantize $\mathbf{y}$ to $\hat{\mathbf{y}}$ with codebook $\mathcal{C}$ (2-bit or 4-bit).
4. **Residual Calculation**: Compute residual in rotated domain:
   $$\mathbf{r}_y = \mathbf{y} - \hat{\mathbf{y}}$$
   Inverse rotate residual back to unrotated domain:
   $$\mathbf{r} = \mathbf{R}^T \mathbf{r}_y$$
5. **QJL Transform**: Project residual with random matrix $\mathbf{S} \in \mathbb{R}^{m \times d}$:
   $$\mathbf{z} = \operatorname{sign}(\mathbf{S} \mathbf{r})$$
   Scale factor:
   $$s_j = \frac{\sqrt{\pi/2}}{m} \sum_{i=1}^m |(\mathbf{S} \mathbf{r})_i|$$

---

## 3. SIMD Vector Acceleration

### A. Vectorized Inverse Rotation ($R^T \cdot \text{res}$)

A naive implementation of $\mathbf{r} = \mathbf{R}^T \mathbf{r}_y$ requires strided column memory accesses when $R$ is stored in row-major layout:
$$r_i = \sum_{j=0}^{d-1} R_{j, i} \cdot r_{y, j}$$
Accessing $R_{j, i}$ for a fixed column $i$ causes non-contiguous stride-$d$ memory accesses, creating cache misses and preventing vectorization.

**Longbow-Quarrel Vectorization Strategy:**
We reformulate the matrix-vector multiplication as a linear combination of contiguous rows of $R$:
$$\vec{\mathbf{r}} = \sum_{j=0}^{d-1} r_{y, j} \cdot \vec{R}_{j, :}$$

For each index $j$:
1. Broadcast scalar $r_{y, j}$ across a SIMD vector register.
2. Contiguously load chunks of row $j$: $\vec{R}_{j, k : k+W}$.
3. Accumulate with Fused Multiply-Add (FMA):
   $$\vec{\mathbf{r}}_{k : k+W} \leftarrow \vec{\mathbf{r}}_{k : k+W} + r_{y, j} \cdot \vec{R}_{j, k : k+W}$$

#### 1. AVX-512 Kernel (`internal/simd/turboquant_avx512.c`)
- Uses `_mm512_set1_ps` to broadcast the scalar weight.
- Streams 16 float32 elements per instruction using `_mm512_loadu_ps`.
- Accumulates with `_mm512_fmadd_ps`.
- Also vectorizes QJL residual projection and fixes scalar `norm_sq` accumulation.

#### 2. AVX2 Kernel (`internal/simd/turboquant_avx2.c`)
- Uses `_mm256_set1_ps` to broadcast the scalar weight.
- Streams 8 float32 elements per instruction with `_mm256_loadu_ps` and `_mm256_fmadd_ps`.

#### 3. ARM NEON Kernel (`internal/simd/turboquant_neon.c`)
- Uses `vdupq_n_f32` to duplicate the scalar weight across 4 lanes.
- Streams 4 float32 elements per instruction with `vld1q_f32`.
- Accumulates with `vfmaq_f32`.

---

## 4. Hardware Backend Implementations

> **Updated October 2026.** The C kernels below were previously written but
> never called from Go: `grep -rn "polar_quant\|qjl_transform" --include=*.go .`
> returned nothing, and every build used a pure-Go fallback. They are now
> dispatched at runtime from
> `internal/simd/turboquant_simd_{amd64,arm64,generic}.go`, with a single
> canonical reference implementation in `internal/simd/turboquant.go`.
> See [nextsteps.md](nextsteps.md) Step 33.

| Backend | Location | Supported Transforms |
|---------|----------|----------------------|
| **AVX-512 (x86_64)** | `internal/simd/turboquant_avx512.c` (dispatched from `turboquant_simd_amd64.go`) | PolarQuant, QJL, Inverse Rotation |
| **AVX2 (x86_64)** | `internal/simd/turboquant_avx2.c` (dispatched from `turboquant_simd_amd64.go`) | PolarQuant, QJL, Inverse Rotation |
| **ARM NEON (arm64)** | `internal/simd/turboquant_neon.c` (dispatched from `turboquant_simd_arm64.go`) | PolarQuant, QJL, Inverse Rotation |
| **NVIDIA CUDA** | `internal/device/cuda.go` | Fused PolarQuant + QJL GPU kernel |
| **Apple Metal** | `internal/device/metal.go` | Fused MSL compute pipeline |
| **Generic Pure Go** | `internal/simd/turboquant.go` | Architecture-independent portable fallback |

The AVX-512 kernel is compiled with `#pragma GCC target("avx512f,avx512bw,fma")`,
so it is gated on the **runtime** `HasAVX512()` check; calling it on a CPU
without AVX-512 would raise SIGILL. The AVX2 kernel is gated on
`HasAVX2()`. NEON is part of the ARMv8-A baseline and needs no check.

All three C kernels reproduce the semantics of the CUDA
`turboquant_encode_kernel` exactly: the quantized value is clamped to
±maxQuantVal, and the residual is returned in the **original (un-rotated)
space**. `internal/device/turboquant_parity_test.go` asserts this against a
Go transcription of the CUDA kernel.

---

## 5. Verification & Testing

1. **Unit & Mathematical Parity Testing**:
   - `internal/simd/turboquant_parity_test.go`: pins the quantization clamp, pins the residual convention, and compares the accelerated path against the canonical reference.
   - `internal/device/turboquant_parity_test.go`: asserts the CPU encoder matches a Go transcription of the CUDA kernel.
   - `internal/simd/turboquant_neon_test.go`: Validates outputs across standard dimensions (16, 32, 64, 128, 256) and unaligned odd lengths (67, 125).
   - Validated ARM64 cross-compilation: `GOARCH=arm64 go test -c ./internal/simd`.
2. **Continuous Fuzz Testing**:
   - `internal/simd/turboquant_fuzz_test.go`:
     - `FuzzPolarQuant`: Validates codebook quantization, norm bounds, and reconstructions against random inputs, and asserts parity with the reference implementation.
     - `FuzzQJLTransform`: Validates sign projection, scale derivation, and residual bounds against arbitrary byte streams.
   - A single 45 s run reached 3.6 M executions with 0 failures. The fuzzer found the round-half-to-even tie-breaking mismatch between the C kernels and the GPU path during development.