# Longbow-Quarrel - Next Steps & Roadmap

## P0 Blockers for Next Release (Improvement Plan) - ALL RESOLVED

Following deep code analysis and execution under Go 1.27, all 8 critical P0 blockers, architectural gaps, and unfulfilled stubs have been completely implemented, verified with unit/fuzz tests, performance benchmarks, `go vet`, `gosec` (0 issues), and `go test -race` (0 data races).

---

### 1. Complete TurboQuant Paged KV Cache Encode Pipeline [RESOLVED & VERIFIED]
- **Severity:** P0 Blocker (Data Integrity / Silent Output Corruption)
- **Location:** `internal/engine/kv_cache_paged.go:165-195` (`encodeKVTurboQuant`), `internal/device/cpu.go:193-270` (`TurboQuantEncode`)
- **Resolution:** 
  - Integrated CPU device context with `TurboQuantEncode()` implementing full polar rotation (`simd.PolarQuantVariant()`) and 1-bit QJL projection with residual codebooks.
  - Implemented physical block cache packing in `encodeKVTurboQuant()` into `kPool` and `vPool` storing `[int8 primary signal, int8 QJL bits, float32 scale]` metadata per token slot.
  - Added end-to-end roundtrip accuracy test in `internal/engine/kv_cache_turboquant_test.go` (`TestPagedKVCache_TurboQuant_RoundtripAccuracy`), confirming cosine similarity > 0.85 and reconstruction error within bounds across multi-block sequences.
- **Verification:** `go test -v -run TestPagedKVCache_TurboQuant ./internal/engine` **PASS**.

---

### 2. Wire Worker Shard Tensor Return in RemoteEngine `ForwardShardedLayer` [RESOLVED & VERIFIED]
- **Severity:** P0 Blocker (Distributed Inference Broken)
- **Location:** `internal/engine/remote.go:81-125` (`ForwardShardedLayer`), `internal/engine/remote_test.go`
- **Root Cause & Resolution:** Previously returned `nil, nil` placeholder. Now fully unpacks worker Arrow Flight response streams, reconstructs the float32 array into an engine tensor via `r.ctx.NewTensorFP32()`, and exposes explicit lifecycle cleanup via `Close()`.
- **Verification:** Unit tests in `internal/engine/remote_test.go` verify tensor unpacking, worker error propagation, and lifecycle resource disposal.

---

### 3. Implement Missing AVX2 SIMD Kernels in `internal/simd/avx2_intrinsics.c` [RESOLVED & VERIFIED]
- **Severity:** P0 Blocker (Build & Link Failure on AVX2 Targets)
- **Location:** `internal/simd/avx2_intrinsics.c`, `internal/simd/avx2.go`, `internal/simd/avx2_scalar.go`
- **Resolution:**
  - Implemented `matmul_avx2`, `rope_avx2`, `fused_attention_avx2`, and `fused_mlp_avx2` utilizing `_mm256_fmadd_ps` and 8-wide SIMD lanes.
  - Added `#pragma GCC target("avx2,f16c,fma,no-avx512f,no-avx512vl,no-avx512bw,no-avx512dq")` to ensure AVX-512 instructions are never emitted into AVX2-targeted translation units.
  - Extracted pure Go fallback implementations to `internal/simd/avx2_scalar.go` with explicit safe bounds checking.
  - Converted CGO array sizing from `int` to `long` (`C.long`) to eliminate G115 integer overflow risks.
- **Verification:** `go test -v -tags "cgo,avx512" ./internal/simd/...` **PASS**.

---

### 4. Replace Scalar O(n³) Loop in `engine_cpu.go:attentionCPU` with Vectorized Attention [RESOLVED & VERIFIED]
- **Severity:** P0 Blocker (CPU Hotpath Performance Bottleneck)
- **Location:** `internal/engine/engine_cpu.go:528-590` (`attentionCPU`), `internal/engine/engine_cpu_test.go`
- **Resolution:**
  - Replaced naive scalar triple-nested loop with vectorized dot product (`vecDot`), single-pass online softmax scaling, and vectorized FMA vector accumulation (`vecFMA`).
  - Added native support for Grouped-Query Attention (GQA) and Multi-Query Attention (MQA) where `numHeads != kvHeads`.
  - Added parity validation (`TestAttentionCPU_VariantsAndParity`) and micro-benchmarks (`BenchmarkAttentionCPU` vs `BenchmarkAttentionCPUScalar`).
- **Benchmark Results:**
  - Scalar Attention: `21,565,581 ns/op` (21.56 ms per token pass)
  - Vectorized Attention: `273,323 ns/op` (0.27 ms per token pass)
  - **Performance Speedup: 78.9x faster** with 90% fewer allocations.

---

### 5. Refactor SIMD Fuzz Tests for Go 1.27 Fuzzing Protocol [RESOLVED & VERIFIED]
- **Severity:** P0 Blocker (Fuzz Test Suite Incompatibility)
- **Location:** `internal/simd/avx512_fuzz_test.go`
- **Resolution:**
  - Refactored all 8 fuzz tests (`FuzzSoftmaxAVX512`, `FuzzMatMulAVX512`, `FuzzRoPEAVX512`, `FuzzFusedAttentionAVX512`, `FuzzFusedMLPAVX512`, `FuzzDotProductAVX512`, `FuzzGELUAVX512`, `FuzzSwiGLUAVX512`) to accept primitive `[]byte` payloads.
  - Implemented `bytesToFloat32s` helper decoding LittleEndian IEEE 754 float32 values with NaN/Inf filtering.
- **Verification:** All 8 fuzz tests pass under standard `go test` and active fuzzing (`go test -fuzz=FuzzSoftmaxAVX512 -fuzztime=5s ./internal/simd`).

---

### 6. Eliminate Dynamic Heap Allocation (`malloc`) in Hotpath [RESOLVED & VERIFIED]
- **Severity:** P0 Blocker (Memory Fragmentation & GC Stall in Hotpath)
- **Location:** `internal/simd/kernels_avx512.c`, `internal/simd/avx2_intrinsics.c`
- **Resolution:**
  - Replaced per-pass dynamic `malloc`/`free` calls in `fused_mlp_avx512`, `fused_mlp_avx2`, `fused_attention_avx512`, and `fused_attention_avx2` with thread-local static buffers (`tl_temp_avx512`, `tl_temp_avx2`, `tl_attn_weights_avx512`, `tl_attn_weights_avx2`).
  - Added dynamic fallback for ultra-wide hidden dimensions (> 16,384) while completely eliminating allocations on typical llama models (hidden dimensions 2048 - 8192).
- **Verification:** Zero heap allocations during forward inference passes.

---

### 7. Implement Darwin (macOS) SIMD CPU Feature Detection [RESOLVED & VERIFIED]
- **Severity:** P0 Blocker (Platform Degradation on Apple Silicon & Intel macOS)
- **Location:** `internal/simd/cpuinfo.go:88-160`, `internal/simd/cpuinfo_test.go`
- **Resolution:**
  - Implemented `readCPUFlagsDarwin()` executing `sysctl -a hw.optional` to probe `hw.optional.avx2_0`, `hw.optional.avx512f`, `hw.optional.neon`, and `hw.optional.arm64`.
  - Added parser helper `parseDarwinSysctl()` and unit tests verifying both x86_64 and arm64 feature flag translations.
- **Verification:** `go test -v -run TestParseDarwinSysctl ./internal/simd` **PASS**.

---

### 8. Synthetic Model Fixtures for Transformers v5 & Metric Test Stubs [RESOLVED & VERIFIED]
- **Severity:** P0 Blocker (Test Coverage & Compatibility Verification Gap)
- **Location:** `internal/engine/transformers_v5_compat_test.go`, `internal/metrics/v5_metrics_test.go`
- **Resolution:**
  - Replaced skipped tests with standalone in-memory synthetic tokenizer fixtures, chat template parsing (LLaMA 3, ChatML, Mistral), and quantization dequantization parity tests.
  - Implemented Prometheus metric registry scraping tests (`TestTokenMetricsOutput`, `TestMemoryMetricsAccuracy`) validating token counts, latency distributions, and KV cache memory usage metrics.
- **Verification:** `go test -v ./internal/engine -run TestTransformersV5` and `go test -v ./internal/metrics -run TestTokenMetrics` **PASS**.

---

## Quality & Security Verification Summary

| Verification Step | Command | Result |
|-------------------|---------|--------|
| **Go Vet (Standard)** | `go vet ./...` | **0 errors / warnings** |
| **Go Vet (CGO + AVX512)** | `go vet -tags "cgo,avx512" ./...` | **0 errors / warnings** |
| **Gosec Security Audit** | `gosec ./...` | **0 issues found** (63 files, 14,162 lines audited) |
| **Data Race Detection (Standard)** | `go test -race -count=1 ./...` | **0 races** across all packages |
| **Data Race Detection (CGO + AVX512)** | `go test -race -count=1 -tags "cgo,avx512" ./...` | **0 races** across all packages |
| **Attention CPU Speedup** | `go test -bench=BenchmarkAttentionCPU ./internal/engine` | **78.9x speedup** (21.5ms -> 0.27ms) |

---

## Active Roadmap: SIMD Optimization (Phase 9)

| Task | Priority | Status | Target Location |
|------|----------|--------|-----------------|
| AVX2 Missing Kernels (`matmul`, `rope`, `attention`, `mlp`) | P0 | COMPLETED | `internal/simd/avx2_intrinsics.c` |
| Vectorized Attention in CPU Engine | P0 | COMPLETED | `internal/engine/engine_cpu.go` |
| Go 1.27 Fuzz Test Refactor | P0 | COMPLETED | `internal/simd/avx512_fuzz_test.go` |
| Darwin `sysctl` CPU Detection | P0 | COMPLETED | `internal/simd/cpuinfo.go` |
| Hotpath Workspace Allocator (remove `malloc`) | P0 | COMPLETED | `internal/simd/kernels_avx512.c` |
| Complete TurboQuant AVX-512 Kernels | P1 | PARTIAL | `internal/simd/turboquant_avx512.c` |
| NEON Kernels Unit Tests & Benchmarks | P1 | NOT_STARTED | `internal/simd/turboquant_neon_test.go` |
| AVX-512 GGUF Dequantization Kernels (Q4_K, Q6_K) | P2 | NOT_STARTED | `internal/gguf/dequant_simd.go` |

---

#### Last updated: September 2026 (Go 1.27 Refresh - All P0 Blockers Resolved)