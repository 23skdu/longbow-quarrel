# Longbow-Quarrel - Next Steps & Roadmap

## 10-Part Architectural & Performance Improvement Plan (v0.4.0) - COMPLETED

Based on recent architectural advances (coherence parity for Qwen 3.5 GatedDeltaNet SSM and Gemma 4, unified CPU/CUDA hybrid layer offloading, and memory reclamation), all 10 milestones and operational backlog tasks are fully implemented and verified:

### Part 1: Native CUDA Quantized Matrix Multiplication (Zero-Dequant GEMM) [COMPLETED]
- **Goal**: Eliminate host FP16 dequantization during model load by evaluating quantized weights (`Q4_K`, `Q8_0`, `Q6_K`, `Q2_K`, `Q3_K`) directly in CUDA kernels.
- **Status**: Implemented warp-level CUDA dequant GEMM kernels in `cuda_kernels.cu`, zero-dequant byte storage in `CUDAModel`, and transparent dispatch in `cuda.go`.
- **Files**: `internal/device/cuda.go`, `internal/device/cuda_kernels.cu`

### Part 2: CUDA Prefill Flash Attention with Sliding Window Support [COMPLETED]
- **Goal**: Accelerate multi-token prompt prefill on GPU by $5\times-10\times$.
- **Status**: Implemented tiled online softmax FlashAttention-2 prefill kernel with native Gemma 4 (512-token local window) and Mistral sliding window masking.
- **Files**: `internal/device/cuda_kernels.cu`, `internal/device/cuda.go`, `internal/engine/engine_cuda.go`

### Part 3: Continuous Batching & Paged Attention Scheduling [COMPLETED]
- **Goal**: High-throughput multi-request serving with dynamic iteration-level scheduling.
- **Status**: ContinuousBatchManager with preemption, priority scheduling, iteration-level metrics, and direct PagedKVCache block allocation.
- **Files**: `internal/engine/continuous_batching.go`, `internal/engine/kv_cache_paged.go`, `internal/engine/engine_cuda.go`

### Part 4: Cross-Engine & Asymmetric Speculative Decoding [COMPLETED]
- **Goal**: Double text generation tokens-per-second on consumer hardware.
- **Status**: AsymmetricSpeculativeEngine pairing CPU/draft with GPU/target, plus dynamic draft length adaptation using EMA acceptance rate metrics.
- **Files**: `internal/engine/speculative.go`, `internal/engine/interface.go`

### Part 5: Gemma 4 End-to-End Vision-Language Pipeline (VLM) [COMPLETED]
- **Goal**: Support multi-modal image + text prompts using Gemma 4's native vision encoder.
- **Status**: Multiplatform vision encoder, cross-platform patch projection embeddings, and OpenAI-compatible `/v1/chat/completions` multimodal endpoint.
- **Files**: `internal/vlm/encoder.go`, `internal/vlm/vlm_loader.go`, `internal/api/server.go`, `cmd/quarrel/main.go`

### Part 6: Grammar-Constrained Sampling Expansion (CFG & Regex) [COMPLETED]
- **Goal**: Deterministic, validated JSON schema and function-call token generation.
- **Status**: Pushdown Automaton (PDA) tracking JSON state/stack, regex grammar, and token bitmask pre-filtering before softmax sampling.
- **Files**: `internal/sampler/grammar.go`, `internal/engine/sampler.go`

### Part 7: AVX-512 VNNI & AMX Quantized Dot Product Kernels [COMPLETED]
- **Goal**: 2x throughput boost for CPU-only inference on modern x86 hardware.
- **Status**: Hardware detection for Intel AMX and AVX-512 VNNI, `VecDotQ8_0_VNNI`, `VecDotQ4_K_VNNI`, and parallel matrix-vector routines.
- **Files**: `internal/simd/cpuinfo.go`, `internal/simd/simd.go`, `internal/simd/vnni_test.go`

### Part 8: FP8 / Q8_0 Quantized Paged KV Cache [COMPLETED]
- **Goal**: 50% VRAM reduction for KV caches, enabling 64k–128k context lengths on 8GB GPUs.
- **Status**: FP8 and Q8_0 page compression in `PagedKVCache` with dynamic per-block scale tracking (`kScales`, `vScales`) and fused quantized attention kernels.
- **Files**: `internal/engine/kv_cache_paged.go`, `internal/device/cuda_kernels.cu`, `internal/device/cuda.go`

### Part 9: Distributed Pipeline & Tensor Parallelism over Arrow Flight [COMPLETED]
- **Goal**: Multi-node and multi-GPU sharding for models exceeding single-device memory.
- **Status**: Arrow Flight RPC client integration with `DistributedEngine`, sharded forward layer execution, and `RecordDistributedTransfer` telemetry.
- **Files**: `internal/engine/distributed.go`, `internal/engine/remote.go`

### Part 10: Production Observability, Distributed Tracing & Memory Governors [COMPLETED]
- **Goal**: Enterprise-grade monitoring and proactive memory protection.
- **Status**: OpenTelemetry TTFT and inter-token latency tracking, proactive `MemoryGovernor` with auto-defrag/offloading at 85% capacity, Prometheus metrics exporter.
- **Files**: `internal/telemetry/telemetry.go`, `internal/metrics/metrics.go`, `internal/device/memory.go`

---

## Active Backlog & Operational Tasks [COMPLETED]

### Developer Experience & CI
| Priority | Item | Files | Description | Status |
|----------|------|-------|-------------|--------|
| P1 | Consolidate test scripts | `scripts/run_all_tests.sh` | Unified test runner for CPU, CUDA, SIMD, VLM, Grammar, Race, and Coverage | Done |
| P1 | Add coverage gates | `.github/workflows/ci.yml` | Enforces minimum 80% coverage on core packages in CI | Done |
| P1 | Release automation | `Makefile` | Release target with changelog generation, cross-platform builds, and checksum packaging | Done |

---

#### Last updated: September 2026 (v0.4.0 10-Part Plan Completed)