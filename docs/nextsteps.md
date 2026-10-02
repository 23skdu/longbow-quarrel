# Longbow-Quarrel - Next Steps & Roadmap

## Executive Summary

Longbow-Quarrel is a high-performance, multi-backend inference engine in Go, CGo, CUDA, and Apple Metal. The latest git tag is **`0.3.0`**; the milestones named v0.4.0 through v0.8.0 in commit messages all landed on `main` without a release tag. This document tracks the remaining work toward **v1.0.0**.

---

## Active Roadmap & Prioritized Remaining Work

### Phase 2: Scale-Out Distributed & Multi-GPU Topology (target v0.9.0) — Priority: P0

#### Part 2: Distributed Arrow Flight Resiliency & Pipeline Stage Chaining
- **Objective**: Move `RemoteWorkerEngine` from point-to-point tensor sharding to resilient distributed clusters.
- **Design**:
  - Implement bi-directional heartbeat detection and auto-reconnect backoff in `FlightClient`.
  - Implement distributed pipeline chaining: allow Worker A (layers 0–15) to stream activations directly to Worker B (layers 16–31) without bouncing through coordinator.
  - Dynamic sequence rerouting upon remote worker fault injection.
- **Files**: `internal/engine/remote.go`, `internal/engine/distributed.go`, `internal/arrow_client/client.go`
- **Acceptance Criteria**: Distributed fault injection test completes without request termination when a fallback worker is configured.

#### Part 3: Dynamic KV Cache Tiering & Memory Defragmentation
- **Objective**: Implement LRU hot-cold page migration between VRAM and host RAM under heavy memory governor pressure.
- **Design**:
  - When `MemoryGovernor` exceeds 0.85 pressure, offload inactive sequence physical blocks to pinned CPU memory.
  - Restore blocks asynchronously when preemption queue resumes sequence.
  - Implement compaction / defragmentation for paged block tables to reduce internal fragmentation.
- **Files**: `internal/engine/kv_cache_paged.go`, `internal/device/memory.go`, `internal/engine/batch.go`
- **Acceptance Criteria**: Continuous batch stress test under 95% KV cache capacity maintains 100% request completion without OOM panics.

---

### Phase 3: Production Serving, WebUI & Parity Validation (target v1.0.0) — Priority: P1

#### Part 4: Modern Embedded WebUI
- **Objective**: Deliver a modern, dynamic web interface based on `docs/webui_plan.md`.
- **Design**:
  - Embed pre-built React/Vite dashboard into Go binary (`go:embed`).
  - Features: real-time token generation speedometer, VRAM & KV cache occupancy visualization, model switching, chat playground, LoRA adapter toggle.
  - Dark mode, glassmorphism aesthetics, responsive SSE streaming.
- **Files**: `cmd/quarrel/`, `webui/`, `internal/api/server.go`, `docs/webui_plan.md`
- **Acceptance Criteria**: `quarrel --webui` starts server and opens responsive interface; end-to-end Playwright tests pass in headless CI.

#### Part 5: Automated Nightly Cross-Engine Benchmarking CI
- **Objective**: Prevent latency and coherence regressions across releases against industry baselines.
- **Design**:
  - Run `scripts/benchmark_engines.sh` and `scripts/benchmark_models.sh` as scheduled GitHub Actions workflow.
  - Compare Quarrel against llama.cpp and vLLM on standard models (Qwen 3.5, Mistral, Gemma 4).
  - Automatically publish comparative performance markdown tables to release artifacts.
- **Files**: `.github/workflows/benchmark.yml`, `scripts/benchmark_report.py`, `scripts/profile_tokens.sh`
- **Acceptance Criteria**: Automated report generated with TTFT, tok/s, P95/P99 latency, and VRAM footprints.

#### Part 6: Multi-Target Developer Experience & Build Tag Isolation
- **Objective**: Streamline IDE diagnostics and cross-compilation across CPU, CUDA, Metal, and TPU.
- **Design**:
  - Separate backend-specific interfaces to prevent gopls warning on non-active build tags.
  - Provide `.vscode/settings.json` multi-view or script toggles for switching active language server target tags (`cuda` vs `cpu` vs `metal`).
  - Keep all core package coverage strictly above the 80% CI gate.
- **Files**: `.vscode/settings.json`, `internal/device/`, `scripts/run_all_tests.sh`
- **Acceptance Criteria**: Zero diagnostics warnings in IDE across all files regardless of active backend tag; 100% CI pass rate.

---

## Milestone History & Completed Work

> **Release status.** The latest git tag is **`0.3.0`**. Everything below
> `v0.3.0` landed on `main` but has **not** been tagged or released. The
> milestone headings below are kept because the commits are named after them,
> but read "Completed" as "landed on `main`, unreleased" rather than
> "shipped to users". Cutting a tag is a release decision, not a docs edit.

### Post-0.3.0 work (landed on `main`, unreleased)

- **Multi-GPU 1F1B Pipeline Parallelism with Tensor-Parallel Fusion**: Advanced `multi_gpu.go` into an operational 1F1B micro-batch scheduling engine with double-buffered input/output activation staging, host fallback staging for non-P2P topologies, asynchronous transfer overlap, and exposed CLI `--devices`, `--tp-size`, and `--pp-stages` in `cmd/quarrel/main.go`.
- **Chunked Prefill & Intermediate Token Guard**: Prevented large prompt prefill operations from starving ongoing token generation; implemented chunked iteration slicing with priority preemption safeguards and TTFT Prometheus telemetry.
- **CUDA TurboQuant End-to-End Infrastructure**: Implemented missing CUDA `FetchKV` and `cudaStoreKVTurboQuant`, polar rotation matrices, QJL residual reconstruction, and verified end-to-end numerical precision on hardware.
- **Fused QKV Projection + RoPE CUDA Kernel**: High-performance fused attention prologue computing Q, K, V projections in a single memory pass with register-level RoPE and precomputed rotary frequency tables for full and partial RoPE.
- **MoE Dynamic Routing & CUDA Expert Dispatch**: High-throughput warp-reduction router logits and top-k argmax selection with fused softmax; fused expert SwiGLU down-projection dispatch avoiding empty expert invocations and verified on Nemotron, Mixtral, and GPT-OSS models.
- **DeepSeek Multi-Head Latent Attention (MLA)**: Low-rank KV cache decompression (`W_UKV`), decoupled query RoPE splitting, absorbed latent space query projection (`W_UK`), and fused decode attention directly over compressed KV caches.
- **Gemma 4 CPU Optimization**: Eliminated heap allocations per token (~630 down to ~42) via `Gemma4LayerBuf` reuse.
- **Metadata Dynamic Extraction**: Replaced hardcoded values with GGUF metadata for sliding window, RoPE theta, head dims, and PLE.
- **AVX-512 Kernel & Gosec Hardening**: F16C intrinsics, `_mm256_cvtepi8_epi16` signed reinterpret casts, CI memory limits (`GOMEMLIMIT`).
- **Cross-Engine Benchmarking Suite**: Delivered `benchmark_engines.sh`, `benchmark_models.sh`, `benchmark_report.py`, and `profile_tokens.sh`.
- **Zero-Dequant GEMM Coverage**: Delivered `dequant_q6_k_gemm_kernel`, Q2_K, Q3_K zero-dequant CUDA GEMMs.
- **Tiled FlashAttention-2**: Shared-memory cooperative KV tiled kernel with Mistral sliding-window attention.
- **Priority Preemption Scheduling**: Priority-aware scheduling with `PreemptLowestPriority` in continuous batching.
- **VLM Pipeline Fusion**: Gemma 4 VLM vision encoder with CLI `--image` and OpenAI multimodal endpoint.
- **AVX-512 VNNI / AMX Intrinsics**: Direct `vpdpbusd` SIMD execution paths.
- **Quantized Paged KV Cache**: FP8/Q8_0 paged cache execution via `PagedAttentionQuantized`.
- **Observability Hotpath**: Real-time TTFT and inter-token latency Prometheus telemetry with Memory Governor integration.
- **Docker / CI Matrix**: Multi-target Dockerfiles, GHCR publish automation, compose validation.
- **Grammar Strictness**: Constrained CFG and regex sampling with negative test verification.
- **Continuous batching, speculative decoding framework, baseline CUDA/Metal kernels, Prometheus exporter, and unified test runner** (originally scoped as "v0.4.0").

---

## Known Defects

### `DoPutTensor` cannot complete a round trip (P0)

`internal/arrow_client/client.go` documents `DoPutTensor` as "sends a tensor to
worker and returns the result tensor data". It cannot do this:

1. The server side of an Arrow Flight `DoPut` stream can only reply with
   `flight.PutResult`, whose only payload is `app_metadata`. There is no field
   for tensor bytes.
2. `DoPutTensor` then calls `stream.Recv()`, which in `arrow-go` v18 expects a
   record stream and fails with
   `arrow/flight: could not create flight reader`.

Measured against a real in-process Flight server, **both** plausible server
behaviours (send a `PutResult` acknowledgement, or send nothing and close) end
in that same error. See
`internal/arrow_client/doput_tensor_test.go::TestDoPutTensor_CannotCompleteRoundTrip`,
which pins the behaviour so a fix has to update it deliberately.

Impact:

- `internal/engine/remote.go` calls `DoPutTensor` for every remote layer, so
  distributed layer sharding always fails with `ForwardBatch DoPutTensor failed`.
- The `result == nil` branch in `DoPutTensor`, which returns an empty slice,
  is unreachable dead code. Had it been reachable, `remote.go`'s
  `if len(resultData) > 0 { LoadFrom(...) }` would have silently skipped the
  load and returned a zero-filled tensor while reporting success.

**Fix direction:** retrieve results with `DoGet` (or an application-level
descriptor echoed back in `PutResult.app_metadata`) instead of trying to decode
the `DoPut` acknowledgement.

---

> For detailed performance benchmarks, pprof profiling data, and vector engine P0 blockers, see [docs/roadmap.md](roadmap.md).

#### Last updated: September 2026 (latest tag `0.3.0`; post-0.3.0 work landed on `main`, unreleased)
