# Longbow-Quarrel - Next Steps & Roadmap

## Executive Summary

Longbow-Quarrel is a high-performance, multi-backend inference engine in Go, CGo, CUDA, and Apple Metal. Following the successful completion of the v0.4.0, v0.6.0, v0.7.0, and v0.8.0 milestones, this document establishes the re-prioritized engineering roadmap for **v0.9.0** and **v1.0.0**.

---

## Active Roadmap & Prioritized Remaining Work

### Phase 2: Scale-Out Distributed & Multi-GPU Topology (v0.9.0) — Priority: P0

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

### Phase 3: Production Serving, WebUI & Parity Validation (v1.0.0) — Priority: P1

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

### v0.9.0 (In Progress - September 2026)
- **Multi-GPU 1F1B Pipeline Parallelism with Tensor-Parallel Fusion**: Advanced `multi_gpu.go` into an operational 1F1B micro-batch scheduling engine with double-buffered input/output activation staging, host fallback staging for non-P2P topologies, asynchronous transfer overlap, and exposed CLI `--devices`, `--tp-size`, and `--pp-stages` in `cmd/quarrel/main.go`.

### v0.8.0 (Completed September 2026)
- **Chunked Prefill & Intermediate Token Guard**: Prevented large prompt prefill operations from starving ongoing token generation; implemented chunked iteration slicing with priority preemption safeguards and TTFT Prometheus telemetry.
- **CUDA TurboQuant End-to-End Infrastructure**: Implemented missing CUDA `FetchKV` and `cudaStoreKVTurboQuant`, polar rotation matrices, QJL residual reconstruction, and verified end-to-end numerical precision on hardware.
- **Fused QKV Projection + RoPE CUDA Kernel**: High-performance fused attention prologue computing Q, K, V projections in a single memory pass with register-level RoPE and precomputed rotary frequency tables for full and partial RoPE.
- **MoE Dynamic Routing & CUDA Expert Dispatch**: High-throughput warp-reduction router logits and top-k argmax selection with fused softmax; fused expert SwiGLU down-projection dispatch avoiding empty expert invocations and verified on Nemotron, Mixtral, and GPT-OSS models.
- **DeepSeek Multi-Head Latent Attention (MLA)**: Low-rank KV cache decompression (`W_UKV`), decoupled query RoPE splitting, absorbed latent space query projection (`W_UK`), and fused decode attention directly over compressed KV caches.

### v0.7.0 (Completed September 2026)
- **Gemma 4 CPU Optimization**: Eliminated heap allocations per token (~630 down to ~42) via `Gemma4LayerBuf` reuse.
- **Metadata Dynamic Extraction**: Replaced hardcoded values with GGUF metadata for sliding window, RoPE theta, head dims, and PLE.
- **AVX-512 Kernel & Gosec Hardening**: F16C intrinsics, `_mm256_cvtepi8_epi16` signed reinterpret casts, CI memory limits (`GOMEMLIMIT`).
- **Cross-Engine Benchmarking Suite**: Delivered `benchmark_engines.sh`, `benchmark_models.sh`, `benchmark_report.py`, and `profile_tokens.sh`.

### v0.6.0 (Completed September 2026)
- **Zero-Dequant GEMM Coverage**: Delivered `dequant_q6_k_gemm_kernel`, Q2_K, Q3_K zero-dequant CUDA GEMMs.
- **Tiled FlashAttention-2**: Shared-memory cooperative KV tiled kernel with Mistral sliding-window attention.
- **Priority Preemption Scheduling**: Priority-aware scheduling with `PreemptLowestPriority` in continuous batching.
- **VLM Pipeline Fusion**: Gemma 4 VLM vision encoder with CLI `--image` and OpenAI multimodal endpoint.
- **AVX-512 VNNI / AMX Intrinsics**: Direct `vpdpbusd` SIMD execution paths.
- **Quantized Paged KV Cache**: FP8/Q8_0 paged cache execution via `PagedAttentionQuantized`.
- **Distributed Remote Worker**: Arrow Flight RPC `RollbackKV` and multi-token `ForwardDraft`.
- **Observability Hotpath**: Real-time TTFT and inter-token latency Prometheus telemetry with Memory Governor integration.
- **Docker / CI Matrix**: Multi-target Dockerfiles, GHCR publish automation, compose validation.
- **Grammar Strictness**: Constrained CFG and regex sampling with negative test verification.

### v0.4.0 (Completed September 2026)
- Initial continuous batching, speculative decoding framework, baseline CUDA/Metal kernels, Prometheus exporter, and unified test runner.

---

> For detailed performance benchmarks, pprof profiling data, and vector engine P0 blockers, see [docs/roadmap.md](file:///home/rsd/REPOS/longbow-quarrel/docs/roadmap.md).

#### Last updated: September 2026 (v0.9.0 Active; v0.8.0 / v0.7.0 / v0.6.0 / v0.4.0 completed)
