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

### Phase 3: Pluggable Math Backend — `emlgo` A/B Evaluation (target v0.9.0) — Priority: P0

#### Part 7: Swap stdlib `math` for `emlgo` behind a build flag

- **Objective**: Replace the Go standard library `math` package in inference hot paths with [`emlgo`](https://github.com/23skdu/emlgo) (the EML — Exp-Minus-Log — operator library), gated behind a build flag so both backends can be A/B benchmarked from a single tree.
- **Why**: `math.*` appears **~700 times across ~100 files**, concentrated in the sampler (softmax, entropy, repetition penalty), RoPE, RMSNorm / SwiGLU / GELU activations, dequantization, and vector distance. emlgo ships SIMD batch equivalents (AVX-512 / AVX2+FMA / NEON / SVE / WASM SIMD128) that could vectorize these per-token scalar loops. This is an **evaluation**: adopt only if parity holds and a measurable win is demonstrated.

##### Step 7.1 — Resolve the module-path blocker first (P0, prerequisite)

`emlgo` **cannot be required as-is**. Its `go.mod` declares `module github.com/emlgo/eml`, but the repository is hosted at `github.com/23skdu/emlgo` and every internal import uses the `github.com/emlgo/eml/...` path. Requesting it under the host path fails:

```
$ go get github.com/23skdu/emlgo@0.5
go: github.com/23skdu/emlgo@0.5 (v0.0.0-20261003213601-b3d1a425cdd5) requires ...:
	module declares its path as: github.com/emlgo/eml
	        but was required as: github.com/23skdu/emlgo
```

There is **no `v0.5.0` semver tag** — the tag is `0.5`, which Go does not treat as semver — so the version is the pseudo-version **`v0.0.0-20261003213601-b3d1a425cdd5`** (commit `b3d1a425cdd5e8956b45593ce98bb96f06075507`, tag `0.5`, "latest 0.5").

**Do this before any code lands:**

1. Fork `github.com/23skdu/emlgo` at tag `0.5` and correct the module path: `module github.com/emlgo/eml` → `module github.com/23skdu/emlgo` in `go.mod`, then rewrite every `github.com/emlgo/eml/...` import to `github.com/23skdu/emlgo/...` (mechanical). Tag the fork `v0.5.0`.
2. In the root `go.mod`, add `require github.com/23skdu/emlgo v0.5.0` (or pin the pseudo-version until the fork tags `v0.5.0`).
3. `go mod tidy` and confirm `go build ./...` is green against the fork.

> **Alternative if a fork is undesirable:** `go mod vendor` the dependency and rewrite the import paths under `vendor/`. This pins the code but forfeits upstream updates; the fork is preferred.

##### Step 7.2 — Compatibility & accuracy audit

emlgo is **not** a drop-in `math` replacement. It is a set of packages (`pkg/arithmetic`, `pkg/trig`, `pkg/hyper`, `pkg/logexp`, `pkg/fastmath`, `pkg/quant`, `internal/eml`, …) with different names and signatures. Two accuracy classes:

- **Scalar `pkg/*` functions delegate to stdlib `math`** → bit-exact, safe to swap 1:1.
- **`pkg/fastmath` uses minimax polynomial approximations** (`FastExp`, `FastLog`, `FastEml`, …) → **not** bit-exact; ULP differences will shift logits and therefore sampling. These must be opt-in behind a separate tag.

Map every stdlib symbol the repo uses to its emlgo equivalent and record the expected ULP delta:

| stdlib | emlgo scalar (bit-exact) | emlgo fast (approx) |
|---|---|---|
| `math.Exp` | `logexp.Exp` / `arithmetic.Exp` | `fastmath.FastExp` |
| `math.Log` | `logexp.Log` / `arithmetic.Log` | `fastmath.FastLog` |
| `math.Log2` | `arithmetic.Log2` | — |
| `math.Sqrt` | `arithmetic.Sqrt` / `fastmath.Sqrt` | — |
| `math.Pow` | `arithmetic.Pow` | — |
| `math.Sin` / `math.Cos` | `trig.Sin` / `trig.Cos` | `trig.SinFast` / `trig.CosFast` |
| `math.Tanh` | `hyper.Tanh` | — |
| `math.Abs` | `arithmetic.Abs` | `eml.AbsBranchless` |
| `math.Max` / `math.Min` | `arithmetic.Max` / `arithmetic.Min` | — |
| `math.Floor` / `Ceil` / `Round` / `Trunc` | `arithmetic.Floor` / `Ceil` / `Round` / `Trunc` | — |
| `math.IsNaN` / `IsInf` / `NaN` / `Inf` | — no equivalent; **keep stdlib** | — |
| `math.Float32bits` / `frombits`, `Float64bits` / `frombits` | — no equivalent; **keep stdlib** | — |
| `math.Copysign`, `MaxInt`, `MaxFloat32/64`, `SmallestNonzeroFloat32`, `Pi` | — no equivalent; **keep stdlib** | — |

**Keep stdlib for** `IsNaN`, `IsInf`, `NaN`, `Inf`, the `*bits`/`frombits` conversions, `Copysign`, `MaxInt`, `MaxFloat32/64`, `SmallestNonzeroFloat32`, and `Pi` — emlgo has no equivalent and these are control-flow / bit-twiddling, not compute.

##### Step 7.3 — Abstraction layer (the only files that import a math package)

Introduce `internal/mathx/` with two build-tagged files so the rest of the tree never imports `math` or `emlgo` directly:

- `internal/mathx/math_stdlib.go` — `//go:build !emlgo` — thin wrappers over stdlib `math`.
- `internal/mathx/math_emlgo.go` — `//go:build emlgo` — wrappers over emlgo (scalar `pkg/*` for bit-exactness; `pkg/fastmath` only behind a second `emlfast` tag).

Expose one API matching the subset the repo uses (`Exp`, `Log`, `Log2`, `Sqrt`, `Pow`, `Sin`, `Cos`, `Tanh`, `Abs`, `Max`, `Min`, `Floor`, `Ceil`, `Round`, `Trunc`, plus the kept-stdlib helpers). Repoint all ~700 call sites from `math.Exp` → `mathx.Exp`, etc.

##### Step 7.4 — Build-flag gating

- Tag name: **`emlgo`** — matches the lowercase convention of `cuda`, `metal`, `tpu`, `webui`, `avx512`, `nccl`.
- Default build (`make cpu`, `make test`) → stdlib (`!emlgo`); **no behavior change**.
- A/B build → `go test -tags emlgo ./...`, `go build -tags emlgo`.
- Optional second tag **`emlfast`** for the non-bit-exact `fastmath` paths, so polynomial-approximation drift can be isolated from SIMD batching gains.
- Add Makefile targets `test-emlgo`, `bench-emlgo`, `bench-emlgo-all` mirroring `test-simd` / `bench-simd`.

##### Step 7.5 — Unit tests & accuracy parity

- **Parity tests** (`internal/mathx/parity_test.go`): for every wrapped function assert `mathx.X(x) == math.X(x)` (or `≤ 1 ULP` for `emlfast`) over a deterministic input sweep (edge values, denormals, large/small magnitudes, negatives). These fail loudly if emlgo drifts.
- **Golden regression**: capture sampler logits / generated token IDs on a fixed prompt+seed with stdlib, then assert the `emlgo` build reproduces them (bit-exact for scalar; tolerance for `emlfast`).
- **Property tests**: `Exp(Log(x))≈x`, `Sqrt(x)²≈x`, `Sin²+Cos²≈1` across the SIMD batch paths.
- **Existing gates must stay green**: `scripts/run_all_tests.sh` (8 gates), coverage ≥ 80%, `go vet`, `gosec`, race detector.

##### Step 7.6 — A/B performance benchmarking

- Add `internal/mathx/bench_test.go` with `Benchmark*` funcs for each hot op (`Exp`, `Log`, `Sqrt`, `Pow`, `Sin`, `Cos`, `Tanh`, softmax) in scalar and batch (`[]float64`) form.
- Run both backends and publish a table:
  ```bash
  go test -run '^$' -bench . -benchmem ./internal/mathx/ > stdlib.txt
  go test -tags emlgo -run '^$' -bench . -benchmem ./internal/mathx/ > emlgo.txt
  scripts/bench_gate.sh compare stdlib.txt emlgo.txt
  ```
- **End-to-end inference A/B**: build `quarrel` with and without `-tags emlgo`, run `scripts/benchmark_models.sh` / `profile_tokens.sh` on a fixed model (Qwen3.5-4B-Q8_0), and record TTFT, tok/s, P95/P99, and peak RSS per backend.
- **Decision gate**: adopt emlgo only if (a) parity tests pass, (b) no regression in the existing `bench_gate.sh` 1.5× gate, and (c) a measurable win on the target hardware. Otherwise keep stdlib and document the negative result.

- **Files**: `go.mod` / `go.sum`, `internal/mathx/`, `internal/engine/sampler.go`, `internal/device/cpu*.go`, `internal/gguf/dequant*.go`, `internal/vector/*.go`, `internal/tokenizer/tokenizer.go`, `internal/simd/*.go`, `Makefile`, `scripts/run_all_tests.sh`, `docs/dependencies.md`, `docs/performance.md`, `docs/testplan.md`, `docs/architecture.md`.
- **Acceptance Criteria**: `go test -tags emlgo ./...` passes with parity tests green; an A/B benchmark table is published showing per-op and end-to-end tok/s for stdlib vs emlgo; a decision (adopt or reject) is recorded with data.

---

### Phase 4: Production Serving, WebUI & Parity Validation (target v1.0.0) — Priority: P1

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
