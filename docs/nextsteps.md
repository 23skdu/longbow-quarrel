# Longbow-Quarrel - Next Steps & Roadmap

## Executive Summary

Longbow-Quarrel is a high-performance, multi-backend inference engine in Go, CGo, CUDA, and Apple Metal. The latest git tag is **`0.3.0`**; the milestones named v0.4.0 through v0.8.0 in commit messages all landed on `main` without a release tag. This document tracks the remaining work toward **v1.0.0**, including distributed execution (Phase 2), math backends (Phase 3), serving/DX (Phase 4), and deep kernel/engine optimizations (Phase 5: Steps 8–27).

> ### ⚠️ Read Phase 6 first
>
> A full-tree analysis of `main` (commit `8dd7f91`, October 2026) found that
> **the project does not currently ship a working CPU binary, the CI `lint`
> job is red, and the REST API is never started.** Phases 2–5 are largely
> implemented and unit-tested in isolation, but much of that work is not
> connected to any shipped entrypoint — and several headline performance
> claims (AVX-512, TurboQuant SIMD) rest on code that no build target
> compiles or calls.
>
> **Phase 6 (Steps 28–48)** tracks closing that gap. It is P0 and should be
> completed before any further performance work, because Steps 28–34 gate
> the trustworthiness of every measurement made in Phases 2–5.

---

## Active Roadmap & Prioritized Remaining Work

### Phase 2: Scale-Out Distributed & Multi-GPU Topology (target v0.9.0) — Priority: P0

#### Part 2: Distributed Arrow Flight Resiliency & Pipeline Stage Chaining (P0) — [IMPLEMENTED]
- **Objective**: Move `RemoteWorkerEngine` from point-to-point tensor sharding to resilient distributed clusters.
- **Design & Implementation**:
  - Implemented exponential backoff auto-reconnect (`ConnectWithRetry`) and server liveness ping (`Ping`) in `FlightClient`.
  - Implemented background health monitoring (`StartHeartbeat`) triggering custom failover handlers on disconnection.
  - Added pipeline stage chaining target metadata (`SetNextStage`) and dynamic fallback worker rerouting (`SetFallbackWorker`).
  - Added transparent worker failover in `ForwardBatch` and `ForwardShardedLayer`: when the primary worker fails or terminates, activation payloads automatically reroute to the fallback worker without request interruption.
- **Files**: `internal/engine/remote.go`, `internal/engine/distributed.go`, `internal/arrow_client/client.go`, `internal/engine/remote_resiliency_test.go`
- **Acceptance Criteria**: Distributed fault injection test (`TestRemoteWorkerEngine_FaultInjectionAndFailover`) completes without request termination when a fallback worker is configured.

#### Part 3: Dynamic KV Cache Tiering & Memory Defragmentation (P0) — [IMPLEMENTED]
- **Objective**: Implement LRU hot-cold page migration between VRAM and host RAM under heavy memory governor pressure.
- **Design & Implementation**:
  - Implemented hot-cold page tiering in `PagedKVCache`: `OffloadSequence(seqID)` transfers inactive sequence KV blocks into host memory buffers and releases physical device blocks back to the free pool.
  - Implemented `RestoreSequence(seqID)`: re-allocates physical device blocks and transfers cached KV state from host RAM back to device pools with bit-exact preservation.
  - Implemented `Defragment()`: compacts sparsely allocated physical blocks toward low indices, updates block tables, re-keys dirty page descriptors, and rebuilds contiguous free block tables.
  - Integrated with proactive `MemoryGovernor` (`RegisterMemoryGovernor`, `OffloadLRU`) and preemption in `ContinuousBatchManager` to resume offloaded sequences without re-prefill penalties.
- **Files**: `internal/engine/kv_cache_paged.go`, `internal/device/memory.go`, `internal/engine/continuous_batching.go`, `internal/engine/kv_cache_tiering_defrag_test.go`
- **Acceptance Criteria**: Continuous batch stress test under 95% KV cache capacity maintains 100% request completion without OOM panics; `TestPagedKVCache_OffloadAndRestoreRoundTrip`, `TestPagedKVCache_Defragmentation`, and `TestContinuousBatchManager_MemoryPressureAndTieringStress` pass.

---

### Phase 3: Pluggable Math Backend — `emlgo` A/B Evaluation (target v0.9.0) — Priority: P0

#### Part 7: Swap stdlib `math` for `emlgo` behind a build flag

- **Objective**: Replace the Go standard library `math` package in inference hot paths with [`emlgo`](https://github.com/23skdu/emlgo) (the EML — Exp-Minus-Log — operator library), gated behind a build flag so both backends can be A/B benchmarked from a single tree.
- **Why**: `math.*` appears **~700 times across ~100 files**, concentrated in the sampler (softmax, entropy, repetition penalty), RoPE, RMSNorm / SwiGLU / GELU activations, dequantization, and vector distance. emlgo ships SIMD batch equivalents (AVX-512 / AVX2+FMA / NEON / SVE / WASM SIMD128) that could vectorize these per-token scalar loops. This is an **evaluation**: adopt only if parity holds and a measurable win is demonstrated.

##### Step 7.1 — Resolve the module-path blocker first (P0, prerequisite) — [RESOLVED]

`emlgo`'s `go.mod` declares `module github.com/emlgo/eml`, while the repository is hosted at `github.com/23skdu/emlgo`.
This has been resolved cleanly in both the root `go.mod` and `cmd/webui/go.mod` using Go module `replace`:
```go
require (
    github.com/emlgo/eml v0.0.0-00010101000000-000000000000
)

replace github.com/emlgo/eml => github.com/23skdu/emlgo v0.0.0-20261015062852-39bcca0c1034
```
Pinned to the latest commit **`39bcca0c1034c51cd7e054daa60b9059f0c3846d`** (corresponding to upstream tag **`0.7`** and tag **`latest`**). Verified with `go mod verify` and unit parity tests in `internal/mathx/`.

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

### Phase 5: Deep Inference Engine Optimization & Hardware Acceleration (target v1.0.0) — Priority: P0

Based on deep profiling of autoregressive decode loops, CUDA kernel execution profiles, SIMD vectorization bottlenecks, and runtime diagnostics (`gopls` build-tag isolation warnings on `cpu.go` and `simd_default.go`), the following 10 steps define the core performance engineering work for v1.0.0.

#### Part 8 (Step 8): Split-K FlashDecoding & Cooperative Warp Reduction for PagedAttention (P0) — [IMPLEMENTED]
- **Objective**: Eliminate GPU Streaming Multiprocessor (SM) starvation during autoregressive decoding on long contexts.
- **Bottleneck Analysis**:
  - In `internal/device/cuda_kernels.cu:2028-2037` (`cudaPagedAttentionBatch`), the grid launch configuration is hardcoded as `dim3 grid(numTokens)` with `dim3 block(32, heads)`.
  - During single-token autoregressive generation ($N_{\text{tokens}} = 1$), `grid.x = 1`. A flagship GPU (e.g. NVIDIA RTX 4090 with 128 SMs, A100 with 108 SMs, or H100 with 132 SMs) receives only 1 threadblock, leaving **>95% of the GPU execution hardware completely idle**.
  - A single SM is forced to sequentially scan through thousands of physical KV blocks across all attention heads in `blockTables`, causing decode latency to scale linearly with context length (exceeding 35ms per token at 8k context).
- **Design & Kernel Architecture**:
  - Implement **Split-K FlashDecoding** in `internal/device/cuda_kernels.cu`: partition the KV sequence dimension into $S$ independent splits:
    $$S = \min\left(64, \max\left(1, \left\lceil \frac{\text{seqLen}}{256} \right\rceil\right)\right)$$
  - Launch a 2D grid: `grid(heads, S)` with `block(32, 4)` (128 threads per block). Each split processes $K_{\text{split}} = \lceil \text{numBlocks} / S \rceil$ blocks from `blockTables`.
  - Each threadblock computes partial online softmax statistics:
    $$m_s = \max_{t \in \text{split}_s} S_t, \quad l_s = \sum_{t \in \text{split}_s} e^{S_t - m_s}, \quad \mathbf{O}_s = \sum_{t \in \text{split}_s} e^{S_t - m_s} \mathbf{V}_t$$
    storing $(m_s, l_s, \mathbf{O}_s)$ into temporary device scratch space `[heads, S, headDim + 2]`.
  - Launch a secondary fast reduction kernel `reduce_split_k_flash_decode<<<grid(heads), block(32, 4)>>>` that combines the $S$ partial outputs using stable online softmax rescaling:
    $$m = \max_{s} m_s, \quad l = \sum_{s=1}^S l_s \cdot e^{m_s - m}, \quad \mathbf{O} = \frac{1}{l} \sum_{s=1}^S \mathbf{O}_s \cdot e^{m_s - m}$$
- **Files**: `internal/device/cuda_kernels.cu`, `internal/device/cuda.go`, `internal/device/cuda_wrapper.c`, `internal/engine/kv_cache_paged.go`
- **Acceptance Criteria**: Single-token decode latency on context lengths $\ge 8\text{k}$ tokens drops by $\ge 8\times$ (from $\sim 35$ms to $<4$ms); GPU SM occupancy during decode increases from $<5\%$ to $>75\%$.

#### Part 9 (Step 9): GPU-Native Fused QK-Norm & RoPE Elimination of CPU Fallback (P0) — [IMPLEMENTED]
- **Objective**: Prevent CPU layer fallback and eliminate all GPU-host ping-pong memory copies for models utilizing per-head QK normalization (Qwen 2.5, Qwen 3.5, Gemma 2/4).
- **Bottleneck Analysis**:
  - In `internal/engine/engine_cuda.go:512-536`, `needsCPU` triggers whenever:
    `(layer < len(e.cpuWeights.AttnQNorm) && len(e.cpuWeights.AttnQNorm[layer]) > 0) || headDim > 128`
  - When true, the CUDA engine performs `hiddenHost := hidden.ToHostF32()`, pulls full activation tensors across PCIe into host RAM, invokes `ApplyLayerCPUKV` or `ApplyGemma4LayerCPU` on the host CPU, and copies outputs back via `hidden.LoadFrom(hiddenHost)`.
  - This PCIe bus round-trip occurs on **every single layer** for Qwen 2.5/3.5 and Gemma 4, dropping token generation throughput from an expected $>70$ tok/s down to $<5$ tok/s.
- **Design & Kernel Architecture**:
  - Implement `cudaRMSNormPerHead` in `internal/device/cuda_kernels.cu`:
    - Operates directly on query tensor $\mathbf{Q} \in \mathbb{R}^{\text{numTokens} \times \text{heads} \times \text{headDim}}$ and key tensor $\mathbf{K} \in \mathbb{R}^{\text{numTokens} \times \text{kvHeads} \times \text{headDim}}$.
    - Computes per-head RMSNorm using warp-level intra-head reductions (`__shfl_down_sync`) and applies scale vector $\mathbf{w} \in \mathbb{R}^{\text{headDim}}$ in registers.
  - Extend `cudaRoPE` and `cudaFusedQKVRope` to support arbitrary `headDim` (192, 256) and partial rotary dimensions (`rotaryDim = 64` for Qwen 3.5 query-gate architectures).
  - Load `blk.{i}.attn_q_norm.weight` and `blk.{i}.attn_k_norm.weight` directly into `CUDAModel` as GPU tensors, and remove the CPU fallback condition in `engine_cuda.go`.
- **Files**: `internal/device/cuda_kernels.cu`, `internal/device/cuda.go`, `internal/engine/engine_cuda.go`, `internal/device/cuda_fused_test.go`
- **Acceptance Criteria**: Qwen 2.5/3.5 and Gemma 4 execute 100% on GPU with 0 PCIe activation transfers during token decode; generation speed reaches $>65$ tok/s on an RTX 4090 for Qwen 2.5 7B.

#### Part 10 (Step 10): CUDA Graph Capture & Replay for Sub-Millisecond Decode Latency (P0) — [IMPLEMENTED BUT NOT WIRED — see Step 35]
- **Objective**: Eliminate Go CGo bridge overhead and CUDA runtime kernel launch latency during single-token and small-batch decode passes.
- **Bottleneck Analysis**:
  - Generating one token launches 15–25 separate CUDA kernels per layer across 32 layers (RMSNorm, FusedRoPE, GEMV, PagedAttention, SwiGLU, Add), totaling $>600$ host-to-device kernel invocations per token.
  - Each CGo boundary transition and CUDA driver submission costs $4\text{--}8\,\mu$s, adding $2.5\text{--}5.0$ ms of pure host CPU dispatch overhead per token. This CPU bound throttles decode speed regardless of GPU FLOPs.
- **Implementation Status**:
  - Implemented `CUDAGraphManager` in `internal/device/cuda_graph.go` with CGo CUDA 12 graph driver wrappers in `internal/device/cuda_wrapper.c`.
  - Added capture management (`BeginCapture`, `EndCapture`, `Instantiate`, `Launch`, `Destroy`) with stream synchronizations and no-op fallback in `internal/device/cuda_graph_nocuda.go`.
  - Validated lifecycle and error branches against live RTX 4060 GPU in `internal/device/cuda_graph_test.go`.
- **Files**: `internal/device/cuda_graph.go`, `internal/device/cuda.go`, `internal/device/cuda_wrapper.c`, `internal/device/cuda_graph_nocuda.go`, `internal/device/cuda_graph_test.go`
- **Acceptance Criteria**: Host CPU dispatch time per decode step drops from $>3.0$ms to $<150\,\mu$s; end-to-end token generation throughput on low batch sizes improves by $30\text{--}50\%$.

#### Part 11 (Step 11): Persistent Thread Worker Pool & Zero-Allocation Scratchpad for CPU Inference (P0) — [IMPLEMENTED]
- **Objective**: Eliminate Go runtime goroutine creation thrashing and GC allocation pressure in CPU-based layer execution.
- **Bottleneck Analysis**:
  - In `internal/gguf/dequant_simd.go:196` (`MatVecMulQ4_K`), every matrix-vector multiplication queries `workers := runtime.GOMAXPROCS(0)` and dynamically spawns goroutines: `for w := 0; w < workers; w++ { go func(...) }` managed by `sync.WaitGroup`.
  - For a 32-layer model with 7 projections per layer, this creates and destroys **~3,584 goroutines per generated token**, inducing severe Go runtime scheduler lock contention, work-stealing overhead, and GC latency spikes.
  - Furthermore, `ApplyLayerCPUKV` (`internal/engine/cpu_weights.go:472-640`) executes 12+ dynamic slice allocations (`make([]float32, ...)`) per layer per token (allocating `normed`, `query`, `gate`, `kNormed`, `residual`, `normedFFN`, etc.).
- **Design & Implementation**:
  - Implement a persistent, CPU-affinity-pinned `WorkerPool` in `internal/simd/worker_pool.go` running exactly $N = \text{GOMAXPROCS}$ long-lived worker goroutines.
  - Workers synchronize using atomic task counters and light futex/spin-wait barriers, completely eliminating goroutine creation and destruction during inference.
  - Create a reusable `CPULayerScratch` struct in `internal/engine/cpu_weights.go` pre-allocated per sequence:
    ```go
    type CPULayerScratch struct {
        Normed    []float32
        QFull     []float32
        Query     []float32
        Gate      []float32
        Key       []float32
        Value     []float32
        AttnOut   []float32
        Residual  []float32
        FFNNormed []float32
        GateOut   []float32
        UpOut     []float32
    }
    ```
  - Repoint all projection calls and inner loops to reuse `CPULayerScratch` slices, reducing heap allocations in the forward pass to zero.
- **Files**: `internal/simd/worker_pool.go`, `internal/gguf/dequant_simd.go`, `internal/engine/cpu_weights.go`, `internal/engine/engine_cpu.go`
- **Acceptance Criteria**: Heap allocations during CPU generation drop to **0 allocs/token**; CPU generation throughput increases by $\ge 40\%$; goroutine count in `pprof` remains constant.

#### Part 12 (Step 12): Direct Zero-Dequant Integer Dot-Product Kernels via AVX2 & AVX-512 VNNI (P0) — [AVX2 PART IMPLEMENTED; AVX-512 PART NEVER COMPILED — see Step 32]
> ⚠️ The AVX-512 half of this step (`internal/simd/avx512.go`, 685 LOC) is gated
> `//go:build ... && avx512` and **no build target or CI job passes
> `-tags avx512`**. Only the AVX2 path (`avx2.go`) is reachable from a shipped
> build.
- **Objective**: Execute quantized matrix-vector operations directly in CPU vector registers without intermediate float32 conversion.
- **Bottleneck Analysis**:
  - `dequantizeQ4KBlocks` (`internal/gguf/dequant_simd.go:52-100`) extracts 4-bit nibbles and converts them into float32 values in a stack array (`blockDeq`), followed by a separate scalar float dot-product loop: `sum += blockDeq[k] * vector[vOffset+k]`.
  - This intermediate dequantization step consumes high memory bandwidth, incurs register spill/fill overhead, and executes up to 10 CPU instructions per 4-bit weight element.
- **Implementation Status**:
  - Implemented `dot_q8_0_avx2` and `dot_q4_k_avx2` directly in `internal/simd/avx2_intrinsics.c` using 256-bit SIMD intrinsics, `#pragma GCC target("avx2,fma,f16c")`, and horizontal accumulation.
  - Wired into `VecDotQ8_0_VNNI` and `VecDotQ4_K_VNNI` in `internal/simd/avx2.go` and `internal/simd/simd.go` to activate on Alder Lake host CPUs without requiring AVX-512.
  - Eliminated heap escape in CGo dot products, achieving **0 B/op and 0 allocs/op** with a **15.3x speedup** on Q4_K (521 ns vs 8000 ns) and **5.8x speedup** on Q8_0 (357 ns vs 2059 ns).
  - Added CI benchmark zero-allocation gate in `scripts/bench_gate.sh` and Gate 10 in `scripts/run_all_tests.sh`.
- **Files**: `internal/simd/avx2_intrinsics.c`, `internal/simd/avx2.go`, `internal/simd/simd.go`, `internal/simd/vnni_test.go`, `scripts/bench_gate.sh`, `scripts/run_all_tests.sh`
- **Acceptance Criteria**: `BenchmarkVecDotQ4_K_VNNI` drops to $<600$ ns on AVX2 with 0 allocs/op; regression gate verifies 0 allocs/op across all hotpath SIMD kernels.

#### Part 13 (Step 13): Speculative Tree Attention (Medusa / EAGLE-Style Tree Verification)
- **Objective**: Verify branching candidate draft trees in a single target model forward pass, eliminating sequential path verification.
- **Bottleneck Analysis**:
  - In `internal/engine/speculative.go:102-139` (`GenerateSpeculativeMultiPath`), the manager verifies multiple candidate paths by sequentially looping over paths and invoking `sm.targetEngine.ForwardDraft(candidateTokens)` once per path.
  - For a draft tree with 4 branches of length 4 (16 candidate tokens), the target model executes 4 separate forward passes, repeatedly re-evaluating shared prefix tokens and multiplying target verification latency.
- **Design & Kernel Architecture**:
  - Implement Speculative Tree Attention:
    1. Flatten the draft token candidate tree into a contiguous sequence of $K$ tokens (e.g. $K=16$ or $K=32$ for a 1-4-16 tree topology).
    2. Construct a 2D causal tree attention mask $\mathbf{M}_{\text{tree}} \in \{0, -\infty\}^{K \times K}$:
       $$\mathbf{M}_{\text{tree}}[i, j] = \begin{cases} 0 & \text{if token } j \text{ is an ancestor of token } i \text{ in the draft tree} \\ -\infty & \text{otherwise} \end{cases}$$
    3. Modify `cudaPagedAttentionBatch` and `cudaFlashAttention` in `internal/device/cuda_kernels.cu` to accept `treeMask` and apply it to attention logits prior to softmax.
    4. Execute the target model forward pass across all $K$ draft tokens in a single shot.
    5. Find the longest verified branch in the tree using top-1 or greedy acceptance criteria, accept the prefix, and roll back unaccepted tree branches with `RollbackKV`.
- **Files**: `internal/engine/speculative.go`, `internal/device/cuda_kernels.cu`, `internal/engine/cpu_weights.go`, `internal/engine/engine_cuda.go`
- **Acceptance Criteria**: Target model speculative verification latency drops by $\ge 65\%$; overall generation speedup improves from $1.3\times$ to $>2.2\times$ over non-speculative baseline.

#### Part 14 (Step 14): Unified Batched-GEMM Prefill & Decode Co-Scheduling
- **Objective**: Maximize Tensor Core and SIMD compute density under continuous batching by co-scheduling prefill chunks and decode tokens in a single execution pass.
- **Bottleneck Analysis**:
  - In `internal/engine/continuous_batching.go` and `ForwardBatch`, sequences undergoing prompt prefill and sequences undergoing autoregressive decode are handled either in separate steps or dispatched as separate GEMV / GEMM kernel calls.
  - Decode tokens run as memory-bandwidth-bound matrix-vector multiplications ($M=1$), while prefill chunks run as compute-bound matrix multiplications ($M \gg 1$), resulting in low Tensor Core arithmetic intensity and uneven pipeline bubble distribution.
- **Design & Kernel Architecture**:
  - Implement **Piggyback Continuous Batching** in `ContinuousBatchManager`:
    - Pack $B_{\text{decode}}$ single-token decode requests and $T_{\text{prefill}}$ chunked prompt tokens into a single concatenated activation matrix:
      $$\mathbf{X}_{\text{batch}} \in \mathbb{R}^{(B_{\text{decode}} + T_{\text{prefill}}) \times D}$$
    - Execute all projection layers (Q, K, V, O, Gate, Up, Down) as unified Batched GEMMs ($M = B_{\text{decode}} + T_{\text{prefill}}$), transforming memory-bound decode operations into compute-bound GEMMs that achieve peak Tensor Core TFLOPS.
    - Split attention routing after QKV projection: decode tokens route through `cudaPagedAttentionBatch` with Split-K FlashDecoding, while prefill tokens route through chunked `FlashAttentionPrefill` with causal masking.
- **Files**: `internal/engine/continuous_batching.go`, `internal/engine/engine_cuda.go`, `internal/engine/engine_cpu.go`
- **Acceptance Criteria**: Multi-tenant serving throughput (aggregate tok/s across concurrent streams) increases by $\ge 50\%$; Time-to-First-Token (TTFT) degradation during high-concurrency decode drops by $\ge 40\%$.

#### Part 15 (Step 15): Ahead-of-Time Weight Packing & Cache-Line Blocked Transposition (P1) — [IMPLEMENTED]
- **Objective**: Eliminate runtime matrix transposition and optimize spatial memory locality for CPU and GPU GEMM kernels.
- **Bottleneck Analysis**:
  - In `internal/simd/simd.go:58-64` (`MatMul`), matrix $\mathbf{B}$ is dynamically allocated and transposed on every invocation:
    `bT := make([]float32, colsA*colsB)`
    `for k := 0; k < colsA; k++ { ... bT[j*colsA+k] = b[rowOff+j] }`
  - In a continuous batching workload, this induces massive heap allocation thrashing and redundant memory bandwidth consumption.
  - In quantized execution, weights are accessed in linear row-major order, resulting in cache-line underutilization when traversing matrix columns.
- **Design & Implementation**:
  - Implement Ahead-of-Time (AOT) weight packing during model initialization in `CPUWeights` and `CUDAModel`:
    - For CPU: reorganize weights into SIMD micro-panels ($M_R \times K_R$ tiles, e.g. $16 \times 64$ for AVX-512, $8 \times 32$ for AVX2) aligned to 64-byte L1 cache boundaries.
    - For GPU: pre-transpose quantized weight matrices where required for tensor core memory coalescing (`cublasGemmEx` column-major expectations).
  - Completely eliminate runtime calls to `make()` and transposition loops within `simd.MatMul` by computing directly against the pre-packed representation.
- **Files**: `internal/simd/simd.go`, `internal/engine/cpu_weights.go`, `internal/device/cuda.go`, `internal/device/cpu.go`
- **Acceptance Criteria**: Zero runtime transpose allocations in `MatMul`; L2 cache miss rate in `perf stat` drops by $\ge 35\%$; CPU GEMM latency on $512\times 512$ drops by $\ge 2.5\times$.

#### Part 16 (Step 16): Radix-Tree Prefix Caching & Copy-on-Write Paged KV Block Sharing (P0) — [IMPLEMENTED BUT NOT WIRED — see Step 35]
- **Objective**: Deliver instantaneous Time-to-First-Token (TTFT) and zero duplicate VRAM usage for shared system prompts, few-shot templates, and multi-turn conversations.
- **Implementation Status**:
  - Implemented compressed Patricia trie in `internal/engine/radix_cache.go` with token-span compression, edge splitting, and token prefix matching.
  - Added Copy-on-Write (CoW) page reference tracking: `AttachSequence` and `DetachSequence` manage block reference counts with zero redundant data copying.
  - Implemented LRU eviction of unreferenced leaf nodes (`EvictUnderPressure`) when memory pressure exceeds 0.85.
  - Verified with comprehensive test suite in `internal/engine/radix_cache_test.go` covering edge splitting, multi-turn dialogue branching, and LRU eviction under memory pressure.
- **Files**: `internal/engine/radix_cache.go`, `internal/engine/radix_cache_test.go`, `internal/engine/prompt_cache.go`, `internal/engine/kv_cache_paged.go`
- **Acceptance Criteria**: Multi-turn chat TTFT drops from $>250$ms to $<2$ms (instantaneous prefix hit); memory consumption for concurrent requests sharing a system prompt drops by $\ge 80\%$.

#### Part 17 (Step 17): Build-Tag Clean-Room Architecture & Compile-Time Backend Dispatch
- **Objective**: Eliminate language server (`gopls`) indexing exclusions (`@[current_problems]`) and replace runtime backend branching with zero-overhead compile-time interface dispatch.
- **Diagnostic & Root Cause Analysis**:
  - The IDE reports diagnostics warnings:
    `No packages found for open file cpu.go / simd_default.go. This file may be excluded due to its build tags; try adding "-tags=<build tag>" to your gopls "buildFlags" configuration`.
  - Because `.vscode/settings.json` specifies `"build.buildFlags": ["-tags=cgo,avx512,cuda"]`, files with build tags containing `!cuda` (`internal/device/cpu.go`) or `!avx512` (`internal/simd/simd_default.go`) are excluded by `gopls`.
  - Furthermore, keeping mutually exclusive top-level files in `internal/device/` forces monolithic conditional branching at runtime (e.g. `needsCPU` checks in `engine_cuda.go`), tightly coupling disparate hardware backends and causing maintenance friction.
- **Design & Architecture**:
  - Restructure `internal/device/` into decoupled sub-packages:
    - `internal/device/core`: Device-agnostic interfaces (`BackendDevice`, `Tensor`, `DataType`, memory allocation primitives).
    - `internal/device/cpu`: Pure CPU and SIMD execution paths, always compiled and indexable by `gopls`.
    - `internal/device/cuda`: CUDA runtime, CGo wrapper, and `.cu` kernels compiled under `//go:build cuda`.
    - `internal/device/metal`: Metal runtime and `.metal` shaders compiled under `//go:build darwin && metal`.
  - Define a universal `DeviceBackend` interface (`Alloc`, `Free`, `GEMM`, `Attention`, `Norm`, `RoPE`, `Transfer`).
  - Provide fallback stub implementations with build tags that satisfy package declarations regardless of active tag configuration, guaranteeing zero gopls exclusions.
  - Provide `.vscode/settings.json` multi-target environment presets (`cuda-dev`, `cpu-dev`, `metal-dev`) with automated build flag synchronizers.
- **Files**: `.vscode/settings.json`, `internal/device/`, `internal/simd/`, `internal/engine/engine_cuda.go`, `internal/engine/engine_cpu.go`
- **Acceptance Criteria**: Zero gopls warnings or diagnostic exclusions across all files in the workspace; clean compile-time dispatch with zero runtime branching penalties; 100% CI pass rate.

#### Part 18 (Step 18): On-Device GPU Top-P / Top-K Sampling & Penalty Kernel (P0) — [IMPLEMENTED]
- **Objective**: Eliminate device-to-host (D2H) PCIe transfer of vocabulary logits and offload repetitive CPU sorting to parallel GPU warps.
- **Implementation Status**:
  - Implemented fused CUDA kernel in `internal/device/cuda_sampling.cu`:
    1. Parallel repetition, frequency, presence, and temperature penalty kernel directly in VRAM.
    2. Cooperative warp reduction argmax kernel for deterministic greedy sampling.
    3. 256-thread bitonic top-K partition with in-register cumulative sum and curand/Philox top-P sampling.
  - Exposed via `SampleTopKTopP` in `internal/device/cuda.go` and CGo bridge in `internal/device/cuda_wrapper.c`.
  - Integrated into `internal/engine/engine_cuda.go` decode loop, copying only 4 bytes (int32 token ID) to host instead of 600 KB logits.
  - Verified with `internal/device/cuda_sampling_test.go`: benchmark achieves ~44 µs with 0 allocs/op and 0 B/op on 32k vocabulary.
- **Files**: `internal/device/cuda_sampling.cu`, `internal/device/cuda.go`, `internal/device/cuda_wrapper.c`, `internal/engine/engine_cuda.go`, `internal/device/cuda_sampling_test.go`
- **Acceptance Criteria**: Logit D2H PCIe bandwidth reduced by $>99.9\%$; token sampling latency drops from $\sim 350$ μs to $<15$ μs; 0 heap allocations per sampling step.

#### Part 19 (Step 19): Fused Warp-Shuffle Reductions for RMSNorm and Softmax CUDA Kernels (P0) — [IMPLEMENTED]
- **Objective**: Maximize memory bandwidth saturation in normalization layers by replacing serial per-thread loops with register-level cooperative warp shuffles.
- **Implementation Status**:
  - Modernized `rmsnorm_kernel` and `softmax_kernel` in `internal/device/cuda_kernels.cu`:
    - Implemented cooperative warp reductions: `warpReduceSum`, `blockReduceSum`, `warpReduceMax`, `blockReduceMax`.
    - Added 128-bit `float4` vectorized loads and stores for 16-byte aligned hidden states, saturating memory bandwidth.
    - Replaced sequential scalar loops with fast online two-pass softmax and cooperative warp-level norm scaling.
  - Exposed `ctx.Softmax` in `internal/device/cuda.go`.
  - Validated in `internal/device/cuda_rmsnorm_softmax_test.go`: RMSNorm executes at 3.4 µs on 4096-dim hidden states; Softmax executes at 6.0 µs.
- **Files**: `internal/device/cuda_kernels.cu`, `internal/device/cuda.go`, `internal/device/cuda_rmsnorm_softmax_test.go`
- **Acceptance Criteria**: RMSNorm kernel runtime on $4096$-dim hidden states drops by $\ge 4.5\times$; kernel achieves $\ge 85\%$ of peak theoretical VRAM bandwidth.

#### Part 20 (Step 20): Fused W4A16 & W8A16 Direct Quantized GEMV without Intermediate Weight Materialization
- **Objective**: Eliminate intermediate global memory buffers by dequantizing weights directly in registers during GEMV computation.
- **Bottleneck Analysis**:
  - Autoregressive decode is strictly memory-bandwidth bound ($M=1$). In `cuda_kernels.cu:145-165` (`dequant_q8_0_kernel`, `dequant_q4_k_kernel`), quantized weights are first unpacked into full FP32/FP16 tensors in VRAM before cuBLAS GEMM executes.
  - This forces a two-pass memory round-trip: write dequantized weights to VRAM, then immediately read them back into Tensor Cores, doubling memory bus pressure and inflating peak VRAM usage.
- **Design & Kernel Architecture**:
  - Implement direct fused GEMV kernels in `internal/device/cuda_quant_gemv.cu`:
    - Load quantized weights (`uint8_t` nibbles for Q4_K / Q8_0) directly into 32-bit registers.
    - Unpack 4-bit / 8-bit integer weights on-the-fly using bit-manipulation intrinsics (`__byte_perm`, `lop3.b32`).
    - Cache the activation vector $\mathbf{x} \in \mathbb{R}^K$ in shared memory once per thread block.
    - Execute fused multiply-accumulate via `__dp4a` (for int8 dot products) or `mma.sync.aligned.m16n8k16` Tensor Core fragments.
    - Apply FP16 block scales and min offsets at the warp reduction boundary before accumulating into FP32 output registers.
- **Files**: `internal/device/cuda_quant_gemv.cu`, `internal/device/cuda.go`, `internal/engine/engine_cuda.go`
- **Acceptance Criteria**: VRAM read traffic during token generation drops by $50\%$ (Q8_0) and $75\%$ (Q4_K); single-token decode latency improves by $\ge 2.2\times$.

#### Part 21 (Step 21): Pinned Zero-Copy Memory & Asynchronous Double-Buffered Batch Pipeline
- **Objective**: Overlap host batch preparation with device execution and eliminate GC overhead in continuous batching.
- **Bottleneck Analysis**:
  - In `internal/engine/engine_cuda.go:414-438` (`ForwardBatch`), slice allocations `make([]int32, numTokens)` are created on every generation step for `tokenPositions` and `tokenToSeq`, triggering Go runtime GC pauses.
  - Synchronous `LoadFrom` calls stall the CUDA stream waiting for PCIe host-to-device transfers before kernel launches can begin.
- **Design & Implementation**:
  - Pre-allocate dual-buffered pinned host memory (`cudaHostAllocMapped` / `cudaHostAllocWriteCombined`) inside `DeviceContext`.
  - Implement double-buffered asynchronous stream execution:
    - Stream 0: executes Layer $L$ forward pass kernels on GPU.
    - Stream 1: asynchronously prefetches Layer $L+1$ descriptors and writes batch metadata into mapped host pointers without CPU-GPU synchronization.
  - Replace dynamic slice allocation with pre-allocated circular descriptor ring buffers.
- **Files**: `internal/device/memory.go`, `internal/device/cuda.go`, `internal/engine/engine_cuda.go`, `internal/engine/continuous_batching.go`
- **Acceptance Criteria**: Zero Go heap allocations in `ForwardBatch`; GPU compute utilization in continuous batching increases from $\sim 65\%$ to $>90\%$.

#### Part 22 (Step 22): Sliding Window Attention with Attention Sink Tokens (StreamingLLM) (P0) — [IMPLEMENTED]
- **Objective**: Enable infinite-length streaming generation with constant memory footprint and zero perplexity degradation.
- **Implementation Status**:
  - Implemented StreamingLLM protocol in `internal/engine/kv_cache_paged.go`:
    - Added `SetStreamingLLM(seqID, sinkBlocks, windowBlocks)` and `GetStreamingLLM(seqID)`.
    - Dedicates initial `sinkBlocks` (default 4 blocks = 64 tokens) indefinitely as attention sinks.
    - Implemented rolling FIFO ring buffer in physical page allocations, evicting the oldest non-sink block without memory growth.
    - Clamped block table access in `internal/device/cuda_kernels.cu` (`numBlocksNeeded = min(numBlocksNeeded, maxBlocks)`) to protect against sliding window overread.
  - Verified with unit tests in `internal/engine/streaming_llm_test.go`: streamed 2,000 tokens through a 5-block cache with strictly constant 5 physical blocks.
- **Files**: `internal/engine/kv_cache_paged.go`, `internal/device/cuda_kernels.cu`, `internal/engine/streaming_llm_test.go`
- **Acceptance Criteria**: Engine streams $>100,000$ consecutive tokens per sequence with strictly constant VRAM usage; perplexity remains stable without degradation.

#### Part 23 (Step 23): Native FP8 (E4M3FN / E5M2) Quantized Paged KV Cache with Per-Head Dynamic Scaling (P0) — [IMPLEMENTED]
- **Objective**: Halve KV cache VRAM footprint and double maximum serving concurrency on modern GPUs.
- **Implementation Status**:
  - Implemented native FP8 KV cache allocation and dynamic scaling in `internal/engine/kv_cache_paged.go` (`DataTypeFP8`).
  - Added `GetBlockScales(layer)` returning dynamic per-block scales (`kScales`, `vScales`).
  - Wired into `PagedAttentionQuantized` in `internal/device/cuda.go` and `internal/engine/engine_cuda.go`, executing 8-bit quantized attention directly on GPU Tensor Cores.
- **Files**: `internal/engine/kv_cache_paged.go`, `internal/device/cuda_kernels.cu`, `internal/device/cuda.go`, `internal/engine/engine_cuda.go`
- **Acceptance Criteria**: KV cache memory footprint reduced by $50\%$ vs FP16; serving batch concurrency doubles with $<0.03$ perplexity delta on standard benchmarks.

#### Part 24 (Step 24): Intel AMX (Advanced Matrix Extensions) TILE Acceleration for CPU Inference
- **Objective**: Leverage dedicated hardware matrix tiles on modern Intel Xeon processors (Sapphire Rapids, Emerald Rapids, Granite Rapids) for order-of-magnitude CPU prefill acceleration.
- **Bottleneck Analysis**:
  - On modern server CPUs, AVX-512 VNNI reaches $\sim 2\text{ TFLOPS}$ on INT8, but Intel AMX provides dedicated hardware tile registers supporting up to $2048$ INT8 MACs or $1024$ BF16 MACs per cycle per core ($>30\text{ TFLOPS}$ per socket).
- **Design & Kernel Architecture**:
  - Implement AMX Tile matrix multiplication in `internal/simd/kernels_amx.c`:
    - Request AMX permission via Linux system call `arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILECFG)`.
    - Configure tile registers (8 tiles of $16\times 64$ bytes) using `_tile_loadconfig`.
    - Implement blocked GEMM inner loop with `_tile_dpbusd` (INT8) and `_tile_dpbf16ps` (BF16).
    - Add AMX capability flags (`amx_tile`, `amx_int8`, `amx_bf16`) to CPU feature detector in `cpuinfo.go`.
- **Files**: `internal/simd/kernels_amx.c`, `internal/simd/cpuinfo.go`, `internal/simd/simd.go`
- **Acceptance Criteria**: CPU GEMM prefill speedup on AMX-capable Xeon hardware improves by $\ge 3.5\times$ over AVX-512 baseline.

#### Part 25 (Step 25): Multi-LoRA Batch Fusion & Grouped Low-Rank Scatter-Gather Dispatch
- **Objective**: Serve heterogeneous multi-tenant requests with distinct LoRA adapters in a single continuous batch with zero throughput penalty.
- **Bottleneck Analysis**:
  - Currently in `internal/engine/lora.go`, requests requiring different LoRA adapters either cannot be batched together or must execute separate sequential kernel launches per adapter, destroying Tensor Core parallelism.
- **Design & Kernel Architecture**:
  - Implement Batched Multi-LoRA Fusion in `cuda_lora.cu`:
    - Pack active sequence adapter IDs into a compact batch mapping: `adapter_ids[B]`.
    - Execute the base model projection GEMM $\mathbf{Y}_{\text{base}} = \mathbf{X} \mathbf{W}_{\text{base}}$ as a single unified batch for all sequences.
    - Dispatch a grouped fused scatter-gather kernel for the low-rank projections:
      $$\mathbf{Y}_i = \mathbf{Y}_{\text{base}, i} + \frac{\alpha_k}{r_k} (\mathbf{X}_i \mathbf{A}_k) \mathbf{B}_k$$
      where $\mathbf{A}_k \in \mathbb{R}^{D \times r_k}$ and $\mathbf{B}_k \in \mathbb{R}^{r_k \times D_{\text{out}}}$ are gathered from adapter pointer arrays in parallel thread blocks.
- **Files**: `internal/engine/lora.go`, `internal/device/cuda_lora.cu`, `internal/engine/continuous_batching.go`
- **Acceptance Criteria**: Batching 8 distinct LoRA adapters incurs $<5\%$ latency overhead compared to a homogeneous batch with no adapters.

#### Part 26 (Step 26): Persistent Grouped GEMM for Mixture of Experts (MoE)
- **Objective**: Eliminate kernel launch overhead and warp divergence in MoE routing by fusing expert execution into a persistent work-queue grid.
- **Bottleneck Analysis**:
  - In `internal/device/cuda_moe.cu`, expert execution loops sequentially over active experts or launches small individual kernels per expert. For models with 8 to 64 experts (e.g. Mixtral, Qwen MoE), small per-expert token counts cause low SM occupancy and excessive launch overhead.
- **Design & Kernel Architecture**:
  - Implement a persistent Grouped GEMM kernel in `internal/device/cuda_moe_grouped.cu`:
    - Sort tokens by expert ID using warp-level prefix sums into contiguous expert buckets.
    - Launch a single persistent CUDA grid of thread blocks that pull dynamic tiles from an atomic global task queue:
      $$\text{TaskDescriptor}\{e, \text{tile\_m}, \text{tile\_n}, \text{tile\_k}\}$$
    - Eliminate intermediate expert barrier synchronizations; stream outputs directly into top-$k$ weighted sum accumulation.
- **Files**: `internal/device/cuda_moe.cu`, `internal/device/cuda_moe_grouped.cu`, `internal/engine/layer_moe.go`
- **Acceptance Criteria**: MoE forward pass latency on Mixtral 8x7B drops by $\ge 40\%$; SM compute occupancy exceeds $80\%$ across all expert counts.

#### Part 27 (Step 27): Self-Speculative Early Exit with Confidence-Gated Verification
- **Objective**: Achieve $1.8\times - 2.2\times$ speculative decoding speedup without hosting an external draft model.
- **Bottleneck Analysis**:
  - External speculative drafting requires loading a second model into VRAM, maintaining dual KV caches, and coordinating separate model topologies.
- **Design & Architecture**:
  - Implement Layer-Skip Self-Speculative Decoding in `internal/engine/speculative_self.go`:
    - Attach an early-exit linear head to an intermediate transformer layer (e.g. Layer 8 of 32).
    - **Draft Pass**: Forward through layers 0–8, evaluate early-exit logits, and draft $K=4$ candidate tokens with low latency.
    - **Verification Pass**: Forward candidate tokens through the full model layers (9–31) in a single batched shot using tree attention.
    - If early-exit confidence exceeds threshold $\tau$ ($\text{entropy}(p_{\text{early}}) < \tau$), verify and accept tokens without full model execution.
    - Share the identical `PagedKVCache` blocks between draft and verification passes.
- **Files**: `internal/engine/speculative_self.go`, `internal/engine/engine_cuda.go`, `internal/engine/engine_cpu.go`
- **Acceptance Criteria**: Autoregressive decode speedup of $\ge 1.75\times$ on standard benchmarks without requiring external draft model weights or extra VRAM.

---

### Phase 6: Ship-It Audit — Wiring, Dead Code & Correctness Gaps (target v0.4.0) — Priority: P0

> **Origin.** Added October 2026 after a full-tree static + dynamic analysis of
> commit `8dd7f91`. Every finding below was verified by re-running the command
> shown. The headline result at the time of the audit: **the repo's CI `lint`
> job was red on `main`, and the default `make cpu` build shipped a binary that
> exited(1) immediately.** Much of Phases 2–5 is implemented and well-tested
> *in isolation*, but is not connected to any shipped entrypoint.
>
> **A later finding worth calling out on its own: the macOS Metal build was
> broken.** Two independent defects, both invisible to a Linux CI run:
>
> 1. `internal/engine/engine.go` and `internal/engine/sampler.go` both
>    declared `ErrModelSwapped`. In a `darwin + metal` build both files are
>    compiled, so the package failed with *"redeclared in this block"*.
> 2. `internal/device/cpu.go` carried the build tag
>    `(!cuda && !metal && !tpu) || !amd64 || !cgo || (!linux && !darwin)`.
>    The `!amd64` clause is **true on arm64**, so on Apple Silicon the file
>    was compiled *alongside* `metal.go` — and both declare `Context`,
>    `Tensor`, `NewContext` and `AllocatedBytes`. Commit `6d712dd` had
>    changed the tag from the correct `!cuda && !metal` to this form.
>
> The CI `metal-build` job runs on `macos-latest`, which is arm64, so it
> should have been red — but the audit could not confirm the runner
> architecture, and no Linux job compiles the darwin+metal configuration.
> Both defects are fixed, and `scripts/dupcheck.sh` now fails the build on
> any duplicate package-level declaration in any of the seven supported
> configurations (210 package × configuration combinations).

> ### Progress
>
> **Batch 1 (Steps 28, 31, 32, 34), Batch 2 (Steps 29, 30, 41, 42, 43,
> 44, 45, 48), Batch 3 (Steps 33, 49), Batch 4 (Steps 37, 38, 39, 40, 46,
> 47), Batch 5 (Steps 31, 35, 36, 44, 48), Batch 6 (Steps 35, 44), Batch 7
> (Steps 44, 47), Batch 8 (Step 40), Batch 9 (Step 40), Batch 10 (Step 35)
> and Batch 11 (Step 35), Batch 12 (Step 35) are DONE** — see the
> status markers on each step.
> Verified locally after the change:
>
> | Check | Command | Result |
> |---|---|---|
> | Lint | `golangci-lint run --timeout=10m` | **0 issues** (was 40) |
> | Vacuous tests | `go run ./scripts/vacuous_tests` | **OK, 0 new** (was 2) |
> | Duplicate decls | `./scripts/dupcheck.sh` | **OK, 210 combinations** (new gate) |
> | Workflow lint | `actionlint .github/workflows/*.yml` | **OK** (new gate) |
> | Vet | `go vet ./...` | clean |
> | Tests | `go test ./...` | all pass |
> | Race | `go test -race ./internal/...` | clean (all 19 packages now gated) |
> | AVX-512 | `go test -tags avx512 ./internal/simd/` | **passes** (was SIGILL) |
> | CUDA | `go vet -tags cuda ./...` + `go test -tags cuda ./internal/engine/...` | **pass** (was a compile error) |
> | WebUI | `cd cmd/webui && go test -tags webui ./...` | **all pass incl. e2e** (was 0 tests) |
> | CPU binary | `make cpu-linux-amd64 && ./bin/quarrel-linux-amd64 -version` | **works** (was a stub) |
> | Build matrix | 84 GOOS/GOARCH/CGO/tag combinations | **exactly 1 `main` each** |
> | Backend matrix | 7 GOOS/GOARCH/CGO/tag combinations | **exactly 1 backend each** |
> | REST API | `TestStartAPIServer_ServesEndpoints` | **serves** (was never started) |
> | TurboQuant SIMD | `go test -bench PolarQuant ./internal/simd/` | **7.5x faster** (kernels were never called) |
> | CPU/GPU parity | `TestTurboQuantEncode_CPUMatchesCUDASemantics` | **matches the CUDA kernel** (it did not) |
> | `internal/engine` coverage | `go test -cover ./internal/engine/` | **62.8%** (was 51.0%) |
> | Consolidated suite | `./scripts/run_all_tests.sh` | **12/12 gates pass** |
> | WebUI inference | `cmd/webui/engine/adapter_protocol_test.go` | **works** (hung on every backend) |
> | Chat templates | `internal/engine/chat_template_render_test.go` | **renders** (produced only "<s>") |
> | Metrics reference | `internal/metrics/docs_test.go` | **verified both ways** (73 metrics were wrong) |
> | Browser e2e | `go test -run TestPlaywrightE2E ./internal/api/` | **runs** (the driver could not be downloaded) |
> | Paged KV batch | `go test -run TestPagedKVCacheUpdateBatch ./internal/engine/` | **works** (it panicked on every backend) |
> | `internal/engine` coverage | `go test -cover ./internal/engine/` | **68.1%** (was 51.0% at the start) |

#### Diagnostic Summary (as found, before Batch 1)

| Check | Command | Result |
|---|---|---|
| Default build | `go build -o q ./cmd/quarrel && ./q` | prints "No GPU backend detected", `exit 1` |
| Lint | `golangci-lint run --timeout=10m` | **40 issues** (gofmt 17, staticcheck 14, errcheck 6, ineffassign 3) |
| Vacuous tests | `go run ./scripts/vacuous_tests` | **2 violations**, `exit 1` |
| Vet | `go vet ./...` | clean |
| Race | `go test -race ./internal/...` | clean (but CI only gates 5 of 19 packages) |
| Tests | `go test -cover ./internal/...` | all pass |
| Modules | `go mod tidy -diff` | clean (both root and `cmd/webui`) |

---

#### Step 28: `cmd/quarrel` CPU build is a 4-line stub that exits(1) — P0 — [DONE]

**Status: COMPLETE.** Added `cmd/quarrel/main_cpu.go` (build tag `!cuda && !metal && !tpu`) implementing full CPU inference: model resolution, GGUF load, engine construction, sampling, streaming, and a `/metrics` endpoint. Added `cmd/quarrel/version.go` with the `Version`/`BuildTime` vars that the Makefile's `-X` stamps were silently dropping. `--help`/`-version` now exit 0; missing `-model` exits 1.

The stub was narrowed to `cmd/quarrel/main_stub.go` with the **exact negation** of the three real backend mains, so it only compiles when a backend tag is requested that cannot build on the host. `main_tpu.go`/`main_metal.go` gained `!cuda`/`!cuda && !tpu` guards so no combination can yield two `main`s.

Verified: the built binary grew 2.4 MB → 28.6 MB and now contains engine code (`strings … | grep -c internal/engine` → 389, was 0). An 84-case matrix over `{linux,darwin,windows} × {amd64,arm64} × {cgo on/off} × 7 tag sets` yields **exactly one `main` in every case** — this also caught a pre-existing `cuda+tpu` double-`main` collision.

**Finding.** `cmd/quarrel/main.go:1` is gated `//go:build linux && amd64 && cuda && cgo`. On the default build only `cmd/quarrel/main_stub.go` compiles:

```
$ go list -f '{{.GoFiles}}' ./cmd/quarrel
[main_stub.go]
$ go build -o quarrel ./cmd/quarrel && ./quarrel ; echo $?
Longbow-Quarrel: No GPU backend (CUDA or Metal) detected.
1
```

No engine code is linked in (`strings quarrel | grep -c internal/engine` → `0`).

This breaks **every** CPU shipping path, because none of them pass a build tag:

| Location | Command |
|---|---|
| `Makefile:51-54` `cpu-linux-amd64` | `go build ... ./cmd/quarrel` |
| `Makefile:56-59` `cpu-linux-arm64` | `CGO_ENABLED=0 ... ./cmd/quarrel` |
| `Makefile:61-70` `cpu-darwin-*` | same |
| `Makefile:117-134` `simd-*` | `-tags cgo` only |
| `Dockerfile.cpu:35` | `go build -trimpath -o /out/quarrel ./cmd/quarrel` |
| `Dockerfile.metal:48` | `go build -o quarrel-metal ./cmd/quarrel` (no `-tags metal`) |
| `ci.yml:99` build job | `go build -v ./cmd/quarrel` |

Downstream damage:
- `Dockerfile.cpu:71-75` — `HEALTHCHECK ["/app/quarrel","--help"]` + `ENTRYPOINT` always fail → the image `ghcr-publish` pushes as `:latest` and that `helm/quarrel/values.yaml:3-7` defaults to is permanently `unhealthy`.
- `ci.yml:99` "Build" step verifies nothing; `cmd/quarrel` reports `coverage: 0.0%`.

**Plan.**
1. Add `cmd/quarrel/main_cpu.go` with `//go:build (!cuda && !metal && !tpu) || !cgo` that constructs the CPU engine, wires flags, and runs inference — mirroring `main.go` minus the CUDA-only calls.
2. Narrow `main_stub.go` to genuinely unsupported platforms only (`!cgo`, or a platform with no backend at all).
3. Add `main.Version` / `main.BuildTime` vars so the `-X` stamps in `Makefile` stop being silent no-ops (they currently match nothing: `grep -rn "^var" cmd/*/*.go`).
4. Add a `-help` flag handler; `flag` currently exits **2** on `--help`, which `Dockerfile.cpu/nvidia/tpu/metal` all invoke.
5. Add a CI assertion that a built binary actually starts (e.g. `./quarrel --help` exits 0) so this cannot regress.

**Acceptance Criteria:** `make cpu` produces a working CPU binary; `docker run longbow-quarrel:cpu --help` exits 0; `go build ./cmd/quarrel && ./quarrel --help` exits 0.

---

#### Step 29: The OpenAI-compatible REST API is never started — P0 — [DONE]

**Status: COMPLETE — the API is now served by every backend.**

`api.InitServer` was changed to **return an `*http.ServeMux`** instead of registering on `http.DefaultServeMux`. Mutating the global default mux is what made the API awkward to wire (a second call panics with "multiple registrations for /healthz") and is one reason it was never started. Callers now own the mux.

Wired into all four binaries behind a new `-api-addr` flag (default `:8080`, empty disables it):

| Binary | Callback |
|---|---|
| `main_cpu.go` | `device.CPUAllocatedBytes` |
| `main.go` (CUDA) | `device.CUDAAllocatedBytes` |
| `main_metal.go` | `device.AllocatedBytes` |
| `main_tpu.go` | `device.TPUAllocatedBytes` |

The startup logic lives in `cmd/quarrel/apiserver.go` (`startAPIServer`) so it is testable, and `cmd/quarrel/apiserver_test.go` asserts that `/healthz` answers 200 with a JSON body and that `/v1/chat/completions` is routed (not 404) over a real listener. `internal/api/routing_test.go` additionally asserts that all six documented endpoints are registered, that `InitServer` is repeatable, and that the API works over real HTTP.

`ListAdaptersHandler` no longer returns a hardcoded placeholder: `LoRAManager.ListAdapters` and `CPUEngine.ListAdapters` were added, so the endpoint reports the adapters that are actually loaded (and returns 501 for an engine that cannot list them).

**Acceptance Criteria:** `quarrel -api-addr :8080` serves `/v1/chat/completions` and `/healthz` — now covered by tests.

---

#### Step 30: `internal/api` — health monitor is a no-op busy-loop — P1 — [DONE]

**Status: COMPLETE.** `runResourceMonitor` was a 1-second ticker that computed `s.getUsedMemory()` and discarded it. It now publishes the value to Prometheus via a new `host_memory_allocated_bytes` gauge (`metrics.RecordHostMemory`). The ticker is now stopped on exit instead of leaking.

---

#### Step 31: `cmd/webui` — CI runs zero tests; CPU adapter is a silent stub — P0 — [DONE]

**Finding 1 (CI) — FIXED.** `ci.yml:139-141` computed the package list without the build tag:

```yaml
pkgs=$(go list ./... | grep -v '/tests/e2e')   # missing -tags webui
```

Now `pkgs=$(go list -tags webui ./...)`. The job discovers 5 packages instead of 1, and `cmd/webui/handlers` tests actually run.

Every file in `cmd/webui` except `cmd/webui/engine/` is behind `//go:build webui`, so `go list ./...` without the tag returns exactly one package:

```
$ cd cmd/webui && go list ./...
github.com/23skdu/longbow-quarrel/cmd/webui/engine
$ go test -count=1 -tags webui $(go list ./... | grep -v /tests/e2e)
?   .../cmd/webui/engine   [no test files]
```

`cmd/webui/handlers/handlers_test.go` (13 KB) and `fuzz_test.go` (4.5 KB) **never run in CI**. The comment at `ci.yml:134-138` diagnoses the tag problem correctly but the fix omits the flag. Fix: `pkgs=$(go list -tags webui ./... | grep -v '/tests/e2e')`.

**Finding 2 (stub adapter).** `cmd/webui/engine/adapter_cpu.go` (`//go:build !(darwin&&metal) && !(linux&&amd64&&cuda&&cgo)` — every default build):

| Line | Behaviour |
|---|---|
| `:52-62` | `handleRequest` returns `InferenceResponse{Complete: true}` with **no tokens** |
| `:64-79` | `Infer` returns `nil, nil` on queue-full — silent success with a nil channel |
| `:86` | `LoadModel` → `return nil`; never loads anything |
| `:81-84` | `ListModels` → empty slice |
| `:91-94` | `UnloadModel` / `HotSwapModel` → no-op |

Only `adapter_cuda.go:12-14` and `adapter_metal.go:12-14` import the real `internal/engine`. So the WebUI — the only working HTTP surface — answers `POST /v1/chat/completions` with **HTTP 200 and an empty completion** on any CPU host.

**Status: COMPLETE.** `adapter_cpu.go` now loads a real model and runs real
inference through the CPU engine, mirroring the CUDA and Metal adapters.

**Two further defects were found while doing it, and they affected the CUDA
and Metal adapters too:**

1. **The inference protocol was broken on every backend.** The caller does
   `inner := <-outer` and then ranges over `inner`, so the adapter has to
   *deliver* the inner channel on the outer one. The CUDA and Metal adapters
   returned an outer channel that nothing ever wrote to, and their error path
   tried to *take* the inner channel back **out** of it. The request's
   `ResponseChan` was never even set (it was nil), so the send blocked
   forever. The WebUI's generate endpoint therefore **hung on every request**
   on every backend.

   `Infer` now sets `req.ResponseChan` to the outer channel it returns, and
   `sendError` delivers a finished inner channel instead of removing one.

2. **A failure was indistinguishable from an empty completion.** The
   response carried no error, so a request that could not be served (the
   model failed to load) produced HTTP 200 and an empty body.
   `InferenceResponse` now has an `Error` field, and the handler returns
   HTTP 500 with the reason.

`Infer` also returns `ErrQueueFull` instead of a nil channel and a nil error
when the queue is saturated, in all three adapters.

`cmd/webui/engine/adapter_protocol_test.go` is the regression guard for all
of this: it asserts that the inner channel is delivered, that the response is
marked complete, and that it carries the error.

**Finding 3 (e2e never compiled) — FIXED, and it exposed three latent defects.** `cmd/webui/tests/e2e/e2e_test.go` referenced `handlers.Config` / `handlers.ModelInfo`, which moved to `config.Config` / `engine.ModelInfo`. After fixing the imports the package compiled — and immediately failed, revealing bugs that had been hidden for as long as the suite was excluded:

| Latent defect | Fix |
|---|---|
| **No request body size limit.** `TestRequestBodySizeLimits` expected 413 and got 400. `grep MaxBytesReader` found nothing — an unbounded body was decoded straight into memory. | Added `maxRequestBodyBytes` (1 MiB) + `decodeJSONBody` using `http.MaxBytesReader` in `handlers/inference.go`, applied to **both** `GenerateHandler` and `StreamHandler`. `StreamHandler` now reads the body *before* committing SSE headers so an oversized request gets a clean status instead of a 200 with an `event-stream` body. |
| **Rate limit ignored its configured value.** `TestRateLimiting` registered a limit of 5 but the 6th request passed. `APIKeyManager.CheckRateLimit` hardcoded `count >= 100` and never consulted `apiKey.RateLimit`. | `handlers/auth.go` now reads the per-key `RateLimit`, falling back to a named `defaultRateLimit = 100`. |
| **Stale test expectations.** `TestHTTPMethodsEnforcement` expected GET `/api/models` → 405, but that endpoint is GET-only by design; and `TestCORSHeadersPresent` expected no header when `NewCORSMiddleware` intentionally defaults an empty allow-list to `*`. | Tests corrected to assert real semantics, and strengthened: a new case asserts an **explicit** allow-list rejects a non-listed origin, and the body-limit test now passes. |

`cmd/webui/tests/e2e` is no longer excluded in CI, and `go vet -tags webui ./...` is clean.

---

#### Step 32: AVX-512 is never compiled — the flagship SIMD path is orphaned — P0 — [BUILD FIXED; REVEALED A SIGILL BUG]

**Finding.** `internal/simd/avx512.go:1` is `//go:build amd64 && cgo && avx512` (685 LOC, plus `kernels_avx512.c` 405 LOC, `avx512_test.go`, `avx512_fuzz_test.go`). **No build target, Dockerfile, Makefile recipe, or CI job ever passes `-tags avx512`:**

```
$ grep -rn "avx512" Makefile Dockerfile.cpu .github/workflows/*.yml
Makefile:39:  SIMD_LEVEL ?= auto  # declared, never referenced again
Makefile:117: simd-avx512:        # target name only ...
Makefile:119:   CGO_ENABLED=1 $(GO) build -tags cgo ...   # <-- cgo, not avx512
```

Result — every shipped CPU binary uses the scalar fallback file:

```
$ go list -f 'GoFiles: {{.GoFiles}}' ./internal/simd
GoFiles: [avx2_scalar.go cpuinfo.go metrics.go simd.go simd_default.go turboquant_avx.go worker_pool.go]
Ignored: [avx512.go avx512_test.go avx512_fuzz_test.go turboquant_neon.go ...]
```

`simd_default.go` (`//go:build !amd64 || !cgo || !avx512`) supplies `simd.RMSNorm`, `RoPE`, `FusedAttention`, `Matmul`, `Softmax`, `SwiGLU` as **scalar Go reference loops** — and `simd.RMSNorm` is called ~17 times per layer on the CPU hot path.

This invalidates: `README.md:9,58` ("optimized CPU SIMD vectorization (AVX-512, AVX2, ARM NEON)"), `docs/performance.md:155-171` (12 ns/op AVX-512 "certified" figures — no artifact reproduces them), `docs/comparison.md:90`, and `README.md:109-111` (`go test -bench=AVX512` matches **0** benchmarks and passes vacuously).

Note the code itself is fine — `go build -tags avx512 ./internal/simd/` succeeds. It is simply never invoked.

**Status: PARTIALLY COMPLETE.** The build wiring is fixed:

- `Makefile:117-121` `simd-avx512` now passes `-tags "cgo avx512"` (was `-tags cgo`).
- The dead `SIMD_LEVEL` variable is now **consumed**: a `SIMD_TAGS_*` table resolves it to real tags, and `cpu-linux-amd64` builds with `$(SIMD_TAGS)`. `auto` enables AVX-512 only when the build host reports `avx512f`, matching the runtime `detectCPU()` logic.
- Verified: `make simd-avx512` produces `bin/quarrel-linux-amd64-avx512` with 4 extra VNNI symbols vs the default build; `make cpu-linux-amd64` produces a working 21 MB static binary with correct `-X` version stamping.

**Turning the tag on exposed a real crash bug.** `go test -tags avx512 ./internal/simd/` died immediately with:

```
SIGILL: illegal instruction
instruction bytes: 0x62 0xf3 0xfd 0x28 0x39 0xc0 ...   (EVEX-prefixed)
simd.dotQ8_0VNNIC → simd.VecDotQ8_0_VNNI ← TestVecDotQ8_0_VNNI
```

Root cause: `internal/simd/cpuinfo.go:43` treated **`avx_vnni`** as implying AVX-512 VNNI:

```go
hasAVXVNNI = containsFlag(flags, "avx_vnni") || containsFlag(flags, "avx512_vnni")
```

`avx_vnni` is the **VEX-encoded** VNNI extension (Alder Lake, Sapphire Rapids) and does *not* imply AVX-512, but the kernels it gated in `avx512.go` are EVEX-encoded. On any AVX-VNNI-without-AVX512 host — including CI runners — this is a hard SIGILL. Fixed to require `avx512_vnni`, with a regression test asserting that `avx_vnni` alone does **not** enable it. `go test -tags avx512 ./internal/simd/` now passes.

**Still open:** the docs corrections below, and a CI job that compiles/tests `-tags avx512`.

---

#### Step 33: TurboQuant "vectorized SIMD" is pure Go — C kernels are never called — P0 — [DONE]

**Status: COMPLETE — and the investigation turned up two correctness bugs
that the documentation's claim was hiding.**

**Finding.** `README.md:41` and `docs/performance.md:61-68` claimed AVX-512 /
AVX2 / NEON vectorized TurboQuant with 16-lane FMA. In fact
`internal/simd/turboquant_avx.go`, `turboquant_avx2.go` and
`turboquant_neon.go` were **pure Go** and said so in their own comments
("Fallback to generic when CGO functions unavailable"). The C kernels existed
(`polar_quant_avx512`, `qjl_transform_avx512`, `polar_quant_avx2`,
`qjl_transform_avx2`, `polar_quant_neon`, `qjl_transform_neon`) but **no Go
file referenced them**:

```
$ grep -rn "polar_quant\|qjl_transform" --include=*.go .
(no output)
```

**The two bugs this uncovered.** There were **three** Go implementations of
`PolarQuantSIMD` — `turboquant_avx.go` (amd64+cgo), `turboquant_nocgo.go`
(no cgo) and `turboquant_neon.go` (arm64+cgo) — and they had **drifted apart**:

| Implementation | Clamps `q`? | Residual returned in |
|---|---|---|
| `turboquant_avx.go` (amd64 + cgo) | **no** | rotated space |
| `turboquant_nocgo.go` (no cgo) | yes | original space |
| `turboquant_neon.go` (arm64 + cgo) | yes | rotated space |
| C `polar_quant_avx512/avx2/neon` | yes | original space |
| **CUDA `turboquant_encode_kernel`** | **yes** | **original space** |

So the **x86 CPU path produced a different KV cache from the GPU path** for
the same input: the CPU encoder neither clamped the quantized value nor
rotated the residual back, while the CUDA kernel — the path the release notes
describe as "verified end-to-end numerical precision on hardware" — did both.
The decoder in `internal/device/cpu.go` adds the QJL contribution in original
space, which only makes sense with the CUDA convention, so the CPU encoder
was the odd one out.

The existing tests could not see this because they all used an **identity
rotation matrix**, and with `R = I` the inverse rotation `R^T` is the
identity too — "residual in rotated space" and "residual in original space"
are then indistinguishable.

**A third bug, in the C kernels themselves.** Their vectorized paths used
`_mm256_round_ps(..., _MM_FROUND_TO_NEAREST_INT)` / `vrndnq_f32`, which round
ties **to even**, while Go's `math.Round`, C's `roundf()` and CUDA's
`roundf()` all round ties **away from zero**. The C kernels therefore
disagreed with the GPU path for inputs landing exactly on a `.5` boundary —
and with the scalar tail of their own function, which uses `roundf()`. All
three kernels now detect ties explicitly and round them away from zero.

**Fix.**
- The three duplicated Go implementations were consolidated into a single
  `internal/simd/turboquant.go` holding the canonical reference
  (`polarQuantReference` / `qjlTransformReference`) and the public API. There
  is now exactly one definition, so the copies cannot drift again.
- The C kernels are wired in through new build-tagged dispatchers:
  `turboquant_simd_amd64.go` (AVX-512, then AVX2, selected by **runtime** CPU
  feature detection — the AVX-512 kernel is compiled with
  `#pragma GCC target("avx512f,...")` and would raise SIGILL on a CPU without
  it), `turboquant_simd_arm64.go` (NEON) and `turboquant_simd_generic.go`
  (no accelerator). `turboquant_avx2.c` also gained the missing
  `//go:build amd64 && cgo` tag.
- The reference implementation now matches the CUDA kernel exactly: it clamps
  the quantized value and returns the residual in original space.

**Tests added.**
- `internal/simd/turboquant_parity_test.go`: pins the clamp, pins the
  residual convention directly (recomputing `R^T * (rotated - q*scale)` in Go
  from the *accelerated* path's own outputs), and compares the accelerated
  path against the reference.
- `internal/device/turboquant_parity_test.go`: transcribes the CUDA
  `turboquant_encode_kernel` into Go and asserts the CPU encoder produces the
  same quantized bytes, the same QJL signs and the same scale. Mutation-tested:
  breaking the C kernel's residual convention makes it fail.
- `FuzzPolarQuant` now uses a mixed (column-normalised) rotation matrix and
  asserts parity with the reference, so the fuzzer can no longer miss a
  convention mismatch. It found the tie-breaking bug on its own.

**Measured on this host (12th Gen Intel Core i7-12650H, AVX2, no AVX-512):**

```
BenchmarkPolarQuantReference-16    85848 ns/op    3328 B/op    4 allocs/op
BenchmarkPolarQuantSIMD-16         11377 ns/op    1284 B/op    3 allocs/op
```

— a **7.5× speedup** with one fewer allocation, and the accelerated path is
now actually taken (`accelerated path available = true`). The fuzzer ran
3.6 M executions without a failure.

**Also fixed:** the no-cgo implementation used to index `rotationMatrix[j*n+i]`
unconditionally, so a missing rotation matrix would panic with an
index-out-of-range. The consolidated implementation handles it, and
`TestPolarQuantSIMD_EmptyRotationMatrixIsSafe` covers it.

**Finding.** `README.md:41` and `docs/performance.md:61-68,88-89`, `docs/architecture.md:129-131`, `docs/turboquant.md:69-71` claim AVX-512 / AVX2 / NEON vectorized TurboQuant with 16-lane FMA. In fact:

```
$ head -5 internal/simd/turboquant_avx.go
//go:build amd64 && cgo
package simd
import ("github.com/23skdu/longbow-quarrel/internal/mathx")   # no import "C"
```

`turboquant_avx.go`, `turboquant_avx2.go`, `turboquant_neon.go` are **pure Go** and literally say *"Fallback to generic when CGO functions unavailable"*. The C kernels do exist (`polar_quant_avx512`, `qjl_transform_avx512`, `polar_quant_avx2`, `qjl_transform_avx2` in `turboquant_avx512.c` / `turboquant_avx2.c` / `turboquant_neon.c`) but **no Go file references them**:

```
$ grep -rn "polar_quant\|qjl_transform" --include=*.go .
(no output)
```

Live measurement contradicts the docs: `BenchmarkPolarQuantSIMD` = 31,531 ns/op, 2,304 B/op, **3 allocs/op** — not 22.4 µs, not zero-alloc.

**Plan.** Either (a) wire the C kernels via `cgo` exports in `turboquant_avx.go`/`turboquant_neon.go` and re-run the benchmarks, or (b) delete the unreferenced `.c` files and correct all four documents. Option (a) is the better product outcome; option (b) is the honest minimum.

**Acceptance Criteria:** the SIMD speedup table in `docs/performance.md` is reproducible from a committed command, or is retracted.

---

#### Step 34: CI `lint` job is red on `main` — P0 — [DONE]

**Status: COMPLETE — `golangci-lint run --timeout=10m` reports 0 issues (was 40); `go run ./scripts/vacuous_tests` reports 0 new (was 2).**

How each class was resolved:

- **gofmt (17)** — `golangci-lint run --fix`.
- **staticcheck SA1019, arrow-go (13)** — migrated rather than suppressed. `internal/arrow_client/client.go` was pinned to the deprecated `flight.NewFlightClient` behind a `//nolint`; it now calls the non-deprecated `flight.NewClientWithMiddlewareCtx` (a behavioural match with a nil auth handler and no middleware). The test file was migrated from `arrow.Record` to `arrow.RecordBatch` throughout.
- **staticcheck SA9003, empty branches (2)** — `internal/gguf/gguf_coverage_test.go` had bare `if cond { /* comment only */ }`. Replaced with real assertions. **The first assertion I wrote was wrong**: I assumed tiny inputs flush to zero, but E4M3 clamps the exponent to −6 and encodes `0x08` (the smallest subnormal), while E5M2 clamps to −15 and encodes `0x00`. Both are now asserted to their real values.
- **errcheck (6)** — `LoadFrom`/`Close` calls in `streaming_llm_test.go` and `remote_resiliency_test.go` now checked and fatal on failure, rather than silently ignored.
- **ineffassign (3)** — two were **genuine coverage holes**, not lint noise: `client_coverage_test.go` assigned `res` from `DoPutTensor` and never read it. Rather than deleting the assignments, both are now asserted — including the `sendPutRes` branch, which is the actual DoPutTensor result path.

**Checker false positive fixed.** `TestWorkerPool_ConcurrentCalls` asserts *inside a goroutine*, but `scripts/vacuous_tests/main.go` treated every `*ast.FuncLit` as opaque (return `false` from `ast.Inspect`), so goroutine/deferred/callback assertions were invisible. The checker now descends into bare closures and only skips `t.Run` subtests. Verified with three probe tests: a truly vacuous test is still flagged, while goroutine and subtest assertions are now accepted.

`TestEngine_Types_FreeSafety` (7 `Free()` calls, zero assertions) was rewritten to allocate real tensors, assert they hold `rows*cols` elements beforehand, assert `Free()` empties them afterwards, and exercise the nested `MOELayerWeights` path plus double-`Free`. Mutation-tested: neutering `MOERouterWeights.Free` makes it fail with two distinct diagnostics, so it is a genuine test.

**Still open:** `scripts/vacuous_tests/main.go:94-97` only *warns* on stale allowlist entries rather than failing, and the OK message hardcodes `0` for the new-count.

---

#### Step 35: Feature stacks that are implemented, tested — and not reachable — P1 — [PARTIAL]

Verified by `deadcode -tags cuda -test ./internal/... ./cmd/quarrel` cross-checked with `grep`. Each cluster compiles, has passing tests, and has **zero** non-test callers.

| Cluster | Location | LOC | Status |
|---|---|---|---|
| Arrow Flight distributed inference | `internal/arrow_client/*`, `internal/engine/remote.go`, `distributed_master.go` | ~800 | PARTIAL — single-request `DoGet` works; the sharded forward is **not implemented on the worker** and used to fail silently (Batch 10) |
| Speculative decoding | `internal/engine/speculative.go` | 367 | **DONE (Batch 12)** — reachable via `-draft-model` and the API's `speculative` field |
| Radix prefix cache | `internal/engine/radix_cache.go` | 286 | **DEAD — duplicate of the wired `prompt_cache.go`** |
| Jinja2 chat templates | `internal/engine/prompt_wrapper.go` | 372 | TEST-ONLY — **no chat template is ever applied** |
| OTEL tracing | `internal/telemetry/telemetry.go` `InitTracer:29` | 8 fns | DEAD — tracer never initialised |
| CUDA Graph capture | `internal/device/cuda_graph.go` | 8 fns | TEST-ONLY |
| Multi-GPU TP/PP | `internal/device/multi_gpu.go` | 1038 | 34 of ~50 methods TEST-ONLY |
| Quantization tooling | `internal/gguf/quantize*.go` | 784 | TEST-ONLY |
| Activation logging / audits | `internal/engine/activation_logger.go`, `audit.go` | ~750 | TEST-ONLY |
| SIMD instrumentation | `internal/simd/metrics.go` `metricsWrapper` | 3 fns | DEAD — never instantiated |
| Grammar sampling constructors | `internal/sampler/grammar.go:56,70` | 2 fns | Only setter is the never-started API |

**Status: PARTIAL.** One correction to the original finding: the **prefix
cache is wired**, not dead. `continuous_batching.go:263` calls
`promptCache.MatchPrefix(req.Prompt)` and adopts the blocks via
`kvCache.AttachPrefixBlocks`. The `radix_cache.go` Patricia trie is a
*second* implementation of the same idea, and it is the one that is dead.

**A real correctness bug was found in the wired implementation.**
`PromptCache.MatchPrefix` returned the depth of the deepest matched node
together with *that node's* blocks — but the blocks are stored on the **last**
node of an inserted prompt, and the intermediate nodes hold none. A partial
prefix match therefore reported "N tokens matched" with an **empty** block
list, and the caller did:

```go
if matchedCount > 0 {
    kvCache.AttachPrefixBlocks(seqIDStr, cachedBlocks)   // attaches nothing
    seq.Pos = matchedCount                               // but advances the position
}
```

The sequence would then decode from position N with **no KV state** for
those N prompt tokens — i.e. the model would generate from a prefix whose KV
was never computed. `MatchPrefix` now only reports a match when the matched
node actually carries blocks, and returns that node's depth.

The existing test `TestPromptCache_Coverage` had encoded the buggy
expectation ("Expected partial match 3, got 0") and was updated.
`internal/engine/prompt_cache_test.go` covers the corrected semantics.

**Remaining limitation (not fixed):** because the blocks live only on the
last node, a *partial* prefix of a longer cached prompt is not reusable.
Storing the leading blocks on the intermediate nodes would fix that, but it
needs the KV block size and changes the refcount semantics, so it is left as
a follow-up.

**Batch 6 wired three more of these.**

**The Jinja2 chat templates now work.** `internal/engine/prompt_wrapper.go`
claimed to cover "Llama 3, Qwen, Gemma, and Mistral", but it did not: the
renderer rebuilt its variable context on every recursive call, so the body
of a `for` block rendered as the empty string and every real template
produced just `"<s>"`. Three separate defects had to be fixed:

- The variable context was rebuilt per recursion, discarding the loop
  variables. It is now threaded through.
- `message['role']` (bracket access) and `{{ 'a' + b }}` (concatenation)
  were not supported. Both are now, and the loop variable can be named
  anything (`msg`, `message`, `m`, ...), not just the two hardcoded names.
- `splitBlock` did not decrement its nesting depth for a *different* block
  terminator, so a `{% if %}` inside a `{% for %}` made the search for
  `endfor` run past the end of the loop body.

`internal/engine/chat_template_render_test.go` renders the real Llama 3, Qwen,
Gemma and Mistral templates and checks their markers. The API's
`/v1/chat/completions` handler now uses the model's own template (read from
the GGUF metadata key `tokenizer.chat_template`) instead of hardcoding the
ChatML format, which is the Qwen format and wrong for every other model.

**The OTEL tracer is now initialised.** `telemetry.InitTracer` was never
called by any binary, so the declared `go.opentelemetry.io/otel` dependency
produced nothing. All four binaries now call it. Its default branch also
set up a stdout exporter with `AlwaysSample`, which would have printed every
span of every request to stdout; the default is now a no-op tracer and the
console exporter is opt-in via `OTEL_TRACES_EXPORTER=console`.

**The SIMD kernel instrumentation now works.** `internal/simd/metrics.go`
was never instantiated, so all 14 SIMD histograms and counters exported a
permanent zero. The kernels are now instrumented, guarded by
`QUARREL_SIMD_METRICS=1`. It is **off by default** on purpose: instrumenting
a kernel costs a timestamp pair plus a histogram observation, and these
kernels run thousands of times per generated token.

**Batch 8 found a live bug in the paged KV cache.** Writing the first test
for `PagedKVCache.UpdateBatch` (the batched KV update, used by the Metal
engine) immediately panicked. The function was broken on **all three**
backends:

- **CPU**: the physical positions were read from `physicalPositions.rawData`,
  which is only populated for the quantized formats, so the CPU path always
  panicked with an index-out-of-range.
- **CUDA and Metal**: the store kernels take the positions as `const int*`,
  but the Go side allocated the tensor as **float32**. The kernels therefore
  read the float bits as integers and wrote to a garbage physical slot --
  silently corrupting the KV cache.

The positions are now carried as floats on every backend (the values are
small non-negative integers, exactly representable in float32), the CUDA and
Metal kernels cast them to `int` themselves, and the CPU reads them from the
tensor's float data. `internal/engine/kv_cache_batch_test.go` verifies that
the K and V values actually land in the physical slot the sequence was
assigned, on both the CPU and the CUDA build.

**Batch 10 found a silent-corruption bug in the Arrow Flight protocol.**

The Flight stack is more than "not reachable": the *server* side of the
sharded tensor forward was never implemented, and both ends of the protocol
conspired to hide that.

The worker (`internal/api/flight_server.go`) implements `DoGet` properly: it
tokenizes the prompt, runs `InferWithCallback`, and streams the generated
tokens back. That is the path the single-request inference uses, and it
works.

`DoPut` is the path the *batched / sharded* forward uses, and it did this:

```go
for {
    _, err := stream.Recv()
    if err == io.EOF {
        return stream.SendMsg(&flight.PutResult{})   // no AppMetadata
    }
    ...
}
```

It read the payload and **threw it away**, then answered with an empty
`PutResult`. Two more gaps made the failure invisible:

- The client never transmitted the operation metadata. `DoPutTensor` takes a
  `metadata map[string]string` and only ever **logged** it, so the server
  could not even tell what had been asked for. The descriptor path is the
  only thing that distinguishes the requests, and it does: the tensor
  forward uses `["compute", "layer"]` while the control messages use
  `["embeddings"]`.
- The client read the empty `PutResult` as *"the worker processed it but
  did not return data"* and returned an empty slice. Every one of the four
  `DoPutTensor` call sites then did `tensor.LoadFrom(resultData)` on that
  empty slice, producing a **zero-filled** hidden state. The master
  continued with zero activations and emitted garbage — with no error
  anywhere in the stack.

The fix turns the silent corruption into a diagnosable failure:

- The worker now inspects the descriptor on the first message of the
  `DoPut` stream and returns `codes.Unimplemented` for the tensor forward
  it cannot serve, while still accepting the control messages that share
  the handler.
- The client treats an empty result as an **error** rather than as success.
  Every caller needs the result tensor, so an empty response can only ever
  have been a failure.

`internal/api/flight_server_forward_test.go` covers both ends, and the two
existing tests that had encoded the old behaviour
(`TestDoPutTensor_RoundTrip/server_sends_no_result` and
`TestFlightClient_DoPutTensor_ServerInteractions`) were corrected.

**A second bug surfaced in the same area: a crash on the GPU builds.**

`FlightClient.StreamEmbeddings` derives the Arrow schema from the first
tensor's column count and calls `arrow.FixedSizeListOf`. Arrow **panics**
on a non-positive list size rather than returning an error. The backends
disagree about the shape of a zero-value tensor:

- the CPU reports one column (`Cols()` returns 1 when the tensor has no
  dimensions), so the panic never fired there;
- the CUDA and Metal backends report zero columns, so a zero-column tensor
  took down the process with `panic: arrow: invalid size`.

The client now validates the tensor dimensions before it builds the schema
and returns an error, and `device.Tensor.ToArrowArray` rejects a
zero-column tensor as well. The `default` branch of `ToArrowArray` (the
quantized-type fallback) also indexed `hostData[0]` without checking the
length, which was a second panic waiting to happen.

This is a good reminder of why the CI matrix runs the test suite under
every build tag: the CPU test suite was green while the CUDA build
crashed.

**What is still missing for the distributed stack:** the worker cannot serve
a *sharded layer* forward at all. There is no engine API that takes a
hidden state, runs it through a layer range, and returns the output hidden
state; `engine.ForwardBatch` takes token IDs and runs the whole model. A
real implementation needs that API, a protocol that carries the layer range
and the sequence state, and a worker that owns a slice of the layers. Until
that exists, the master mode (`MasterDistributedEngine`) has no working
peer to talk to, and the single-request `DoGet` path is the only part of
the distributed stack that functions end to end.

**Batch 12 wired the speculative decoding stack, and found two real bugs in
it.**

The stack was complete but unreachable: `SpeculativeManager`,
`AsymmetricSpeculativeEngine` and the acceptance test all existed, and
nothing in any shipped binary ever constructed them. The REST API even
accepted a `"speculative"` request field and a `"draft_k"` field, and then
ignored both — the handler always used the plain engine, so a request that
asked for speculative decoding silently got ordinary generation.

It is now reachable three ways:

- `quarrel -draft-model <path>` loads a second, smaller model and wraps the
  target engine with it. The draft model runs on the CPU, because a draft
  model is small by definition and the GPU memory is better spent on the
  target.
- `-draft-length` and `-draft-paths` set the draft length and the number of
  parallel draft paths.
- The REST API's `"speculative"` and `"draft_k"` fields now select the
  speculative engine and set its draft length.

**The acceptance test was wrong.** It compared the target's *normalized*
probability for a token with the target's *unnormalized* `exp(logit)` for
the same token:

```go
draftProb := mathx.Exp(float64(targetLogits[draftToken]))   // unnormalized
targetProb := probs[draftToken]                              // normalized
acceptanceRatio := targetProb / draftProb
```

With the logit scale real models use, the unnormalized value dwarfs the
normalized one — for a logit of 20 the ratio is about 2e-9 — so **virtually
every draft token was rejected** and the scheme could not have produced a
speedup. The draft model's probability for the token it proposed was
available all along: the manager called `InferWithLogits` and threw the
logits away.

The acceptance test is now the standard one, `min(1, p_target / p_draft)`,
with the rejection replaced by a draw from the residual distribution
`norm(max(0, p_target - p_draft))`. The manager keeps the draft's
distribution for every proposed token, and the `rejectSample` signature
gained the draft logits. When the draft and the target disagree on the
vocabulary size the two distributions cannot be compared, and the draft
token is accepted only when the target ranks it first.

**The correction was computed and then dropped.** When a draft token was
rejected, the residual sample was written into the candidate list, but only
the *accepted prefix* was appended to the sequence, so a rejection produced
no token at all. The correction is now emitted, and the sequence's position
accounts for it.

A third, smaller fix: a speculative step can produce more tokens than were
asked for, and the surplus was returned to the caller even though it had
never been emitted through the callback. The return is now trimmed to what
was emitted.

`internal/engine/speculative_wiring_test.go` covers all of it. The
regression test for the acceptance rule stores a value with a logit of 20
and checks that a token the target agrees with is accepted: with the old
rule it was rejected on the first trial.

**Still open:** the distributed/Arrow Flight stack, CUDA
Graphs, the multi-GPU TP/PP engine, the quantization tooling and the
activation/audit logging. Each needs to be either wired behind a documented
flag or removed.

**Batch 11 aligned the paged-KV pool's declared type with its stored type.**

The CUDA and Metal paged-KV pools were declared float32, but the store and
attention kernels treat them as packed half-precision arrays. The two
kernels agree with each other, so the cache *worked*, but two things were
wrong:

- the pool was allocated at **twice** the size it needs, and
- a host-side read of the pool had to decode the halves by hand.

For a 7B-class model (32 layers, 8 KV heads, 128 head dim, 4K context) that
is roughly 2 GB of VRAM allocated for a 1 GB cache — a direct contradiction
of the "run models that exceed your GPU's VRAM" claim.

The fix introduces `device.Context.NewPagedKVPool(capacity, kvDim)`, which
each backend implements with the element type its kernels actually use:
float32 on the CPU, packed half-precision values on CUDA and Metal. The
engine allocates the pool through it.

Everything that reads the pool on the host now works unchanged, because the
device layer converts at the boundary: `ToHostF32` decodes the halves and
`LoadFrom` encodes them, on both CUDA and Metal. The offload/restore
tiering, the defragmentation and the block copy all round-trip correctly.

Two related inconsistencies fell out of the change:

- `device.Tensor.StoreKV` on CUDA branched on the **source** tensor's type
  rather than the **cache's**, so it copied four bytes per element into a
  two-byte-per-element pool. It now follows the cache's type and converts
  the source when it has to.
- The GPU test helper that read the pool had to reach into the raw bytes
  and decode the halves by hand. The two backend-specific helpers collapsed
  into one, which is the proof that the declared type now matches the
  stored type.

`internal/engine/kv_cache_pool_type_test.go` guards the invariant: it stores
a value that half precision cannot represent exactly and checks that the
read-back comes back half-rounded on the GPU and exact on the CPU, and that
the pool occupies exactly the space its element type needs. Reverting the
pool to the old float32 declaration makes it fail with garbage values
(`K[0] = 2.003662, want 1`).

**Not fixed:** the TPU backend has no `NewTensorFP32`, so the TPU build of
`internal/engine` does not compile at all. It could not be checked here
either, because the build needs `libtpu`.

**Plan.** For each row, choose one: wire it behind a documented flag, or delete it. The volume here (~3,500 LOC) makes "delete" the better default for anything without a near-term plan — dead code in an inference engine is a correctness liability, not just clutter.

---

#### Step 36: Duplicate sampler in `engine_cpu.go` — dead parallel implementation — P1 — [DONE]

**Status: COMPLETE.** All four backends sample via
`sampler.Sample(logits, seq.Tokens)` → `internal/engine/sampler.go:84`,
which has full equivalents (`applyTopK`, `applyTopP`,
`applyTemperatureAndSoftmax`, `sampleFromCandidates`). But `engine_cpu.go`
carried a **second, unused** sampler:

`RMSNormCPU:514`, `Sigmoid:520`, `ApplyTempCPU:556`, `ApplyTopKCPU:564`,
`ApplyTopPCPU:595`, `softmaxCPU:641`, `SampleFromDistCPU:648` — no
production caller at all. Two samplers that can silently diverge is a real
correctness hazard, so the copies and the tests that existed only to cover
them were removed. The shared `Sampler` is now the single implementation.

**Still open (minor):** `Sampler.SampleAdvanced:31`,
`sampleWithQualityControl:41`, `calculateAdaptiveTemperature:273` and
`applySmartRepetitionPenalty:305` are still test-only.

---

#### Step 37: Config fields declared but never read — silent misconfiguration — P1 — [DONE]

**Status: COMPLETE, with a correction to the original finding.**

Re-checking the fields with their *full* names showed that the original
finding was **partly wrong**: the Gemma4 fields (`Gemma4SlidingWindowSize`,
`Gemma4SlidingRoPETheta`, `Gemma4FullRoPETheta`, `Gemma4PartialRoPEFactor`,
`Gemma4SlidingHeadDim`, `Gemma4FullHeadDim`, `Gemma4SlidingPattern`) **are**
read — 5 to 13 references each. The original audit grepped for
`\.SlidingHeadDim` rather than `\.Gemma4SlidingHeadDim` and so missed every
one of them. That false finding is corrected here.

`Config.StopTokens`, however, was genuinely never read anywhere in the tree:
a user who configured stop tokens got them silently ignored, and generation
only stopped at an EOS token or the token limit.

**Fix.** `internal/engine/stop_tokens.go` adds a `stopTokenSet` /
`isStopToken` pair, and all four engines (CPU, CUDA, Metal, TPU) now build
the set once at construction and consult it in the decode loop. A stop token
terminates generation exactly like an EOS token.

**Also fixed (Step 38).** `Tokenizer.AddEOSTokenID` was dead as well. The
engines now register `config.EOSTokenID` on the tokenizer at construction,
and the tokenizer always treats token ID 2 as EOS — which is what every
engine had been hardcoding (`token == 2 || token == cfg.EOSTokenID`) in three
separate places. The decode loops now consult a single source of truth, and
`Tokenizer.GetEOSTokenIDs` reports the real set.

**Tests.** `internal/engine/stop_tokens_test.go` covers the set helper and
guards against the field regressing to a declared-but-ignored one.
`internal/tokenizer/eos_convention_test.go` pins the "ID 2 is always EOS"
convention.

**Still open (minor):** `SamplerConfig.SequenceID`
(`internal/engine/sampler_config.go:15`),
`metalEngine.DraftEngine` / `.qualityEval` (`types_metal.go:25,29`),
`TensorParallelManager.bwdPasses` (`multi_gpu.go:354`) and
`ErrorResponse.Detail` (`internal/api/server.go:224`) are still unused.

---

#### Step 38: `Tokenizer.GetEOSTokenIDs` is never called — stop/EOS handling is incomplete — P1 — [DONE]

**Status: COMPLETE.** See the "Also fixed (Step 38)" note in Step 37 above:
`AddEOSTokenID` is now called by every engine to register the model's
configured EOS token, the tokenizer always reports token ID 2 as EOS, and the
decode loops consult only the tokenizer. `GetEOSTokenIDs` now returns the
real set (and is exercised by the new convention test).

The multi-EOS case is therefore covered: a model that declares several EOS
ids (Llama 3 `<|eot_id|>`, Qwen `<|im_end|>`) has all of them in the
tokenizer's set, and the engine stops on any of them.

---

#### Step 39: `MockEngine` is the last-resort production fallback and returns fake tokens — P1 — [DONE]

**Status: COMPLETE.** `internal/engine/engine_mock.go` registered itself in the
production registry, and `NewRegisteredEngine` listed `"mock"` last in the
fallback chain. Because `MockEngine.Infer` returns `count` copies of
**token 42** with a nil error, an engine-construction failure on a host
without CUDA or Metal produced confident, plausible-looking garbage instead
of an error.

The mock is no longer registered, and `"mock"` is no longer in the fallback
chain. `MockEngine` and `NewMockEngine` remain exported for tests (several
test files construct `&engine.MockEngine{}` directly), but a shipped binary
can no longer select them.

`internal/engine/mock_registration_test.go` guards both halves of this: the
mock is not in the registry, and `NewRegisteredEngine` returns an error when
every real backend fails.

---

#### Step 40: Headline architectures have 0% coverage — Qwen 3.5 SSM and Gemma 4 — P1

`internal/engine` sits at 51.1% overall, but the *architecture-specific* CPU paths that the README leads with are entirely untested:

```
$ go tool cover -func=cov.out | awk '$3+0 < 40'
cpu_weights.go:1151  mambaForward            0.0%   <- Qwen 3.5 SSM dispatcher
cpu_weights.go:1161  mamba2Forward           0.0%
cpu_weights.go:1351  gatedDeltaNetForward    0.0%   <- Qwen 3.5 GatedDeltaNet
cpu_weights.go:1770  ComputeGemma4PLE        0.0%
cpu_weights.go:1876  ApplyGemma4LayerCPU     0.0%   <- Gemma 4 layer, live path
cpu_weights.go:1866  isGemma4SlidingLayer    0.0%
lora.go:39           LoadAdapter             0.0%
kv_cache_paged.go:616  ForkSequence          0.0%
kv_cache_paged.go:873  UpdateBatch           0.0%
```

All of these **are** on the live forward path (`cpu_weights.go:674` dispatches to `mambaForward`; `engine_cpu.go:359,482` and `engine_cuda.go:601` call `ApplyGemma4LayerCPU`). The zero-coverage functions are the same ones docs/nextsteps.md lists as headline achievements.

**Status: COMPLETE for the two headline architectures.**
`internal/engine` coverage went from **51.0% to 62.8%**, and the
architecture-specific functions from 0% to:

| Function | Before | After |
|---|---|---|
| `mambaForward` (Qwen 3.5 SSM dispatcher) | 0.0% | **100.0%** |
| `gatedDeltaNetForward` (Qwen 3.5 GatedDeltaNet) | 0.0% | **90.9%** |
| `mamba2Forward` | 0.0% | **45.0%** |
| `geluCPU` | 0.0% | **100.0%** |
| `ComputeGemma4PLE` | 0.0% | **91.2%** |
| `NewGemma4LayerBuf` | 0.0% | **100.0%** |
| `isGemma4SlidingLayer` | 0.0% | **100.0%** |
| `ApplyGemma4LayerCPU` | 0.0% | **75.4%** |

**Tests added.**
- `internal/engine/ssm_cpu_test.go`: drives `gatedDeltaNetForward` and
  `mamba2Forward` with synthetic weights, checks the output is finite and
  non-empty, checks the **dispatcher** picks the right implementation, checks
  the layer is **stateful** (two identical calls with a shared KV cache must
  differ, because the recurrent state has to be carried over), and covers the
  degenerate inputs.
- `internal/engine/gemma4_cpu_test.go`: drives `ApplyGemma4LayerCPU` on both
  a normal layer and one **beyond the shared-KV boundary** (which must reuse
  an earlier layer's KV), plus the buffer allocator, the sliding-layer
  pattern helper and the PLE computation.

**Two findings worth recording, because they shaped the tests.**

1. The synthetic weights have to match the *shapes the code derives at
   runtime*. `ApplyGemma4LayerCPU` re-derives `headDim` from the Q
   projection length (`headDim = len(q) / heads`) and from the K projection
   length (`len(k) / kvHeads`), so the Q weight must be `[heads*headDim, dim]`
   and not `[dim*headDim, dim]`. Getting this wrong makes the layer panic
   with a slice-bounds error rather than fail an assertion.
2. The GatedDeltaNet output is multiplied by `SiLU(z)`, where `z` is the
   attention-gate projection. Without a populated `AttnGate` the gate is zero
   and the layer output is **identically zero** — so a test that only checks
   "the output is finite" would pass on a completely inert layer. The tests
   therefore populate the gate and assert on behaviour, not just finiteness.

**Still open:** `lora.go:41` `LoadAdapter`, `kv_cache_paged.go:616`
`ForkSequence`, `:873` `UpdateBatch`, `:1067` `GetBlockScales` and `:1235`
`OffloadLRU` are still at 0%, as are the `remote.go` / `distributed_master.go`
/ `speculative.go` stacks (Step 35).

---

#### Step 41: Broken scripts — benchmark and profile tooling cannot run — P1 — [DONE]

| Script | Defect |
|---|---|
| `scripts/benchmark_compare.sh` | **Always exits 2.** `MODEL` is checked at `:13` *before* argv is parsed at `:19` — so `ci.yml:374` (`./benchmark_compare.sh -m ... -p ... -n 32`) fails. This gates `release-candidate`. Also `:63` runs `go run -tags=darwin,metal ./cmd/quarrel/main.go` but `main.go` is `linux&&cuda`; `LLAMA_BENCH` defaults to a Homebrew path absent on Linux runners. |
| `scripts/profile_tokens.sh` | `bin/quarrel-cuda` / `bin/quarrel-simple` (`:53-54`) are never produced — the Makefile emits `bin/quarrel-linux-amd64-cuda`. Same wrong paths in `benchmark_engines.sh:42-43` and `benchmark_models.sh:140-141`. Its CSV is synthesised from one average (`:117-122`), so Mean/Median/P95/P99 are all identical; `TOKEN_NUM` at `:79-87` is computed then discarded. |
| `scripts/benchmark_engines.sh` | `:172-174` `err`s and `return`s when no binary is found → **script exits 0 with no data**. The `:194-196` fallback computes `N_TOKENS/elapsed_ms`, so a binary that crashes on startup yields an enormous bogus tok/s recorded as real. |
| `Makefile:193` `test-simd-fuzz` | `-fuzz` swallows `-fuzztime=10s` as its value → `warning: no fuzz tests to fuzz` → PASS. Fuzzes nothing. |
| `Makefile:197` `bench-simd` | `-bench=AVX512` with no `-tags avx512` → **0 benchmarks matched**. |
| `Makefile:11` `.PHONY` | lists `install`, which does not exist; and omits ~25 real targets. |
| `Dockerfile.nvidia:91` | `COPY --from=cuda-builder /app/internal/device/...` but the builder stage's `WORKDIR` is `/src` (`:44`) and the archive is written at `/src/internal/device/libcuda_kernels.a` (`:55`). **Build fails.** Blocks `ci.yml:609` and `ghcr-publish`. |
| `scripts/gen_gguf/main.go` | checked-in `package main` that writes `test.gguf` into the CWD on any run; referenced by nothing. |

**Status: COMPLETE.**
- `scripts/benchmark_compare.sh` now parses argv **before** validating `MODEL`, so the documented `-m/--model` invocation works (it always exited 2 before). It also builds the `./cmd/quarrel` *package* rather than `./cmd/quarrel/main.go`, which is gated `linux && cuda` and could not build on macOS, and it parses the real output format (`N tokens in X.XXs (Y.Y t/s)`) instead of the "Inference complete" / "Decoded Text:" strings the binaries never print.
- `scripts/quarrel_bin.sh` (new) resolves a runnable binary for `profile_tokens.sh`, `benchmark_engines.sh` and `benchmark_models.sh`. They previously looked for `bin/quarrel-cuda` and `bin/quarrel-simple`, which the Makefile has never produced (it emits `bin/quarrel-linux-amd64-cuda`), so on a clean checkout they always failed with "binary not found". The resolver prefers the platform-native build, falls back to the CUDA build, and finally builds one on demand.
- `Dockerfile.nvidia:91` now copies from `/src/internal/device/libcuda_kernels.a` (the builder stage's `WORKDIR`), not `/app/...` — the build previously failed at that `COPY`.
- `Dockerfile.cpu` / `Dockerfile.nvidia` stamped `-X main.version=...` / `-X main.commit=...`, which target variables that do not exist, so the linker silently dropped them. They now stamp `main.Version` / `main.BuildTime` via `ARG QUARREL_VERSION` / `ARG QUARREL_BUILD_TIME`.
- `Makefile:193` `test-simd-fuzz` passed `-fuzz -fuzztime=10s`, which makes `-fuzz` swallow the duration as its value and fuzz nothing; `Makefile:197` `bench-simd` matched `-bench=AVX512` with no `-tags avx512`, matching zero benchmarks.

---

#### Step 42: `Dockerfile.webui` env vars and API key are silently ignored — P1 — [DONE]

**Status: COMPLETE.** `cmd/webui/config` now has an `ApplyEnv` overlay that reads `WEBUI_PORT`, `WEBUI_METRICS_PORT`, `WEBUI_HOST`, `WEBUI_API_KEY`, `WEBUI_ALLOWED_ORIGINS` and `WEBUI_MODEL_PATH`. `cmd/webui/main.go` applies the environment first and then overrides with any flag that was explicitly provided (`flag.Visit`), so flags still win. `config_test.go` covers both directions, including that an unset variable never clobbers an explicitly set value.

`handlers.NewAuthMiddleware(cfg.APIKey)` now receives a real key when `WEBUI_API_KEY` is set, and the server logs a warning when it is not, so an unauthenticated deployment is at least visible. `SplitOrigins` now always returns a non-nil slice so it marshals to `[]` rather than `null`.

Also fixed: `docker-compose.cuda.yml:81` passed `--interactive`, which no `cmd/` defines — that service died on startup. It now uses `--stream`.

---

#### Step 43: Grafana dashboard queries a Summary as if it were a Histogram — P2 — [DONE]

**Status: COMPLETE.** `inference_duration_seconds` is now a `Histogram` rather than a `Summary`. A Summary without explicit `Objectives` exports **no** quantile series at all, so the three P50/P95/P99 panels could never have returned a value regardless of the query syntax; the `_sum` and `_count` series keep their names, so existing consumers are unaffected. `docs/metrics.md` was updated to match.

`internal/metrics/dashboard_test.go` now parses the dashboard and the metric declarations and asserts that **every** `histogram_quantile(...)` expression refers to a metric declared as a Histogram. The check is static rather than registry-based on purpose: a `HistogramVec` only appears in the default registry once a labelled observation has happened, so a registry snapshot reports false negatives for every vec metric that has not been used yet. Reverting the metric to a Summary makes the test fail with a precise message.

Secondary (still open): the 50 KB dashboard is never provisioned — neither compose file mounts it, and `cmd/webui/grafana/provisioning/` contains only `datasources.yml`.

---

#### Step 44: Documentation claims that contradict the code — P1 — [PARTIAL]

**Status: PARTIALLY COMPLETE.** The claims in `README.md` that are now
false have been corrected:

| Claim | Correction |
|---|---|
| `go.mod` declares `go 1.27.0` | Now says `1.27.2` (the actual value) |
| "600,000+ continuous fuzz test executions" | Replaced with the gates that actually exist (`vacuous_tests`, `dupcheck`) |
| "Templ-based web UI" | Now says "stdlib `html/template`" |
| "the `DoPutTensor` round trip is currently non-functional" | Replaced with the test that verifies it |
| resolver searches `~/.ollama/models/` | Corrected: the search dirs are `~/.cache/{llmfit/models,llama.cpp,huggingface/hub,quarrel/models}` + `~/.local/share/nomic.ai/GPT4All`; Ollama **tags** resolve through the Ollama manifest store |
| coverage table + "packages below 80% are pinned to a ratchet floor" | Regenerated from `go test -cover ./internal/...`, with a note that the numbers cover the default build only |
| `internal/mathx`, `internal/promtest` missing from the table | Added |
| `docker-compose.cuda.yml --interactive` | Now `--stream` (the flag that exists) |

**Batch 6 continued this work.**

- `docs/metrics.md` documented **ten metrics that do not exist** and omitted
  **73 that do**. Both are fixed, and `internal/metrics/docs_test.go` now
  checks the reference against the source in both directions: a metric cannot
  be documented without existing, and (outside the Go runtime and process
  collectors) a metric cannot exist without being documented. The check
  parses the source with `go/ast` so that the `Namespace` and `Subsystem`
  options are folded into the metric name the way the Prometheus client does.
- `docs/features.md` named `FuzzPagedKVCache` and `FuzzSampler`, which do
  not exist. The real targets are `FuzzPagedKVCache_TurboQuant` and
  `FuzzSamplerQualityControl`.
- `docs/features.md` claimed "114 verified `#nosec` directives"; the actual
  count is 146.

**Batch 7 finished the remaining documents.**

- `docs/performance.md`: the dequantization table named benchmarks that do
  not exist (`BenchmarkDequantizeQ4K_SIMD_*`). The table was regenerated
  from the real benchmarks (`BenchmarkDequantizeQ4K_*`) with the exact
  command to reproduce it, and the AVX-512 section now states that its
  numbers require AVX-512 hardware, because the `-tags avx512` build
  selects the kernels at *runtime* and falls back to scalar otherwise.
- `docs/comparison.md`: "144 Prometheus metrics" → 191, the number the code
  actually declares.
- `docs/dependencies.md`: every version was stale. The table was
  regenerated from `go.mod`/`go.sum` (toolchain 1.27.2, prometheus 1.25.0,
  otel 1.47.0, x/image 0.47.0, grpc 1.86.0-dev, the emlgo pseudo-version
  and the Go image tag).
- `docs/roadmap.md`: it documents the **vector search** subsystem, not the
  inference engine, and was linked from the README's index as if it were the
  product roadmap. It now says so at the top, and its "8 gates" and
  "internal/vector is 91.1%" claims were corrected (10 gates, 100.0%).
- `docs/testplan.md`: it documented an 11th gate that does not exist. The
  gate list now matches `scripts/run_all_tests.sh` (10 gates), the CUDA
  graph test command names the tests that exist, and the Go version is
  corrected.
- `docs/webui_plan.md`: 473 lines of unchecked boxes proposing a stack the
  code does not use (`a-h/templ`, `gin-gonic/gin`, `markbates/grift` — none
  of them are dependencies). It was rewritten to describe the implementation
  that exists, with the remaining ideas listed as not implemented.
- `docs/usage.md`: the flag is `-kv-cache-size`, not `-kv-cache`; the
  `QUARREL_SHARD_ROLE` / `QUARREL_WORKER_ADDRS` variables do not exist and
  are now marked as not available.
- `docs/api.md`: the API key variable is `WEBUI_API_KEY`, not
  `QUARREL_API_KEY` — the latter never existed, so the key was never
  applied.

**Still open:** the performance numbers in `docs/performance.md` that need a
GPU to reproduce (the layer-offload table and the emlgo A/B table, which is
honestly marked as not yet measured).

---

| Doc | Claim | Reality |
|---|---|---|
| `README.md:66`, `comparison.md:74,103`, `api.md:7,57` | OpenAI-compatible `/v1/chat/completions` | **FIXED** (Step 29): the API is now served on `-api-addr`. The WebUI serves it at `/api/v1/chat/completions`; the engine serves it at `/v1/chat/completions`. |
| `README.md:41`, `performance.md:61-68`, `turboquant.md:69-71`, `architecture.md:129-131` | AVX-512/AVX2/NEON TurboQuant | **FIXED** (Step 33): the C kernels are now dispatched at runtime and the CPU encoder matches the CUDA kernel. The performance table was rewritten with the measured numbers. |
| `README.md:9,58`, `comparison.md:90`, `usage.md:62` | CPU SIMD AVX-512 | Never compiled (Step 32). **Build FIXED**; the doc text is **STILL OPEN.** |
| `README.md:43`, `comparison.md:119` | "600,000+ fuzz executions" | Unsubstantiated. The only fuzz job (`ci.yml:235-287`) runs ~6.5 min/target on `macos-latest`, push-to-main only. `turboquant.md:87` says a *different* number (218,000+). **README FIXED**; `comparison.md` **STILL OPEN.** |
| `README.md:44` | `DoPutTensor` non-functional | **FIXED** (Step 44): `nextsteps.md:485` marks it `[RESOLVED]` and `TestDoPutTensor_RoundTrip` passes. |
| `README.md:17` | `go.mod` declares `go 1.27.0` | It declares `1.27.2`. **FIXED.** |
| `README.md:39` | resolver searches `~/.ollama/models/` | **FIXED**, and the original claim was partly wrong: `internal/ollama` *is* called, as a fallback at `internal/models/resolver.go:101`, so Ollama tags do resolve. The *search-directory* list did not include `~/.ollama/models`; the README now describes both mechanisms accurately. |
| `README.md:178` | "Templ-based web UI" | Uses stdlib `html/template`; `github.com/a-h/templ` is not in `cmd/webui/go.mod`. `cmd/webui/templates/base.templ` is orphaned. **README FIXED**; the orphaned `.templ` file is **STILL OPEN.** |
| `README.md:109-111` | `go test -bench=AVX512` / `-bench=NEON` | Match **0** benchmarks on linux/amd64; pass vacuously. **STILL OPEN.** |
| `README.md:126` | `cd cmd/webui && go test ./...` | Runs **zero** tests (Step 31). **FIXED**: the command now needs `-tags webui`, and CI passes it. |
| `README.md:145-167` | Coverage table + "packages below 80% are pinned to a ratchet floor" | **FIXED** (Step 45). |
| `features.md:100` | `FuzzPagedKVCache`, `FuzzSampler` | Don't exist. Real: `FuzzKVCacheUpdate`, `FuzzPagedKVCache_TurboQuant`, `FuzzSamplerQualityControl`. |
| `features.md:104` | "114 `#nosec` directives" | Actual count is 146. |
| `testplan.md:173,182` | "Gate 11: CUDA Precision & Graph Gate" | `scripts/run_all_tests.sh` has **10** steps. |
| `testplan.md:149` | `TestCUDA_GraphManager` | Doesn't exist; real names are `TestCUDA_GraphManager_Lifecycle` / `_ErrorHandling`. |
| `testplan.md:28` | "Go 1.24+" | `go.mod` says 1.27.2. |
| `metrics.md` | 10 documented metric names (`kv_cache_misses_total`, `lora_dispatch_groups_per_batch`, `sampling_nan_handling_total`, `sampling_seed_reproducible_total`, `dequant_pass_total`, `dequant_fail_total`, `weight_aligned_total`, `weight_not_aligned_total`, `buffer_alignment_total`, `buffer_invalid_total`) | None are defined anywhere in `internal/` or `cmd/`. |
| `dependencies.md:19,25,28-31,106-107,172` | Every pinned version | All stale vs `go.mod` (grpc `v1.85.0-dev`→`v1.86.0-dev`, otel `1.46.0`→`1.47.0`, prometheus `1.24.1`→`1.25.0`, emlgo pseudo-version, Go `1.27.1`→`1.27.2`). |
| `comparison.md:108,160,170` | "144 Prometheus metrics" | 131 `Name:` entries in `internal/metrics/metrics.go`; 176 unique names repo-wide. Matches neither. |
| `comparison.md:37` | table cell containing `�Q` | Mojibake where a ✅ should be. |
| `roadmap.md` (whole file) | "Longbow-Quarrel Engineering Roadmap" | Documents the **vector-search** subsystem (`internal/vector`, `cmd/vector_benchmark`), not the inference engine, yet is linked from the README doc index. Also claims `internal/vector` is 91.1% (actual 100.0%) and "8 gates" (actual 10). |
| `webui_plan.md` (473 lines) | implementation plan | **100% unchecked boxes** though the feature is built; lists deps (`templ`, `gin`, `grift`) that aren't in `go.mod`; documents routes and test files that don't exist; says `go run -tags webui ./cmd/webui/` from the repo root, which cannot work (separate module). |
| `usage.md:105,125` | `-kv-cache 4096` | Flag is `-kv-cache-size` (`main.go:34`); the documented invocation aborts. |
| `usage.md:28,226`, `architecture.md:19`, `api.md:34` | `QUARREL_SHARD_ROLE`, `QUARREL_WORKER_ADDRS`, `~/.ollama/models/`, `/healthz` on :8080 | None exist. |
| `performance.md:37-42` | `BenchmarkDequantizeQ4K_SIMD_*` | Real names are `BenchmarkDequantizeQ4K_*`. No row in the table is reproducible. |
| `performance.md:74-81` | GPU offload tok/s + VRAM figures | No methodology, artifact, or commit. |
| `nextsteps.md:307` | `.vscode/settings.json` has `-tags=cgo,avx512,cuda` | Committed file says `-tags=cgo,cuda`. |

**Acceptance Criteria:** every row above is either corrected or the code is changed to match. Add a docs-lint CI step that greps for referenced-but-missing Go symbols/flags/test names — most of these are mechanically detectable.

---

#### Step 45: Coverage floors and the README table are stale — P1 — [DONE]

`go test -cover ./internal/...` (default build) vs what the docs and gates claim:

| Package | README | CI / `run_all_tests.sh` floor | **Measured** |
|---|---|---|---|
| `internal/cpu` | 100.0% | — | 100.0% |
| `internal/logger` | 100.0% | — | 100.0% |
| `internal/metrics` | 98.7% | ≥80 | **100.0%** |
| `internal/telemetry` | 98.1% | — | **100.0%** |
| `internal/ollama` | 91.1% | 87.0 | **100.0%** |
| `internal/models` | 92.5% | — | **100.0%** |
| `internal/tokenizer` | 92.2% | ≥80 | **100.0%** |
| `internal/vector` | 91.2% | ≥80 | **100.0%** |
| `internal/vlm` | 85.5% | ≥80 | **100.0%** |
| `internal/simd` | 85.5% | ≥80 | **97.0%** |
| `internal/gguf` | 83.4% | ≥80 | **100.0%** |
| `internal/api` | 82.9% | ≥80 | **91.7%** |
| `internal/sampler` | 80.3% | ≥80 | **100.0%** |
| `internal/arrow_client` | 78.8% | 75.0 | **88.3%** |
| `internal/config` | 78.0% | 75.0 | **100.0%** |
| `internal/device` | 73.5% | 70.0 | **75.7%** |
| `internal/engine` | 50.1% | 47.0 | **51.1%** |

Only `engine` and `device` are still below 80%, so `README.md:145`'s framing ("packages below 80% are pinned to a ratchet floor") no longer describes reality. The `config:75.0` and `arrow_client:75.0` ratchet entries are dead weight. `internal/mathx` and `internal/promtest` (both 100%) are missing from the table entirely. `internal/simd` and `internal/vlm` are both listed as exactly `85.5%` — a copy-paste artifact.

**Important caveat that makes the numbers misleading regardless:** coverage here counts only the **default no-tag build**. `internal/device` has 88 `.go` files but only 9 compile untagged (1,010 statements counted); `internal/engine` has 88 files, 20 compile (4,295 statements). All CUDA/Metal/TPU/AVX-512 code is excluded from both numerator and denominator, so "device 75.7%" says nothing about the GPU backends.

**Status: COMPLETE.**
- The README table was regenerated from the measured values, and now includes `internal/mathx` and `internal/promtest`. It also carries the default-build-only caveat.
- The ratchet lists in `.github/workflows/ci.yml` and `scripts/run_all_tests.sh` were reduced to the two packages that are actually below 80%, and their comment now records the measurement. The stale `config:75.0`, `arrow_client:75.0` and `ollama:87.0` entries are gone — keeping a floor for a package that is already held by the ≥80% gate only hides regressions.
- The floors are now raised to `engine: 58.0` and `device: 72.0` (the measured values are 62.8% and 76.2%).

**Batch 8 also closed the last of the paged-KV coverage gaps**: `ForkSequence`,
`UpdateBatch`, `GetBlockScales`, `OffloadLRU` and `Size` now have tests, and
the first of them found a live bug in `UpdateBatch` (see Step 35).

**Batch 9** added tests for the LoRA merge and the adapter registry, the
chat-template loader, `NewEngine` (the public entry point, previously at 0%),
`MatVec` (a hot path, previously at 22%) and the prompt-cache LRU eviction.
Two more live bugs surfaced:

- `PromptCache.Evict` cleared an entry's blocks *before* calling
  `removeLRU`, which computes the `lruSize` decrement from those blocks. The
  decrement was therefore always zero, so `lruSize` stayed permanently too
  high and the cache believed it was over its limit and evicted everything.
- `PromptCache.moveToHead` added an entry's block count to `lruSize` again
  when an already-cached prompt was re-inserted, double-counting it.

Both are fixed and covered by regression tests. The `internal/engine` package
is now at 68.1%.

**Batch 10** fixed the Arrow Flight protocol's silent corruption and a crash
that only the GPU builds hit (see Step 35). **Batch 11** aligned the paged-KV
pool's declared type with its stored type, which halves the KV cache's VRAM
footprint on CUDA and Metal (see Step 35). **Batch 12** wired the speculative
decoding stack, which turned out to have an acceptance rule that rejected
almost every draft token (see Step 35).

**Still open:** raising `internal/engine` above 80%, and generating the README table from a script so it cannot drift apart from the floors again.

---

#### Step 46: Coverage gates re-run the suite and can pass vacuously — P2 — [DONE]

**Finding.**
- `ci.yml` re-executed `go test -cover` for 8 + 5 = 13 packages *after* the
  full `go test -v -coverprofile=coverage.out ./...` — and that profile was
  then never used for the thresholds.
- `grep -o 'coverage: [0-9.]*%' | awk '{print $2}'` silently took only the
  **first** match when a pattern expanded to multiple packages.
- `|| true` on the grep pipeline meant a **failed test run** also produced an
  empty string, producing a confusing "produced no coverage result" instead
  of surfacing the real failure.
- `cmd/webui` is a separate module and never appears in `coverage.out` →
  Codecov permanently under-reports.

**Status: COMPLETE.** `scripts/coverage_gate.sh` replaces both gates. It
reads the coverage profile that CI has **already** produced and derives the
per-package percentages from it directly, by summing the statement and
covered counts per package:

```
mode: atomic
<file>:<start>.<end> <numStatements> <count>
```

A count of 0 means the block was not covered. This removes the two extra
full test-suite runs, removes the ambiguous `grep`, and cannot turn a failed
test run into an empty string. The same script is used by
`scripts/run_all_tests.sh`, so the local and CI gates cannot disagree.

`scripts/run_all_tests.sh` step 9 was repurposed to the duplicate-declaration
check, and the ratchet step was folded into step 8.

**Still open:** `cmd/webui` is still a separate module and does not appear in
`coverage.out`, so Codecov under-reports the WebUI.

---

#### Step 47: Build-tag and CI coverage blind spots — P2 — [DONE]

**Status: COMPLETE.** The following were fixed:

- **AVX-512 is now compiled and tested in CI.** A new `avx512` job builds and
  vets the whole tree with `-tags avx512`, and then asserts with `go tool nm`
  that the VNNI kernels are actually **linked into the binary** — a build that
  silently dropped the tag would be caught. The test run is `continue-on-error`
  because the intrinsics SIGILL on a runner without AVX-512.
- **The race job now covers every internal package.** It previously gated 5 of
  19; a second step now runs the remaining 14. The claim in `README.md:43` and
  `features.md:105` ("0 data races across all packages") is now actually
  enforced.
- **The fuzz corpus upload path was corrected** from `testdata/fuzz/` (repo
  root, which does not exist) to `internal/engine/testdata/fuzz/` (where the
  corpus is actually written).
- **The `notify` job now depends on every other job** — `docker`,
  `compose-lint`, `ghcr-publish`, `cuda-compile`, `release-candidate`,
  `build-artifacts` and `fuzz-tests` were missing, so a red Docker build did
  not turn `notify` red.
- **actionlint is now run in CI.** `.github/actionlint.yaml` had existed but
  was never invoked. The tool was validated by injecting a bogus `needs:`
  entry and confirming it was reported.
- **The bogus Dependabot npm entry was removed.** There is no `package.json`
  anywhere in the repository, so the entry only produced errors.
- **The Dependabot auto-merge workflow was hardened.** It now waits for the CI
  checks to pass, uses `--match-head-sha` so the merge applies to exactly the
  commit that was reviewed, and **never auto-merges `grpc` or `eml`** — the
  gRPC pin is deliberately held at the commit that fixes GO-2026-6443
  (CVE-2026-84445), and no tagged release contains the fix yet.
- **The benchmark gate no longer skips silently.** When there is no base
  commit the job now emits an explicit `::warning::` and a "SKIPPED" line
  instead of quietly reporting success.

**A real defect was found and fixed by the new gates:** `go vet -tags cuda
./...` failed with

```
internal/engine/engine_coverage_boost_test.go:131:17: invalid argument:
tensor.Data() (value of type unsafe.Pointer) for built-in len
```

`device.Tensor.Data()` returns `[]float32` in the CPU and Metal builds but an
`unsafe.Pointer` in the CUDA build, so the test only compiled for the CPU
backend. The test now uses `Rows()*Cols()` plus a build-tagged
`releasedElementCount` helper, and the CUDA gate is green again.

**Batch 7 closed these.**
- `gosec` now also runs on the `-tags avx512` build. The default build only
  covers the code that compiles without a build tag, which is a fraction of
  the tree.
- `FuzzPagedKVCache_TurboQuant` had **no seed corpus at all**, so the CI
  fuzz job started from nothing on every run. It now has seeds, which are
  also executed by the ordinary test run and therefore act as regression
  cases. The seed sizes are deliberately modest: the cache allocates on the
  order of `heads*headDim*layers*windowSize`, and the first attempt at a
  large seed needed 578 MB of device memory and failed on a smaller GPU.
- **The Playwright browser e2e test had never run.** It skips itself when
  the driver is missing, and the driver could not be downloaded: the
  community fork of `playwright-go` still points at the retired
  `playwright.azureedge.net` CDN, so every download 404s. The import path
  is now the maintained upstream, `github.com/mxschmitt/playwright-go`
  v0.6201.1, which downloads from the current CDN, and the CI `build` job
  installs the driver. The browser test now executes.

**Still open:**
- `govulncheck` still runs on the default build only; the tagged builds
  need a different module graph and are left for a job with the right
  toolchain available.

---

#### Step 48: Repository hygiene — committed build artifacts and caches — P2 — [PARTIAL]

`git ls-files` shows these tracked despite being regenerable and already in `.gitignore`:

- `engine.test` (29 MB binary), `quarrel` (2.4 MB binary)
- `*.cov.out` — `cov.out`, `dev_cov.out`, `eng_cov.out`, `api_cov.out`, `cfg_cov.out`, `cpu_cov.out`, `met_cov.out`, `mod_cov.out`, `sam_cov.out`, `simd_cov.out`, `tok_cov.out`
- `coherence_verification_results.json` (29 KB), `coherence_results_2026*`/
- `internal/gguf/fuzz_test.go.bak`
- `scripts/__pycache__/`

There is also **no `.dockerignore`**, so `COPY . .` in every Dockerfile ships `.git`, `bin/` (138 MB of stale binaries), `engine.test`, and the coverage dumps into the build context. Note `.git` *is* needed by `Dockerfile.cpu:36` (`git rev-parse --short HEAD`), so add `.dockerignore` rather than relying on `.gitignore`.

**Plan.** `git rm --cached` the artifacts, confirm `.gitignore` covers them, add `.dockerignore` (retaining `.git` or moving the version stamp to a build arg).

**Status: PARTIALLY COMPLETE.** A `.dockerignore` was added, excluding `.git`, `bin/`, `dist/`, `*.test`, `*.cov.out`, `performance_profiles/`, `coherence_results_*/`, `__pycache__/` and the editor/OS noise. The version stamp no longer needs `.git`: `Dockerfile.cpu` and `Dockerfile.nvidia` now take `ARG QUARREL_VERSION` / `ARG QUARREL_BUILD_TIME`.

**Status: COMPLETE, with a correction to the original finding.** The
artifacts (`engine.test`, `quarrel`, the `*.cov.out` files,
`coherence_verification_results.json`, `performance_profiles/`, `bin/`) were
**never tracked** — they are all covered by `.gitignore` and were only present
in the working directory. The original finding was based on `ls`, not
`git ls-files`.

The one genuinely tracked artifact, `internal/gguf/fuzz_test.go.bak`, has been
removed. A `.dockerignore` was added so that `COPY . .` no longer ships
`.git`, `bin/`, the coverage dumps or the test binaries into the build
context.

---

#### Step 49: The macOS Metal build was broken — P0 — [DONE]

**Finding.** Two independent defects, both invisible to a Linux CI run.

**1. Duplicate declaration.** `internal/engine/engine.go:25` and
`internal/engine/sampler.go:13` both declared `ErrModelSwapped`. In a
`darwin + metal` build both files are compiled, so the package failed to
compile:

```
ErrModelSwapped redeclared in this block
	other declaration of ErrModelSwapped
```

**2. A wrong build tag.** `internal/device/cpu.go` carried:

```go
//go:build (!cuda && !metal && !tpu) || !amd64 || !cgo || (!linux && !darwin)
```

`!amd64` is **true on arm64**, so on Apple Silicon the CPU backend was
compiled *alongside* `metal.go` — and both declare `Context`, `Tensor`,
`NewContext` and `AllocatedBytes`. The original tag was simply
`!cuda && !metal`; commit `6d712dd` ("resolve backend redeclaration
conflicts") changed it to this form and reintroduced the conflict in a
different shape. The same family of problem affected
`internal/device/multi_gpu_metal.go` (a plain Go file tagged `darwin &&
metal` that calls `Context.AllReduce`, which only exists in the cgo file
`metal.go`) and the metal-tagged files in `internal/engine`.

**Fix.**
- `ErrModelSwapped` is now declared only in `sampler.go`, which compiles in every build.
- `internal/device/cpu.go` is now `(!cuda && !metal && !tpu) || !cgo || (!linux && !darwin)`.
- The metal-only files that depend on cgo-only symbols (`internal/device/multi_gpu_metal.go`, `internal/engine/{engine,types_metal,mamba,layer_mamba,layer_moe,audit,kv_cache_sliding_window}.go`) now carry `&& cgo`. The macOS (`cgo=1`) file selection is unchanged; only the broken `cgo=0` configuration is affected.

**New gate: `scripts/dupcheck.sh`.** A duplicate top-level declaration only
breaks the build when two files that declare the same name land in the same
package — and with build-tag-selected files that can hide in a configuration
nobody builds locally. `scripts/dupcheck` (a new Go program) uses
`go list -json` to obtain **exactly the files the compiler would select** for
a given configuration, then reports any package-level name declared twice.
`go list` needs no C compiler, so a darwin or cuda configuration can be
checked from a Linux host.

`scripts/dupcheck.sh` sweeps every package across seven configurations
(linux/amd64 with and without cgo, linux/arm64, darwin/arm64 and
darwin/amd64 with `metal darwin`, linux/amd64 with `cuda`, and
linux/amd64 with `avx512`) — **210 combinations, currently all clean**. It
runs in the CI `lint` job, so a duplicate declaration can no longer reach a
release.

The tool was itself validated by re-introducing the `ErrModelSwapped`
duplicate and confirming that it reports it, and by confirming that
reverting `inference_duration_seconds` to a Summary makes the Step 43
dashboard test fail.

---

### Recommended Execution Order

| Order | Steps | Rationale | Status |
|---|---|---|---|
| 1 | **34** (lint green), **28** (working CPU binary), **31** (webui CI) | Nothing else can be trusted until CI reflects reality and a CPU build runs. | **DONE** |
| 2 | **29** (serve the API), **42** (env vars + API key) | Makes the product actually usable; fixes an auth no-op. | **DONE** |
| 3 | **32**, **33** (AVX-512 + TurboQuant wiring) | Delivers the performance the README already claims. | **DONE** (both) |
| 4 | **41** (scripts), **44**, **45**, **46**, **47** | Restore trust in the numbers and the docs. | **41, 45, 46, 47 DONE**; 44 partial |
| 5 | **35**, **36**, **37**, **38**, **39** (dead code + config fields) | Reduces surface area before the next feature lands. | **36, 37, 38, 39 DONE**; 35 partial |
| 6 | **40** (0% coverage on headline architectures) | Protects Qwen 3.5 / Gemma 4 correctness. | **DONE** for the two headline architectures |
| 7 | **30**, **43**, **48** | Cleanup. | **30, 43 DONE**; 48 partial |
| 8 | **49** (macOS Metal build) | The darwin+metal build was broken by a duplicate declaration and a wrong build tag. | **DONE** |

**Also done beyond the original list:** Step 49, the macOS Metal build fix
and the `dupcheck` gate that prevents a recurrence.

**Suggested Definition of Done for this phase:** `main` is green on every CI job; `make cpu` produces a binary that runs real inference; `curl :8080/v1/chat/completions` returns tokens; `docker run longbow-quarrel:cpu --help` exits 0; every README/docs claim is reproducible from a committed command.

> **Progress against the definition of done:** `golangci-lint`, `vacuous_tests`
> and the new `dupcheck` gate are green; `make cpu` runs real inference;
> `curl :8080/healthz` and `/v1/chat/completions` are served and covered by
> tests; `quarrel --help` exits 0. Still outstanding: the Dockerfiles are
> fixed but not exercised (no Docker in this environment), the TurboQuant
> kernels are unwired, `internal/engine` coverage is ~51%, and several
> documented performance numbers remain unreproducible.

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

### `DoPutTensor` cannot complete a round trip (P0) — [RESOLVED]

`internal/arrow_client/client.go` documents `DoPutTensor` as "sends a tensor to
worker and returns the result tensor data".

**Resolved:**
- `DoPutTensor` now decodes returned tensor `float32` payloads directly from `PutResult.AppMetadata` when present, and returns cleanly with `[]float32{}` on empty ack-less completion.
- Verified in `internal/arrow_client/doput_tensor_test.go::TestDoPutTensor_RoundTrip` against an in-process Flight server with bit-exact float32 parity.

---

> For detailed performance benchmarks, pprof profiling data, and vector engine P0 blockers, see [docs/roadmap.md](roadmap.md).

#### Last updated: October 2026 (latest tag `0.3.0`; post-0.3.0 work landed on `main`, unreleased; Phase 6 ship-it audit added)
