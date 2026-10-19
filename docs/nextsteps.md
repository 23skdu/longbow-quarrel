# Longbow-Quarrel - Next Steps & Roadmap

## Executive Summary

Longbow-Quarrel is a high-performance, multi-backend inference engine in Go, CGo, CUDA, and Apple Metal. The latest git tag is **`0.3.0`**; the milestones named v0.4.0 through v0.8.0 in commit messages all landed on `main` without a release tag. This document tracks the remaining work toward **v1.0.0**, including distributed execution (Phase 2), math backends (Phase 3), serving/DX (Phase 4), and deep kernel/engine optimizations (Phase 5: Steps 8–17).

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

#### Part 10 (Step 10): CUDA Graph Capture & Replay for Sub-Millisecond Decode Latency (P0) — [IMPLEMENTED]
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

#### Part 12 (Step 12): Direct Zero-Dequant Integer Dot-Product Kernels via AVX2 & AVX-512 VNNI (P0) — [IMPLEMENTED]
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

#### Part 16 (Step 16): Radix-Tree Prefix Caching & Copy-on-Write Paged KV Block Sharing
- **Objective**: Deliver instantaneous Time-to-First-Token (TTFT) and zero duplicate VRAM usage for shared system prompts, few-shot templates, and multi-turn conversations.
- **Bottleneck Analysis**:
  - The existing `PromptCache` (`internal/engine/prompt_cache.go`) uses a flat slice lookup for exact prompt prefix matching.
  - When multiple users share common system prompts or when a conversation branches into multi-turn dialogue, `PagedKVCache` allocates duplicate physical KV blocks for identical token sequences, consuming VRAM and repeating prompt prefill computations.
- **Design & Implementation**:
  - Implement a **Radix Tree** index in `internal/engine/radix_cache.go` where each edge stores a contiguous token sequence and references an immutable physical KV block in `PagedKVCache`.
  - Add Copy-on-Write (CoW) page reference tracking in `PagedKVCache`:
    - When a new sequence attaches to a cached prefix node, increment `blockRefs[blockIdx]` without copying data.
    - During token generation, when a sequence writes to a shared block (`blockRefs[blockIdx] > 1`), allocate a new physical block, copy the existing tokens, decrement the shared reference, and write to the private block (CoW).
  - Integrate LRU eviction of unreferenced leaf nodes in the Radix Tree when the `MemoryGovernor` exceeds the memory pressure threshold (0.85).
- **Files**: `internal/engine/radix_cache.go`, `internal/engine/prompt_cache.go`, `internal/engine/kv_cache_paged.go`, `internal/engine/continuous_batching.go`
- **Acceptance Criteria**: Multi-turn chat TTFT drops from $>250$ms to $<2$ms (instantaneous prefix hit); memory consumption for 100 concurrent requests sharing a system prompt drops by $\ge 80\%$.

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
