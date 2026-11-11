# Longbow-Quarrel Roadmap

> **Scope.** This document is the single record of the engineering work
> that is **not finished**. It merges the engine roadmap, the vector-search
> roadmap and the WebUI plan. What has already landed is summarised in
> [What has landed](#what-has-landed); the detail is in the git history.
> The architecture is in [architecture.md](architecture.md).

---

## Summary

The engine runs real inference on CPU, CUDA and Metal. The consolidated
test suite, the accuracy and oracle gates, the security, lint and
profiling gates, and the pre-commit hook are all green. What remains is
the work below: recommendations derived from memory profiling, the vector
performance benchmarks, and the feature gaps against llama.cpp, vLLM and
Ollama.

**Definition of done for v1.0.0:** every P0 and P1 item closed; the CI
pipeline green on every job; every number in
[performance.md](performance.md) reproducible from a committed command.

---

## The gates

The repository keeps itself honest with local gates and the CI jobs that
run them. `scripts/run_all_tests.sh` runs the local ones; `make ci` runs
the adapted quality pipeline.

| # | Gate | What it catches |
|---|------|-----------------|
| 1 | Core package unit tests | The default build |
| 2 | SIMD & VNNI tests | The AVX2 / AVX-512 / NEON kernels |
| 3 | VLM pipeline tests | The vision path |
| 4 | Grammar PDA & sampler tests | The constrained sampler |
| 5 | Speculative decoding & paged cache | The draft/verify loop |
| 6 | CUDA compile, vet and kernel tests | The GPU build and its kernels |
| 7 | Race detector | The concurrency-critical packages |
| 8 | Coverage gates | `internal/engine` ≥ 80 %, `internal/device` ≥ 75 % |
| 9 | Duplicate declaration check | A name declared twice in one configuration |
| 9b | Metal type check | The Metal backend compiles, tests included |
| 9c | ARM64 / NEON build and tests | The NEON kernels compile and agree with the reference |
| 10 | Zero-allocation hotpath | An allocation in a hot kernel |
| 11 | Workflow lint | A broken workflow file |
| 12 | Documentation drift | The docs, the chart and the dashboard against the code |
| 13 | Dead code | A function that no code path reaches |
| 14 | Vacuous tests | A test that cannot fail |
| 15 | Accuracy gate | Numerical precision across the vector dtypes |
| 16 | Oracle gate | Ground-truth parity vs the scalar reference oracles |
| 17 | Security gate | gosec AST analysis and govulncheck |
| 18 | Fuzz coverage gate | Every fuzz-critical package has a Fuzz target |
| 19 | Lint gate | golangci-lint across all packages |
| 20 | Git pre-commit hook | `scripts/verify.sh` installed by `make makehooks` |

`make security`, `make perf-check`, `make mem-profile`, `make lint`,
`make accuracy-gate`, `make oracle-gate` and `make fuzz-coverage` are the
entry points for the security, profiling, linting and accuracy/oracle
checks. `make verify` runs `scripts/verify.sh`; `make makehooks` installs it
as `.git/hooks/pre-commit`.

---

## Recommendations

Recommendations are grouped by the evidence that produced them. Each is
measurable against the existing gates.

### A. Memory profiling

The vector benchmark audited the heap: baseline 1.86 MB, final 3.48 MB,
a net delta of 1.62 MB across 50k–250k-vector workloads, with no leaks.
The mmap store keeps the Go heap flat, and `TestMMapLeakAudit` enforces a
5 MB delta over 16 load/search/close cycles. The remaining memory
exposure is in the decode path, where activation buffers are still
allocated per call and weight pages are not pinned.

| # | Recommendation | Evidence | Acceptance |
|---|----------------|----------|------------|
| A1 | **Pin weight pages and double-buffer activations** so the decode loop no longer page-faults | mem profile of `cpu.out` / `mem.out` | fewer page faults in decode; no growth in `HeapAlloc` |
| A2 | **Add a per-engine memory budget gate** that fails when `HeapAlloc` grows more than 5 MB over a fixed workload | `TestMMapLeakAudit` pattern | gate passes on CPU and CUDA |
| A3 | **Pool the KV cache scratch buffers** instead of allocating per token | mem-escape output | zero new allocations in the decode hot path |
| A4 | **Bound the working set of the mmap store** with eviction hints so a 1M-vector index fits 256 MB | P0-05 measurement | 1,000,000 vectors inside a 256 MB working set |

### B. Performance testing

The vector benchmark covers 50k, 100k and 250k vectors at 128 and 384
dimensions across `float32`, `turboquant`, `complex128` and `uint8`, for
flat, IVF and HNSW topologies. HNSW `float32` reaches 25,062 QPS at
128 dimensions but `complex128` is roughly 4× slower, and the
`turboquant` IVF path is the slowest combination. Cosine is consistently
slower than dot because it was not using the dispatched kernel.

| # | Recommendation | Evidence | Acceptance |
|---|----------------|----------|------------|
| B1 | **FP16 IVF store** so the scan reads half the bytes and the 250k×384 query lands under 500 µs | 1.10 ms at 384 dimensions today | 250k IVF query < 500 µs |
| B2 | **Adaptive nprobe** chosen from query entropy instead of a fixed value | probe count dominates latency | fewer probes at the same recall |
| B3 | **Complex128 on the GPU float32 path** instead of the 4×-bandwidth CPU path | 25 QPS single, 172 QPS batched | above 500 QPS |
| B4 | **GPU k-means transfer overlap** so the 384 MB upload stops dominating the build | 150 ms of the 317 ms GPU build | IVF build under 50 ms |
| B5 | **Direct W4A16 / W8A16 GEMV** with no dequantization temporary | Q4_K path dequantizes first | decode latency down, zero temporary allocations |
| B6 | **Dispatched cosine** (done) so cosine scoring uses the native kernels | cosine was scalar-only | cosine parity within 1e-4 of the oracle |
| B7 | **Speculative tree attention** (Medusa / EAGLE) for a higher acceptance rate | single linear chain today | acceptance rate above the chain baseline |
| B8 | **Unified batched-GEMM prefill/decode** scheduler | opposite shapes today | higher tokens/s under mixed load |
| B9 | **Sliding-window attention with sinks** for constant VRAM at 100,000 tokens | deleted block-table policy | constant VRAM at 100,000 tokens |

### C. Feature gaps with other inference engines

`docs/features.md` lists what Quarrel does not yet do compared with
llama.cpp, vLLM and Ollama: no AMD ROCm, no GPTQ/AWQ, no Vulkan, no
embedded WebUI, and no multi-target build conveniences. The remaining
planned features (AMX, multi-LoRA, persistent grouped GEMM, early exit,
clean-room dispatch) are engine-side gaps.

| # | Recommendation | Gap | Acceptance |
|---|----------------|-----|------------|
| C1 | **AMD ROCm backend** | llama.cpp and vLLM support AMD GPUs | ROCm build and tests green |
| C2 | **GPTQ and AWQ quantization** | popular formats not supported | load and run a GPTQ/AWQ model |
| C3 | **Vulkan backend** | llama.cpp supports Intel/AMD GPUs | Vulkan build and tests green |
| C4 | **Embedded WebUI** | WebUI is a separate binary today | UI served from the main binary |
| C5 | **Multi-target build conveniences** | manual build tags today | documented one-command matrix build |
| C6 | **Intel AMX tile kernels** | AVX-512 only today | measurable speedup on AMX hosts |
| C7 | **Multi-LoRA batch fusion** | one adapter merged at load time | batched multi-adapter inference |
| C8 | **Persistent grouped GEMM for MoE** | expert weights streamed per token | lower per-token latency on MoE |
| C9 | **Self-speculative early exit** | no early exit today | speedup with bounded accuracy loss |
| C10 | **Compile-time backend dispatch** | selection spread across the tree | backend matrix checked by the compiler |

### D. Testing and quality

| # | Recommendation | Acceptance |
|---|----------------|------------|
| D1 | Extend fuzzing to the mmap store, the IVF/HNSW indexes and the engine hot paths | new `Fuzz*` targets in those packages |
| D2 | Add an oracle test for every dispatched kernel against its scalar reference | parity within the documented tolerance |
| D3 | Keep `make accuracy-gate`, `make oracle-gate`, `make fuzz-coverage` in CI | all three green on every PR |

---

## Planned features

The items above are the planned work. Items A1, B1–B5, C1–C3 and D1 are
the highest priority.

## What has landed

Everything below is on `main` and unreleased — the latest git tag is
`0.3.0`. The CPU build is a real binary; the OpenAI-compatible REST API
and streaming are live; the WebUI has a working CPU adapter and browser
tests. The SIMD, CUDA, Metal and model-format paths are repaired and
gated. Speculative decoding, the multi-device flags, the audits and the
activation logger are real. The vector search has hand-written AVX2 and
AVX-512 kernels, split complex planes, asymmetric TurboQuant, a memory
mapped store and GPU batch kernels. 133 Prometheus metrics, a Grafana
dashboard and a drift checker are in place. Every field of `config.Config`
is read or gone, about 4,750 lines of dead code are deleted, the TPU
backend is gone, and the Metal backend compiles and type-checks. Every
test can fail, and the CI pipeline is enabled again. The quality gates,
the pre-commit hook and the profiling targets described above are on
`main`.