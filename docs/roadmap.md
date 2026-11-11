# Longbow-Quarrel Roadmap

> **Scope.** This document is the single record of the engineering work that
> is **not finished**. It merges the engine roadmap, the vector-search
> roadmap and the WebUI plan. What has already landed is summarised in
> [What has landed](#what-has-landed); the detail is in the git history.

## Summary

The engine runs real inference on CPU, CUDA and Metal. Every defect listed
by the previous roadmap is closed. What remains is the work below: twenty
performance and accuracy improvements, a wider unit and fuzz testing scope,
and the accuracy, oracle and security gates that keep them honest.

**Definition of done for v1.0.0:** every P0 and P1 item closed; the CI
pipeline green on every job; every number in
[performance.md](performance.md) reproducible from a committed command.

## The gates

The repository keeps itself honest with local gates and the CI jobs that run
them. `scripts/run_all_tests.sh` runs the local ones; `make ci` runs the
adapted Longbow quality pipeline:

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

## Planned features

The items below were specified and never built, or are the twenty
improvements listed in [Open work](#open-work). Each is a design, not a
defect.

## Open work

### The vector search

`internal/vector` is a standalone component. The five blockers of the
2026-09 audit have their implementations in the tree; their performance
targets are only partly met. The measurements and the gap for each are in
[performance.md, section 10](performance.md#10-vector-search).

### Twenty performance and accuracy improvements

Status note (this pass): items 17 and 18 are implemented in
`internal/vector/distance.go`; items 19 and 20 gained fuzz/parity coverage in
`internal/vector/fuzz_distance_oracle_test.go` and
`internal/vector/oracle_improvements_test.go`. Items 1-16 remain open
design/performance work.

Each item below is a concrete, measurable improvement. Items 1-6 are the
vector-search blockers; 7-16 are the planned engine features; 17-20 are
accuracy gaps found by the oracle and fuzz work.

| # | Improvement | Kind | Target / acceptance |
|---|-------------|------|---------------------|
| 1 | **FP16 IVF store** — store inverted lists in half precision so the scan reads half the bytes and the 250k×384 query lands under 500 µs | Performance | `L2Float32` scan < 15 % of CPU; 250k IVF query < 500 µs |
| 2 | **Adaptive nprobe** — choose the number of probes from query entropy instead of a fixed value | Performance | fewer probes at the same recall on the standard workloads |
| 3 | **Complex128 on the GPU float32 path** — run the complex scan on the device instead of the 4x-bandwidth CPU path | Performance | 250k `complex128` flat search above 500 QPS |
| 4 | **GPU k-means transfer** — overlap and chunk the dataset upload so the 384 MB transfer stops dominating the build | Performance | IVF build on 250k×384 under 50 ms |
| 5 | **Working-set bounded mmap store** — page hints and eviction so a million-vector index fits a 256 MB working set | Memory | 1,000,000 vectors inside a 256 MB working set |
| 6 | **TurboQuant HNSW tuning** — keep the 7.8× compression while pushing past 16,156 QPS | Performance | HNSW search above 20,000 QPS at 50,000 nodes |
| 7 | **Direct W4A16 / W8A16 GEMV** — integer dot product over quantized weights with no dequantization temporary | Performance | decode latency down, no temporary allocation |
| 8 | **Speculative tree attention** — Medusa / EAGLE-style tree verification for a higher acceptance rate | Accuracy / throughput | acceptance rate above the linear-chain baseline |
| 9 | **Unified batched-GEMM prefill/decode** — one scheduler that picks the kernel shape per iteration | Throughput | higher tokens/s under mixed prefill+decode |
| 10 | **Sliding-window attention with sinks** — circular physical blocks so VRAM stays constant at 100,000 tokens | Memory | constant VRAM at 100,000 tokens |
| 11 | **Pinned zero-copy memory** — pin weight pages and double-buffer activations to remove page faults from decode | Performance | fewer page faults in the decode loop |
| 12 | **Intel AMX tile kernels** — a dedicated tile layout instead of an AVX-512 flag | Performance | measurable speedup on AMX-capable hosts |
| 13 | **Multi-LoRA batch fusion** — gather low-rank factors per sequence so several adapters run in one pass | Throughput | batched multi-adapter inference |
| 14 | **Persistent grouped GEMM for MoE** — keep expert weights resident instead of streaming them per token | Performance | lower per-token latency on MoE models |
| 15 | **Self-speculative early exit** — run the cheap layers, check confidence, stop early when it is high enough | Latency | measurable speedup with bounded accuracy loss |
| 16 | **Compile-time backend dispatch** — one interface, one implementation per backend, checked by the compiler | Maintainability | backend matrix checkable by the compiler |
| 17 | **Cosine dispatch consistency** — `CosineFloat32` should use the dispatched dot kernel, not only the scalar path | Accuracy / performance | cosine parity within 1e-4 of the oracle |
| 18 | **L2Uint8 scaling** — revisit the averaged scale so uint8 distances match the float reference | Accuracy | L2Uint8 within 1e-3 of the float32 oracle |
| 19 | **TurboQuant QJL residual** — validate the residual correction against the float32 oracle across random inputs | Accuracy | DotTurboQuant within the documented tolerance |
| 20 | **Complex cosine normalization** — normalize the Hermitian cosine so complex vectors agree with the reference | Accuracy | CosineComplex128 within 1e-9 of the naive reference |

### Unit and fuzz testing scope

The unit and fuzz scope now covers the scalar oracles and the dispatched
kernels, the mmap store, the IVF and HNSW indexes, the engine hot paths,
the API request parsing, the sampler grammars and the SIMD kernels. New
fuzz targets added in this pass:

- `FuzzDotFloat32Oracle`, `FuzzL2Float32Oracle`, `FuzzCosineFloat32Oracle`
  in `internal/vector`
- `FuzzChatCompletionRequestParsing` in `internal/api`
- `FuzzRegexGrammar` in `internal/sampler`

Every fuzz-critical package is required to contain at least one `Fuzz`
target; `make fuzz-coverage` (backed by `scripts/check_fuzz_coverage.sh`)
fails the build otherwise. `make test-fuzz` runs the seed corpus of every
target.

### Accuracy and oracle code quality gates

- `make accuracy-gate` runs the numerical precision tests across the vector
  dtypes and the SIMD kernels.
- `make oracle-gate` runs the parity tests against the scalar reference
  oracles in `internal/vector`, `internal/simd`, `internal/mathx` and
  `internal/gguf`.
- `make fuzz-coverage` verifies that every fuzz-critical package has a
  Fuzz target.
- `scripts/verify.sh` is the git pre-commit hook: it runs `gofmt`,
  `go vet`, `golangci-lint`, `gosec` and the short unit and fuzz suites.
  `make makehooks` installs it into `.git/hooks/pre-commit`.

### Makefile targets

Security: `make security` (alias `sec`), `make sec-gosec`, `make sec-vulncheck`,
`make sec-audit`.

Performance and memory profiling: `make perf-check`, `make cpu-profile`,
`make profile-check`, `make mem-check`, `make mem-bench`, `make mem-escape`,
`make mem-profile`, `make mem-asan`, `make bench-ci`.

Linting: `make lint`, `make lint-fast`, `make lint-fix`, `make vet`,
`make staticcheck`, `make deadcode`, `make fmt`, `make fmt-check`,
`make markdown-lint`.

Quality gates: `make accuracy-gate`, `make oracle-gate`, `make fuzz-coverage`,
`make test-fuzz`, `make verify`, `make makehooks`.

## What has landed

Everything below is on `main` and unreleased — the latest git tag is
`0.3.0`. The work that closed the previous findings is in the git history;
the gates above are the mechanism that keeps it honest from here on.