# Longbow-Quarrel Roadmap

> **Scope.** This document is the single record of the engineering work
> that is **not finished**. It merges what used to be three documents:
> the engine roadmap (`nextsteps.md`), the vector-search roadmap
> (`roadmap.md`) and the WebUI plan (`webui_plan.md`).
>
> What has already landed is summarised in [What has landed](#what-has-landed)
> at the end; the detail of that work is in the git history, not here.
> The architecture is in [architecture.md](architecture.md).

---

## Summary

The engine runs real inference on CPU, CUDA and Metal. **Every defect
this document listed is closed.** The findings were:

| Priority | Count | The findings |
|----------|------:|--------------|
| **P0** | 4 | Steps 50, 51, 59, 60 — the CUDA engine dropped the caller's configuration; 55 of the 133 metrics were never written; the Metal backend did not compile, and once it did, its layer forward had seven numerical defects |
| **P1** | 9 | Steps 52, 53, 55, 57, 58, 61, 62, 65, 66 — write-only configuration fields; the ignored `stream` field; ~4,750 lines of dead code; the Flight sharded forward; the TPU backend; four features that were implemented and never enabled |
| **P2** | 4 | Steps 54, 56, 63, 64 — the commands the CI never builds; the CUDA graph manager; the 26 tests that could not fail; the skips that hid defects |

The work that closed them is in [What has landed](#what-has-landed).

What is **not** finished is what the document always said it was not
finished with: the [planned features](#planned-features) — the designs
that were specified and never built — and the [vector-search
blockers](#the-vector-search), whose implementations are in the tree but
whose performance targets are not all met.

**Definition of done for v1.0.0:** every P0 and P1 item closed; the CI
pipeline green on every job; every number in [performance.md](performance.md)
reproducible from a committed command.

---

## The gates

The repository keeps itself honest with local gates and the CI jobs that
run them. `scripts/run_all_tests.sh` runs the local ones:

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
| 9b | Metal type check | The Metal backend compiles, **tests included** |
| 9c | ARM64 / NEON build and tests | The NEON kernels compile and agree with the reference |
| 10 | Zero-allocation hotpath | An allocation in a hot kernel |
| 11 | Workflow lint | A broken workflow file |
| 12 | Documentation drift | The docs, the chart and the dashboard against the code |
| 13 | Dead code (9d) | A function that no code path reaches |
| 14 | Vacuous tests (9e) | A test that cannot fail |

The CI jobs that run the same checks: `lint`, `build`, `webui`,
`webui-browser`, `metal-build`, `metal-tests`, `fuzz-tests`,
`bench-gate`, `benchmarks`, `coverage`, `security`, `avx512`,
`cuda-compile`, `docker`, `compose-lint`, `ghcr-publish`,
`build-artifacts`, `release-candidate` and `alternate-builds`.

Two more run only in CI: `gosec` and `govulncheck`, each on the default
and the `avx512` build.

`scripts/docsync` is the drift checker behind gate 12. It verifies that
every flag the Helm chart passes is a flag the command has, that every
metric the Grafana dashboard queries is declared and queried in the right
shape, that every flag the reference documentation mentions exists and
every flag the commands declare is in it, that the API reference and the
server agree on the routes, that the pinned versions in
[dependencies.md](dependencies.md) are the ones the module files hold,
that every document is in the index, that every link between the
documents resolves, and that the layout in the architecture document names
every package.

`scripts/deadcode.sh` is the gate behind gate 13: the reachability
analysis over the whole module, with the tests as roots, fails when a new
unreachable function appears. `scripts/vacuous_tests` is the gate behind
gate 14: a test whose body has no assertion fails the build.

---

## Open work

### The vector search

`internal/vector` is a standalone component: the inference path does not
use it. The five blockers of the 2026-09 audit have their
**implementations** in the tree, and their **performance targets** are
only partly met. The measurements, the command that reproduces them and
the gap for each blocker are in
[performance.md, section 10](performance.md#10-vector-search).

| Blocker | The implementation | The target | Measured |
|---------|--------------------|-----------|----------|
| **P0-01** | AVX2 and AVX-512 distance kernels (`internal/vector/distance_avx2.c`, `distance_avx512.c`), the CUDA batch kernel for the inverted-list scoring (`cuda_vector_kernels.cu`) and the reordered store, where the rows of a list are next to each other | `L2Float32` below 15 % of the CPU; 250k IVF query under 500 µs | the kernel is 6 % of the search; the query is 222 µs at 128 dimensions and 1.10 ms at 384 |
| **P0-02** | the complex type is stored as split real and imaginary planes (`complex_soa.go`) and the CUDA batch path uses them | 250k `complex128` flat search above 500 QPS | 25 QPS single, 172 QPS batched; the target needs 750 GB/s and is not reachable for a 1.5 GB scan |
| **P0-03** | the asymmetric TurboQuant distance with a lookup table (`turboquant_adc.go`) and the query matrices built once per index instead of once per query | TurboQuant HNSW above 10,000 QPS, keeping the 7.8× compression | **16,156 QPS** at 50,000 nodes; the compression is kept |
| **P0-04** | the mini-batch k-means on the device (`cuda_vector_kernels.cu`), the multi-query L2 kernel for the assignment and the lane-per-centroid assignment kernel | IVF construction on 250k×384 under 50 ms | 618 ms on the CPU, 416 ms with the GPU (the transfer of the 384 MB dataset is 159 ms of it) |
| **P0-05** | the memory-mapped store with pre-faulting and page-eviction hints (`mmapstore.go`) | a million-vector index inside a 256 MB working set | the store is memory mapped and the Go heap stays flat: 100,000 × 384 vectors load into a 150 MB file and the heap does not grow |

The gap that dominates the rest is the memory bandwidth of the scan. At
250k × 384 with sixteen probes the scan reads twelve megabytes per
query, and at 1.10 ms that is the rate a sequential read of that size
gets out of this host's memory: the query is at the floor of what a
float32 store can do. Two things would take it under the target — a
store in half precision, or fewer probes. The second gap is the
bandwidth of the complex type, which is four times that of float32 on
the CPU — the float32 path the GPU already has is the answer.

The verification matrix for these: the consolidated suite keeps all its
gates green, `internal/vector` stays at or above 80 % coverage (it is at
84.8 %), the leak gate stays under 5 MB of delta, and the CPU and CUDA
implementations stay within 10⁻⁴ of each other.

---

## Planned features

These were specified and never built. Each is a design, not a defect.

### Speculative tree attention (Medusa / EAGLE-style)

The current speculative decoding proposes a single linear chain of draft
tokens. A tree — several candidates per position, verified in one pass —
would raise the acceptance rate per draft token. It needs a tree-shaped
attention mask in the verification pass and a multi-head draft model.

### Unified batched-GEMM prefill and decode co-scheduling

Prefill and decode have opposite shapes: prefill is a large GEMM over many
tokens, decode is a memory-bound GEMV over one. Batching them into one
kernel needs a fused scheduler that picks the shape per iteration.

### Build-tag clean-room architecture and compile-time backend dispatch

Every backend is selected by build tags, and the selection is spread
across the tree. A single dispatch layer — one interface, one
implementation per backend, and a compile-time assertion that exactly one
is present — would make the backend matrix checkable by the compiler
instead of by a script.

### W4A16 and W8A16 direct quantized GEMV without intermediate materialization

The Q4_K path dequantizes into a temporary before the dot product on some
backends. A direct integer dot product over the quantized weights removes
the temporary and the bandwidth it costs.

### Pinned zero-copy memory and asynchronous double-buffered batch pipeline

The weights are memory-mapped, but the activation buffers are allocated
per call. Pinning the weight pages and double-buffering the activations
would remove the page faults from the decode loop.

### Intel AMX tile acceleration

The AVX-512 and VNNI kernels are implemented. The AMX tiles are a
different instruction set with a different data layout, and they would
need their own kernels rather than a flag.

### Multi-LoRA batch fusion and grouped low-rank scatter-gather dispatch

The LoRA path merges an adapter into the CPU weights at load time. Batching
several adapters in one pass needs the low-rank factors gathered per
sequence rather than per model.

### Persistent grouped GEMM for mixture-of-experts

The MoE router and the expert dispatch exist. A persistent grouped GEMM
would keep the expert weights resident instead of streaming them per
token.

### Self-speculative early exit with confidence-gated verification

A single model could act as its own draft: run the cheap layers, check
the confidence, and stop early when it is high enough. The risk is a
distribution shift between the early and the full computation.

### Sliding-window attention with attention sinks

The block-table policy that implemented it was deleted: the attention
kernels map a logical position to a physical slot through the block
table, so a rolling window would read the wrong tokens. A correct
implementation needs a different mapping — a circular buffer of physical
blocks and an attention kernel that walks the window and the sinks
instead of the block table. The "constant VRAM at 100,000 tokens"
property is the point of it.

### The embedded WebUI, the nightly cross-engine benchmark, and the multi-target developer experience

The three parts of the old Phase 4 that were not built: the embedded UI
(the WebUI exists as a separate binary, not embedded — see
[webui.md](webui.md)), the nightly cross-engine benchmark (the scripts
exist and the nightly job now runs them), and the multi-target build
conveniences.

---

## What has landed

Everything below is on `main` and **unreleased** — the latest git tag is
`0.3.0`, and everything since it has landed on `main` without a tag.
Cutting a tag is a release decision, not a documentation edit.

| Area | What landed |
|------|-------------|
| **The shipped binary** | The CPU build was a stub that exited 1; it is now a real binary that runs inference. The OpenAI-compatible REST API is started and served, and `"stream": true` returns a real `text/event-stream`. The WebUI has a working CPU adapter, a CI job that runs its tests and a second one that runs its browser tests. |
| **The SIMD path** | The AVX-512 build was never compiled and had a SIGILL bug; it is now built, tested and gated. The TurboQuant C kernels were never called; they now are, and the CPU and CUDA encoders agree. The fuzz targets never ran — `make test-simd-fuzz` passed the flags in a form that fuzzed nothing — and they found two real bugs when fixed. The arm64 configuration is built and tested too, which is how the NEON kernel's wrong absolute-value mask was found: it rounded half to even while the CPU reference and the GPU round half away from zero. |
| **The CUDA path** | Six defects were found by a CPU-vs-CUDA parity test: the activation dtype, the KV pool's declared type, the KV store's token count, the scratch sizing, the block table's width and the embedding dimension. The broken single-token forward and ~1 GB of unused VRAM went with it. The engine used to throw the caller's configuration away: `-devices` and `-activation-log` were silently ignored. The quantized KV cache works now: the pool holds one byte per element, the store writes the codes and the scale of every slot, and a test compares it with the F16 path on the GPU. |
| **The model formats** | A GGUF writer, so a model can be created inside a test. The Q4_K and FP8 codecs were corrected against the GGML standard, and `cmd/quantize` produces quantized models. The model file is validated at load: a tensor table whose offsets do not describe the data behind it, or a model without an embedding table, is rejected with a message instead of a nil dereference. |
| **Speculative decoding** | Wired end to end through `-draft-model` and the API's `speculative` field, with the correct `min(1, p_target/p_draft)` acceptance rule and a residual correction. |
| **The multi-device flags** | `-devices 0,1`, `-tp-size 2` and `-pp-stages 2` now exit with an error instead of pretending to shard a model. The managers that could have done it are deleted: the inference engine loads the whole model onto one device. |
| **The audits and the activation logger** | Made real: `-activation-log` writes the activations as JSON, and the health audit reports on the logit distribution. |
| **The vector search** | The distances have hand-written AVX2 and AVX-512 kernels, the complex type is stored as split planes, the TurboQuant distance is asymmetric with a lookup table, the index is memory mapped and the GPU has batch kernels for the scan and the k-means. The inverted list is stored in the order it is scanned, so the scan is a sequential read, and the TurboQuant query matrices are built once per index instead of once per query. The TurboQuant HNSW search runs at 16,156 queries per second. The multi-query kernel derived the dimension of a block from the length of its buffer, so a block that was not full was scored against the wrong vectors; the dimension is a parameter now, and the k-means stopped falling back to the scalar path. The GPU assignment gives every lane one centroid instead of reducing one centroid across the lanes, and the flat buffer is built by all the workers: the build of a 250k×384 index went from 634 ms to 416 ms. |
| **The observability surface** | 133 Prometheus metrics, a Grafana dashboard over all of them, and a drift checker that fails the build when the dashboard, the Helm chart and the code disagree. Every metric is written by production code, and a static test says so. |
| **The configuration** | Every field of `config.Config` is either read or gone. The precision mode, the seven debug switches and the grouped-expert metadata were write-only; they are deleted or reported in the log. The stop tokens can be set from the command line and from the API, the maximum token count is the engine's default, and the KV cache type is a setting that reaches the allocation. |
| **The dead code** | The Arrow Flight client, the multi-GPU managers, the NCCL wrapper, the second radix cache, the FP8 and TurboQuant quantizers, the quality evaluator and the mock engine in the production package are gone: about 4,750 lines. What is left is a gate: the reachability analysis fails the build when a new unreachable function appears. |
| **The backends** | The TPU backend is deleted, with its image, its build target and its build tag. It did not compile, it was a sketch, and it had never been built by anything that reports a result. The Metal backend compiles, and its **test files** are type-checked too: they had not been compiled since the contiguous KV cache was deleted, and one of them called a helper that did not exist. |
| **The tests** | Every test in the repository can fail. The checker resolves the helper bodies transitively, across packages, and a skip is not an assertion. The five tokenizer tests that loaded a model from a hard-coded path on one developer's machine build their own now, and the CUDA tests build theirs, so the GPU hot path is covered on any machine with a GPU. |
| **The WebUI** | Every API endpoint answered 404: the router mounted the API under `/api/` without stripping the prefix, and the end-to-end suite called the handlers directly, so it never noticed. The browser tests were in the repository and could not run — there was no `package.json` and nothing started the server. They run now, and they found four more defects. The model gallery and the LoRA adapter browser are in the sidebar. |
| **The CI** | The lint, the build, the race detector, the CUDA compile gate, the Metal build and the Metal type check, the coverage gate, the duplicate-declaration check, the vacuous-test check, the dead-code check, the workflow lint, the security scans, the documentation drift check, the browser tests and the alternative builds (the `emlgo` and `emlfast` math backends and the build without cgo). The pipeline had been **disabled on the repository** since 2026-09-04; it is enabled again. |

Every item above was verified against the current code before it was
recorded, and the ones that turned out to be incomplete are in the open
work rather than here. The history of the individual changes is in the git
log; the gates above are the mechanism that keeps the history honest from
here on.
