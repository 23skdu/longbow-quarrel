<img width="2784" height="1536" alt="quarrel_logo" src="https://github.com/user-attachments/assets/e1ab45ae-f4de-4f68-91a5-fe931a720c21" />

# Longbow-Quarrel

**Latest release: v0.3.0** (`git tag 0.3.0`). `main` is ahead of that tag with
unreleased engine, CI and dependency work — see [docs/roadmap.md](docs/roadmap.md)
for what has landed and what is still planned.

> ### ⚠️ Known state of `main`
>
> Two full-tree audits of `main` (October 2026) found that the project did
> **not** ship a working CPU binary, that its CI `lint` job was **red**, that
> the OpenAI-compatible REST API was **never started**, that the macOS Metal
> build **did not compile**, and that the CI pipeline itself had been
> **disabled on the repository since 2026-09-04**. All of that is fixed: the
> CPU binary runs real inference, the REST API is served, the Metal build
> compiles, and the pipeline is green on every job.
>
> **The roadmap.** Every defect the roadmap listed is closed: the CUDA
> engine preserves the caller's configuration, every produced metric is
> written by production code, the Metal backend compiles and has a CPU
> parity test, the TPU backend has been deleted rather than left
> unbuildable, and the ~4,750 lines of dead code are gone behind a gate
> that fails when a new one appears. What is still open — the planned
> features and the vector-search performance targets — is in
> [docs/roadmap.md](docs/roadmap.md).

High-performance, memory-efficient LLM inference engine written in Go with native GPU acceleration (Apple Silicon Metal, NVIDIA CUDA) and optimized CPU SIMD vectorization (AVX-512, AVX2, ARM NEON).

---

## Requirements

| Component | Version | Notes |
|---|---|---|
| Go | 1.27+ | `go.mod` declares `go 1.27.2` |
| CUDA (NVIDIA path) | 12.9 | Driver ≥ 575.51.03, or ≥ 525.60.13 with minor-version compatibility |
| cuDNN | 9 | Package name `libcudnn9-cuda-12` tracks the CUDA 12 series |
| Alpine (CPU/metal images) | 3.24 | |
| Prometheus (monitoring stack) | 3.15 | |
| Grafana (monitoring stack) | 13.2 | |

Pinned third-party dependency versions are tracked in one place —
see [docs/dependencies.md](docs/dependencies.md).

> **Note on `google.golang.org/grpc`:** this module is pinned to the exact
> commit that fixes [GO-2026-6443](https://pkg.go.dev/vuln/GO-2026-6443)
> (CVE-2026-84445), a server panic in xDS routing. No tagged release contains
> the fix yet, so the pseudo-version cannot be replaced with a stable `v1.x.y`
> without reintroducing the vulnerability. Re-check after the next gRPC tag.

---

## What's New since v0.2.0

| Area | What changed |
|---|---|
| **The shipped binary** | The CPU build was a stub that exited 1; it is now a real binary that runs inference. The OpenAI-compatible REST API is served, and `"stream": true` returns a real `text/event-stream`. |
| **Memory** | The weights stay memory-mapped and quantized: the dot products run over the bytes directly, so a 4B model needs < 20 MB of heap instead of 17.9 GB. |
| **The SIMD path** | The AVX-512 build is compiled, tested and gated. The TurboQuant C kernels are called, and the CPU and CUDA encoders agree. The arm64 configuration is built and tested too. |
| **The CUDA path** | A CPU-vs-CUDA parity test found six defects in the batched forward; the quantized (int8/FP8) paged KV cache works. |
| **Speculative decoding** | Wired end to end through `-draft-model` and the API's `speculative` field, with the `min(1, p_target/p_draft)` acceptance rule. |
| **The model formats** | A GGUF writer, a corrected Q4_K and FP8 codec, `cmd/quantize`, and a model file validated at load. |
| **The observability surface** | 133 Prometheus metrics, a Grafana dashboard over all of them, and a drift checker that fails the build when the chart, the dashboard and the code disagree. |
| **The gates** | A dead-code gate, a vacuous-test gate, a duplicate-declaration gate, a workflow lint, the security scans and the documentation drift check. |

The full record, including the defects these fixes closed, is in
[docs/roadmap.md](docs/roadmap.md#what-has-landed); the feature list is in
[docs/features.md](docs/features.md).

---

## Features

- **Models**: Qwen 3.5 (hybrid SSM + attention), Llama 3/3.1/3.2, Mistral,
  Gemma 4, SmolLM2, Phi3, Granite; Q4_0, Q4_K, Q6_K, Q8_0, FP16, FP32, FP8
  and TurboQuant.
- **Backends**: CUDA and Metal on the GPU, AVX-512, AVX2 and ARM NEON on the
  CPU, with the layers split between the GPU and the host (`-ngl`).
- **Serving**: continuous batching, a paged KV cache with prefix caching, an
  8× compressed TurboQuant KV cache, speculative decoding, an
  OpenAI-compatible REST API, an Arrow Flight server and a WebUI.
- **Operations**: 133 Prometheus metrics, a Grafana dashboard, a Kubernetes
  Helm chart, Docker images and an engine health audit.

The comparison with the other engines is in
[docs/features.md](docs/features.md).

---

## Quick Start

### Build & Run

```bash
# 1. CPU Mode (Default, Zero-Copy SIMD)
go build -o quarrel ./cmd/simple/
./quarrel -model Huihui-Qwen3.5 -prompt "Explain quantum computing briefly."

# 2. NVIDIA CUDA Mode (Linux)
make nvidia
./bin/quarrel-linux-amd64-cuda -model mistral:latest -gpu-layers 24 -prompt "Hello!"

# 3. Partial GPU Offloading (e.g. 16 layers on GPU, remainder on CPU)
./quarrel -model /path/to/model.gguf -ngl 16 -prompt "Tell me a story"

# 4. Apple Silicon Metal Mode (macOS)
go run -tags darwin,metal ./cmd/simple/main.go -model Llama-3.2-3B -prompt "Hello"
```

### Universal Model Resolver

No need to pass long file paths:
```bash
# Resolves from ~/.cache/llmfit/models/, ~/.cache/llama.cpp/,
# ~/.cache/huggingface/hub/, or an Ollama tag (mistral:latest)
./quarrel -model Qwen3.5
./quarrel -model mistral
./quarrel -model llama3
```

---

## Benchmark & Performance

```bash
# Benchmark CPU zero-copy kernels & SIMD
go test -bench=BenchmarkDequantize -benchmem ./internal/gguf/...

# Benchmark TurboQuant SIMD kernels (AVX-512 / AVX2 / NEON)
go test -bench=PolarQuant -benchmem ./internal/simd/...
go test -tags avx512 -bench=AVX512 -benchmem ./internal/simd/...
# The NEON benchmarks are in the arm64 build.
GOARCH=arm64 go test -bench=NEON -benchmem ./internal/simd/...

# End-to-end inference benchmark
go run ./cmd/benchmark --mode inference --model model.gguf --prompt "Benchmark prompt"
```

---

## Testing & Quality Assurance

```bash
# Run all unit tests
go test ./...

# cmd/webui is a separate Go module with its own go.mod; test it separately
(cd cmd/webui && go build -tags webui ./... && go vet -tags webui ./... \
  && go test -tags webui ./...)

# Run CUDA tests
go test -tags cuda ./internal/device/... ./internal/engine/...

# Run with race detector
go test -race ./internal/...

# Run continuous fuzz testing
go test -fuzz=FuzzDequantizeQ4K_SIMD -fuzztime=30s ./internal/gguf/
go test -fuzz=FuzzPolarQuant -fuzztime=30s ./internal/simd/
go test -fuzz=FuzzApplyLayerCPU -fuzztime=30s ./internal/engine/

# Enforce that every test can actually fail (fails on new vacuous tests)
go run ./scripts/vacuous_tests
```

### Current test coverage

Measured with `go test -cover ./internal/...` on the **default (no build tag)
build**. Packages below 80% are pinned to a ratchet floor in CI so coverage may
rise freely but never regress. The floors live in `scripts/coverage_gate.sh`.

The table is generated: `scripts/coverage_table.py` measures, renders and
rewrites it, and the CI fails when it is out of date.

<!-- coverage:start -->

> These numbers cover the default build only. `internal/device` has
> 92 `.go` files but only 14 compile without a build tag, and
> `internal/engine` has 114 files of which 71 compile — all CUDA,
> Metal, TPU and AVX-512 code is excluded from both the numerator
> and the denominator. The percentages therefore say nothing about
> the GPU backends.

| Package | Coverage |
|---|---|
| `internal/cpu` | 100.0% |
| `internal/logger` | 100.0% |
| `internal/mathx` | 100.0% |
| `internal/models` | 100.0% |
| `internal/ollama` | 100.0% |
| `internal/promtest` | 100.0% |
| `internal/sampler` | 100.0% |
| `internal/vlm` | 100.0% |
| `internal/metrics` | 99.5% |
| `internal/tokenizer` | 98.6% |
| `internal/simd` | 95.5% |
| `internal/telemetry` | 94.0% |
| `internal/config` | 88.1% |
| `internal/api` | 87.9% |
| `internal/gguf` | 86.3% |
| `internal/vector` | 84.8% |
| `internal/engine` | 80.4% |
| `internal/device` | 78.6% |
| `internal/testmodel` | 0.0% |

<!-- coverage:end -->

---

## Project Structure

```
cmd/
  quarrel/         # The engine and its REST API
  simple/          # A minimal CLI over the same engine
  webui/           # The browser UI (its own Go module)
internal/
  engine/          # The inference engine, the KV caches, the batching
  device/          # The backends: CUDA, Metal, CPU
  simd/            # The SIMD kernels: AVX-512, AVX2, ARM NEON
  gguf/            # The GGUF reader and writer, the zero-copy dequant
  vector/          # The vector search package
  metrics/         # The Prometheus instrumentation
docs/              # The documentation index
```

---

*For the full layout, see [architecture.md](docs/architecture.md); for the
specifications, the [documentation index](docs/).*