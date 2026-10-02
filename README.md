<img width="2784" height="1536" alt="quarrel_logo" src="https://github.com/user-attachments/assets/e1ab45ae-f4de-4f68-91a5-fe931a720c21" />

# Longbow-Quarrel

**Latest release: v0.3.0** (`git tag 0.3.0`). `main` is ahead of that tag with
unreleased engine, CI and dependency work — see [docs/nextsteps.md](docs/nextsteps.md)
for what has landed and what is still planned.

High-performance, memory-efficient LLM inference engine written in Go with native GPU acceleration (Apple Silicon Metal, NVIDIA CUDA) and optimized CPU SIMD vectorization (AVX-512, AVX2, ARM NEON).

---

## Requirements

| Component | Version | Notes |
|---|---|---|
| Go | 1.27+ | `go.mod` declares `go 1.27.0` |
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

- **Zero-Copy Quantized Inference (RAM Exhaustion & OOM Elimination):** Direct matrix-vector dot products over memory-mapped quantized weights (`Q8_0`, `Q4_K`, `Q6_K`) and on-demand token embedding lookups. Slashes CPU heap memory by **99.9%** (from 17.9 GB down to < 20 MB for a 4B parameter model), allowing large models to run smoothly without disk swapping or OOM crashes.
- **Partial GPU Layer Offloading (`-ngl` / `-gpu-layers`):** Seamlessly split transformer layers across GPU VRAM and CPU host RAM with automatic activation roundtripping. Run models that exceed your GPU's dedicated VRAM.
- **Universal Multi-Engine Model Resolver:** Automatically discovers and fuzzy-matches models from `~/.cache/llmfit/models/`, `~/.cache/llama.cpp/`, `~/.cache/huggingface/hub/`, `~/.ollama/models/`, and local files. Pass `--model Qwen3.5` or `--model mistral` and Quarrel locates it instantly.
- **Qwen 3.5 Hybrid Architecture Support:** Native support for hybrid GatedDeltaNet linear State Space Models (SSM) and full self-attention with per-head Q/K RMSNorm.
- **Vectorized TurboQuant Kernels:** Full AVX-512, AVX2, and ARM NEON SIMD implementations for PolarQuant and QJL transforms with 16-lane fused multiply-accumulate operations.
- **SIMD GGUF Dequantization Kernels:** Vectorized batch dequantization and zero-copy matrix-vector multiplication (`MatVecMulQ4_K`, `MatVecMulQ6_K`).
- **Comprehensive Verification:** Clean `go vet`, 0 `gosec` security vulnerabilities, 0 data races (`go test -race`), and 600,000+ continuous fuzz test executions.
- **Distributed Inference (Arrow Flight):** Tensor and pipeline parallelism over Apache Arrow Flight RPC. Note: the `DoPutTensor` round trip is currently non-functional — see the caveat in [docs/nextsteps.md](docs/nextsteps.md#known-defects).

---

## Features

### Model Support
- **Architectures**: Qwen 3.5 (Hybrid SSM/Linear + Full Attention), Llama 3/3.1/3.2, Mistral, Gemma 4 (Hybrid sliding + full), SmolLM2, Phi3, Granite
- **Quantizations**: Q4_0, Q4_K, Q6_K, Q8_0, FP16, FP32, FP8 (E4M3/E5M2), TurboQuant (PolarQuant + QJL)
- **Zero-Copy**: Weights remain in quantized mmap format; operations evaluate directly on quantized bytes

### Hardware Acceleration
- **Apple Silicon (Metal)**: 60+ custom MSL compute kernels (MatMul, RMSNorm, RoPE, SwiGLU, Flash Attention)
- **NVIDIA GPU (CUDA)**: Fused FP16/FP8 kernels, Tensor Cores, cuDNN flash attention, multi-GPU tensor & pipeline parallelism
- **CPU SIMD**: AVX-512, AVX2, and ARM64 NEON vectorized arithmetic
- **Hybrid Offloading**: Partition layers dynamically between GPU VRAM and CPU system RAM

### Advanced Serving & Cache
- **Continuous Batching**: Dynamic iteration-level batching with preemption
- **Paged KV Cache**: Virtual memory block allocation with prefix caching
- **TurboQuant KV Cache**: 8x KV cache compression with PolarQuant + QJL residual
- **Distributed Sharding**: Zero-copy tensor parallelism via Apache Arrow Flight RPC
- **API**: OpenAI-compatible (`/v1/chat/completions`, `/v1/completions`), WebSocket streaming, and Prometheus metrics (`/metrics`)

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
# Resolves from ~/.cache/llmfit/models/, ~/.cache/llama.cpp/, ~/.ollama/models/, or Hugging Face hub
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
go test -bench=NEON -benchmem ./internal/simd/...
go test -bench=AVX512 -benchmem ./internal/simd/...

# End-to-end inference benchmark
go run ./cmd/benchmark --mode inference --model model.gguf --prompt "Benchmark prompt"
```

---

## Testing & Quality Assurance

```bash
# Run all unit tests
go test ./...

# cmd/webui is a separate Go module with its own go.mod; test it separately
(cd cmd/webui && go build ./... && go vet ./... && go test ./...)

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

Packages below 80% are pinned to a ratchet floor in CI so coverage may rise
freely but never regress. The floors live in
`.github/workflows/ci.yml` and `scripts/run_all_tests.sh` and must be kept in sync.

| Package | Coverage |
|---|---|
| `internal/cpu` | 100.0% |
| `internal/logger` | 100.0% |
| `internal/metrics` | 98.7% |
| `internal/telemetry` | 98.1% |
| `internal/ollama` | 91.1% |
| `internal/models` | 92.5% |
| `internal/tokenizer` | 92.2% |
| `internal/vector` | 91.2% |
| `internal/vlm` | 85.5% |
| `internal/simd` | 85.5% |
| `internal/gguf` | 83.4% |
| `internal/api` | 82.9% |
| `internal/sampler` | 80.3% |
| `internal/arrow_client` | 78.8% |
| `internal/config` | 78.0% |
| `internal/device` | 73.5% |
| `internal/engine` | 50.1% |

---

## Project Structure

```
cmd/
  simple/          # Minimal CLI inference with universal resolver & offload
  quarrel/         # High-performance CUDA CLI & server (Linux)
  benchmark/       # SIMD & inference benchmarking
  webui/           # Templ-based web UI with WebSocket streaming
internal/
  engine/          # Core inference engine, continuous batching, layer offload
  device/          # GPU backends (Metal, CUDA, CPU, multi-GPU)
  simd/            # SIMD kernels (AVX-512, AVX2, ARM NEON, TurboQuant)
  gguf/            # GGUF parser, metadata extraction, zero-copy dequant
  models/          # Universal multi-engine model cache resolver
  tokenizer/       # Fast BPE & SentencePiece tokenization
  metrics/         # Prometheus observability instrumentation
docs/              # Detailed architecture, API, and performance documentation
```

---

*For detailed specifications, see the [Documentation Index](docs/).*