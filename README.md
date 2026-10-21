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
> **Still outstanding — the remaining P1/P2 items are listed in full in
> [docs/roadmap.md](docs/roadmap.md).** The four P0 items recorded there
> are all closed: the CUDA engine now preserves the caller's
> configuration, every produced metric is written by production code or
> explicitly allowlisted, the Metal backend compiles and has a CPU parity
> test, and the TPU backend has been deleted rather than left unbuildable.
>
> The engine also had ~4,750 lines of dead code; the deletion and wiring
> pass that removes it is recorded in the roadmap.

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

- **Zero-Copy Quantized Inference (RAM Exhaustion & OOM Elimination):** Direct matrix-vector dot products over memory-mapped quantized weights (`Q8_0`, `Q4_K`, `Q6_K`) and on-demand token embedding lookups. Slashes CPU heap memory by **99.9%** (from 17.9 GB down to < 20 MB for a 4B parameter model), allowing large models to run smoothly without disk swapping or OOM crashes.
- **Partial GPU Layer Offloading (`-ngl` / `-gpu-layers`):** Seamlessly split transformer layers across GPU VRAM and CPU host RAM with automatic activation roundtripping. Run models that exceed your GPU's dedicated VRAM.
- **Universal Multi-Engine Model Resolver:** Automatically discovers and fuzzy-matches models from `~/.cache/llmfit/models/`, `~/.cache/llama.cpp/`, `~/.cache/huggingface/hub/`, `~/.cache/quarrel/models/`, `~/.local/share/nomic.ai/GPT4All/`, and local files. Ollama tags (`mistral:latest`) are resolved through the Ollama manifest store (`~/.ollama/models/` or `$OLLAMA_MODELS`). Pass `--model Qwen3.5` or `--model mistral` and Quarrel locates it instantly.
- **Qwen 3.5 Hybrid Architecture Support:** Native support for hybrid GatedDeltaNet linear State Space Models (SSM) and full self-attention with per-head Q/K RMSNorm.
- **Vectorized TurboQuant Kernels:** AVX-512, AVX2 and ARM NEON implementations of the PolarQuant and QJL transforms, dispatched at runtime by CPU feature detection, with a canonical pure-Go fallback.
- **SIMD GGUF Dequantization Kernels:** Vectorized batch dequantization and zero-copy matrix-vector multiplication (`MatVecMulQ4_K`, `MatVecMulQ6_K`).
- **Comprehensive Verification:** Clean `go vet`, 0 `gosec` security vulnerabilities, 0 data races (`go test -race`), a `vacuous_tests` gate that fails any test which cannot fail, and a `dupcheck` gate that fails on duplicate package-level declarations in any build configuration.
- **CPU/GPU Parity:** The CPU and the GPU are compared on the same model, on a model the test suite builds itself. Six defects in the GPU's batched forward came out of it: the activations were half precision where the kernels are single precision, the tensor pool handed out buffers with the wrong element type, the KV store was called with the batch size instead of the token count, the scratch was sized for one token, the block table could be zero wide, and the embedding lookup used the vocabulary count where the dimension belongs.
- **Correct Paged Attention:** The position tensors, the block tables and the physical slots all say what they hold, on the CUDA and the Metal side alike. The CPU and the GPU parity test checks the result, and a device-level test checks that every token lands in the slot its position names.
- **Model Quantization:** `quantize -i model.gguf -o model-q4_k.gguf -t q4_k` converts a model into a quantized one. The repository could read GGUF files but not write them, so this was not possible before; and the Q4_K and FP8 codecs did not round-trip, so the CPU and the GPU decoded the same Q4_K model differently.
- **Engine Health Audit:** Every forward pass audits its own logit distribution. A model that produces NaNs, infinities, extreme values or a flat distribution is reported in the log and in the Prometheus metrics (`logit_nan_count_total`, `logit_flat_distribution_total`, `logit_max_value`, and the rest) — all of which were previously permanently zero.
- **Activation Logging:** `quarrel -activation-log <path>` writes a JSON file with the embedding, the per-layer activations (peak magnitude, a sample, and the NaN and Inf counts of each tensor) and the final logits, so a broken layer can be found.
- **Speculative Decoding:** A smaller draft model proposes the next tokens and the target model verifies them in one forward pass, with the standard `min(1, p_target/p_draft)` acceptance test and a residual-distribution correction. Enable it with `-draft-model <path>`; the draft model runs on the CPU so the GPU memory stays with the target.
- **Distributed Inference (Arrow Flight):** A Flight client, a master engine, a remote worker engine and a Flight server for single-request inference. The **sharded** tensor forward is *not* implemented on the worker side: the worker rejects it with `Unimplemented` rather than silently returning zeroed activations. See `docs/roadmap.md` (Step 57) for the exact boundary.

---

## Features

### Model Support
- **Architectures**: Qwen 3.5 (Hybrid SSM/Linear + Full Attention), Llama 3/3.1/3.2, Mistral, Gemma 4 (Hybrid sliding + full), SmolLM2, Phi3, Granite
- **Quantizations**: Q4_0, Q4_K, Q6_K, Q8_0, FP16, FP32, FP8 (E4M3/E5M2), TurboQuant (PolarQuant + QJL)
- **Zero-Copy**: Weights remain in quantized mmap format; operations evaluate directly on quantized bytes

### Hardware Acceleration
- **Apple Silicon (Metal)**: 60+ custom MSL compute kernels (MatMul, RMSNorm, RoPE, SwiGLU, Flash Attention)
- **NVIDIA GPU (CUDA)**: Fused FP16/FP8 kernels, Tensor Cores, cuDNN flash attention, and a single selectable device (`-devices N`). Multi-device tensor and pipeline parallelism are implemented in the device layer but **not used by the engine**: `-tp-size 2` / `-pp-stages 2` exit with an error rather than pretending to shard the model.
- **CPU SIMD**: AVX-512, AVX2, and ARM64 NEON vectorized arithmetic
- **Hybrid Offloading**: Partition layers dynamically between GPU VRAM and CPU system RAM

### Advanced Serving & Cache
- **Continuous Batching**: Dynamic iteration-level batching with preemption
- **Paged KV Cache**: Virtual memory block allocation with prefix caching
- **TurboQuant KV Cache**: 8x KV cache compression with PolarQuant + QJL residual
- **Distributed Sharding**: Client, master and worker plumbing via Apache Arrow Flight RPC. The worker-side sharded forward is not implemented yet (it fails loudly, not silently).
- **Speculative Decoding**: Draft-model token proposals verified by the target model, exposed as `-draft-model` and as the REST API's `"speculative"` / `"draft_k"` fields.
- **API**: OpenAI-compatible (`/v1/chat/completions`, `/v1/completions`) on the API listener, and Prometheus metrics (`/metrics`) on a separate metrics listener. The WebSocket streaming lives in the WebUI binary, not in the API server.

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
> 92 `.go` files but only 13 compile without a build tag, and
> `internal/engine` has 122 files of which 78 compile — all CUDA,
> Metal and AVX-512 code is excluded from both the numerator
> and the denominator. The percentages therefore say nothing about
> the GPU backends.

| Package | Coverage |
|---|---|
| `internal/config` | 100.0% |
| `internal/cpu` | 100.0% |
| `internal/logger` | 100.0% |
| `internal/mathx` | 100.0% |
| `internal/models` | 100.0% |
| `internal/ollama` | 100.0% |
| `internal/promtest` | 100.0% |
| `internal/sampler` | 100.0% |
| `internal/vector` | 100.0% |
| `internal/vlm` | 100.0% |
| `internal/metrics` | 99.5% |
| `internal/tokenizer` | 98.5% |
| `internal/simd` | 95.2% |
| `internal/telemetry` | 94.0% |
| `internal/api` | 91.6% |
| `internal/gguf` | 89.6% |
| `internal/arrow_client` | 82.0% |
| `internal/engine` | 80.4% |
| `internal/device` | 75.3% |

<!-- coverage:end -->

---

## Project Structure

```
cmd/
  simple/          # Minimal CLI inference with universal resolver & offload
  quarrel/         # High-performance CUDA CLI & server (Linux)
  benchmark/       # SIMD & inference benchmarking
  webui/           # Server-rendered web UI with WebSocket streaming (stdlib html/template)
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