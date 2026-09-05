<img width="2784" height="1536" alt="quarrel_logo" src="https://github.com/user-attachments/assets/e1ab45ae-f4de-4f68-91a5-fe931a720c21" />

# Longbow-Quarrel (v0.2.0)

High-performance, memory-efficient LLM inference engine written in Go with native GPU acceleration (Apple Silicon Metal, NVIDIA CUDA) and optimized CPU SIMD vectorization (AVX-512, AVX2, ARM NEON).

---

## What's New in v0.2.0

- **Zero-Copy Quantized Inference (RAM Exhaustion & OOM Elimination):** Direct matrix-vector dot products over memory-mapped quantized weights (`Q8_0`, `Q4_K`, `Q6_K`) and on-demand token embedding lookups. Slashes CPU heap memory by **99.9%** (from 17.9 GB down to < 20 MB for a 4B parameter model), allowing large models to run smoothly without disk swapping or OOM crashes.
- **Partial GPU Layer Offloading (`-ngl` / `-gpu-layers`):** Seamlessly split transformer layers across GPU VRAM and CPU host RAM with automatic activation roundtripping. Run models that exceed your GPU's dedicated VRAM.
- **Universal Multi-Engine Model Resolver:** Automatically discovers and fuzzy-matches models from `~/.cache/llmfit/models/`, `~/.cache/llama.cpp/`, `~/.cache/huggingface/hub/`, `~/.ollama/models/`, and local files. Pass `--model Qwen3.5` or `--model mistral` and Quarrel locates it instantly.
- **Qwen 3.5 Hybrid Architecture Support:** Native support for hybrid GatedDeltaNet linear State Space Models (SSM) and full self-attention with per-head Q/K RMSNorm.
- **Vectorized TurboQuant Kernels:** Full AVX-512, AVX2, and ARM NEON SIMD implementations for PolarQuant and QJL transforms with 16-lane fused multiply-accumulate operations.
- **SIMD GGUF Dequantization Kernels:** Vectorized batch dequantization and zero-copy matrix-vector multiplication (`MatVecMulQ4_K`, `MatVecMulQ6_K`).
- **Comprehensive Verification:** Clean `go vet`, 0 `gosec` security vulnerabilities, 0 data races (`go test -race`), and 600,000+ continuous fuzz test executions.

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
go test -bench=BenchmarkNeon ./internal/simd/...

# End-to-end inference benchmark
./cmd/benchmark --mode inference --model model.gguf --prompt "Benchmark prompt"
```

---

## Testing & Quality Assurance

```bash
# Run all unit tests
go test ./...

# Run CUDA tests
go test -tags cuda ./internal/device/... ./internal/engine/...

# Run with race detector
go test -race ./internal/...

# Run continuous fuzz testing
go test -fuzz=FuzzDequantizeQ4K_SIMD -fuzztime=30s ./internal/gguf/
go test -fuzz=FuzzPolarQuant -fuzztime=30s ./internal/simd/
go test -fuzz=FuzzApplyLayerCPU -fuzztime=30s ./internal/engine/
```

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