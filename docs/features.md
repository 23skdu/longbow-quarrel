# Longbow-Quarrel Features

A high-performance LLM inference engine written in Go with native GPU acceleration.

## Inference Engine

- **GGUF Model Loading** — Parse and load GGUF format models with full metadata support
- **Token Generation Loop** — Autoregressive sampling with configurable parameters
- **Sampler** — Temperature, top-K, top-P (nucleus), repetition penalty
- **Batched Inference** — Concurrent request handling with per-sequence KV cache isolation
- **Streaming Output** — Real-time token streaming via SSE and WebSocket
- **Model Architectures** — Llama 3, Mistral, Qwen2, Phi3, StarCoder2, Granite

## Quantization Support

| Format | Dequantization | Quantization | Notes |
|--------|---------------|-------------|-------|
| FP16 / FP32 | ✅ | ✅ | Native precision |
| Q3_K | ✅ | - | Low-bit K-quant |
| Q4_K | ✅ | - | Standard K-quant |
| Q6_K | ✅ | - | High-quality K-quant |
| Q8_0 | ✅ | - | 8-bit quantization |
| IQ1_M | ✅ | - | 1-bit quantization (56 bytes/256 elements) |
| FP8 E4M3 / E5M2 | ✅ | ✅ | H100 Hopper support |

## GPU Backends

### Metal (Apple Silicon)
- 61 custom GPU kernels (FP16, Q3_K, Q4_K, Q6_K, Q8_0)
- MatMul, RMSNorm, RoPE, SwiGLU, Attention kernels
- Thread-safe async GPU dispatch with tensor pooling and memory budget
- MetalPerformanceShaders integration

### CUDA (NVIDIA)
- Tensor Core WMMA support
- Flash Attention kernel
- Paged KV Cache on GPU
- cuDNN integration for grouped convolutions and optimized attention
- FP8 Tensor Core support (Hopper architecture)
- Multi-GPU support:
  - Tensor parallelism with AllReduce, AllGather, Broadcast collectives
  - Pipeline parallelism with micro-batch scheduling
  - Peer-to-peer cross-GPU communication
  - Hybrid parallelism manager

### CPU Fallback
- AVX2 SIMD: Softmax, SwiGLU, FP16-to-FP32 conversion
- AVX-512: Zen 4+ and Ice Lake+ support
- Reference implementation for all operations

## KV Cache Management

- **Contiguous Cache** — Traditional sequential allocation
- **Paged Cache** — Virtual memory-style page allocation for efficient memory use
- **Sliding Window** — Fixed-window attention for long contexts
- **Shared Cache** — Cross-sequence cache sharing for prefix caching
- **Per-Sequence Tracking** — Independent cache positions per concurrent request

## Advanced Model Support

### Mixture of Experts (MoE)
- Expert routing with configurable top-K selection
- Grouped convolution support via cuDNN
- Per-expert weight loading and activation

### Mamba / State Space Models
- Hybrid Mamba/Transformer architecture support
- Config-based layer detection (`all`, `none`, `even`, `odd`, custom patterns)
- ConvState and SSMState management with proper zero-initialization
- Automatic detection from GGUF metadata

## WebUI Service

- **Templ-based UI** — Server-rendered HTML with real-time updates
- **WebSocket Streaming** — Bidirectional real-time inference
- **Model Hot-Swapping** — Zero-downtime model switching with active request draining
- **API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/healthz` | GET | Simple liveness probe |
| `/readyz` | GET | Readiness probe |
| `/version` | GET | Version info |
| `/metrics` | GET | Prometheus metrics |
| `/api/models` | GET | List loaded models |
| `/api/generate` | POST | Synchronous text generation |
| `/api/stream` | POST | SSE streaming generation |
| `/ws` | WebSocket | Real-time inference |

## Observability

- **Prometheus Metrics** — Inference latency, throughput, KV cache utilization, GPU memory, MoE layer latency, token counts
- **Grafana Dashboards** — Pre-built dashboard JSON for deployment monitoring
- **Structured Logging** — Zero-based logger with configurable levels
- **Health Checks** — Liveness and readiness probes for orchestration
- **Activation Tracing** — Layer-by-layer activation logging and debugging
- **Quality Evaluation** — Perplexity and output validation metrics

## Deployment

- **Docker** — Multi-stage Dockerfile for CPU and CUDA variants
- **Kubernetes** — Helm chart (v0.2.0) with configurable replicas and resource limits
- **Nginx** — Reverse proxy configuration with SSL termination
- **Cross-platform Builds** — darwin/amd64, darwin/arm64, linux/amd64

## vLLM Integration

- Export Go CUDA operators as C-shared library for Python vLLM
- Exported functions: Init, MatMul, RMSNorm, SwiGLU, RoPE, Attention, Dequantize, Synchronize
- Enables using Quarrel's CUDA kernels from Python inference pipelines

## Ollama Integration

- Native Ollama model resolution (e.g., `mistral:latest` resolves to GGUF path)
- Reads model registry from `~/.ollama/models`
- Seamless model loading without manual path specification

## Validated Models

| Model | Size | Architecture | Status |
|-------|------|-------------|--------|
| SmolLM2 | 135M / 360M | Llama-like | ✅ Bundled test model |
| Llama 3.2 | Various | Llama 3 | ✅ Validated |
| Mistral | 7B | Mistral | ✅ Validated |
| Granite | 4B | Granite | ✅ Validated |
| TinyLlama | 1.1B | Llama-like | ✅ Validated |
| Nemotron-3-Nano | MoE | MoE | ✅ Validated |

## Testing

- **102 test files** with unit, integration, fuzz, smoke, and E2E tests
- **3 fuzz test suites** covering KV cache, sampler, and engine hot-swap
- **Playwright E2E** for WebUI validation
- **CI/CD** — GitHub Actions matrix across Go 1.24/1.25/1.26, daily scheduled runs
- **Coverage** — ~46.8% overall (config 100%, logger 100%, metrics 88.2%)
- **Benchmarks** — Go benchmarks with llama.cpp comparison baselines
