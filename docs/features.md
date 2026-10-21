# Longbow-Quarrel Features and Comparison

> **Version scope.** This document describes release **v0.3.0**, the latest
> git tag. `main` carries additional unreleased work; see
> [roadmap.md](roadmap.md) for what has landed and what is planned.

A high-performance, memory-efficient LLM inference engine written in Go
with native GPU acceleration and advanced SIMD vectorization.

The first half is what the engine does; the second is how it compares
with the other engines, and when to pick which.

---

## 1. Inference engine

- **Zero-Copy Quantized Inference** — Direct dot products over memory-mapped weights (`Q8_0`, `Q4_K`, `Q6_K`) and on-demand token embeddings, reducing heap footprint by 99.9% and preventing system OOM.
- **Partial GPU Layer Offloading** — Dynamic layer partitioning (`-ngl` / `-gpu-layers`) between GPU VRAM and CPU RAM with automatic activation roundtripping.
- **Universal Model Resolver** — Automatic model discovery and fuzzy matching across `~/.cache/llmfit/models/`, `~/.cache/llama.cpp/`, `~/.cache/huggingface/hub/`, `~/.ollama/models/`, and local filesystem paths.
- **Dynamic Architecture Parsing** — Full support for Llama 3/3.1/3.2, Qwen 3.5, Mistral, Gemma 4, SmolLM2, Phi3, and Granite.
- **Qwen 3.5 Hybrid Attention** — Native support for alternating GatedDeltaNet linear SSM and Full Self-Attention with per-head Q/K RMSNorm.
- **Token Generation Loop** — Autoregressive sampling with temperature, top-K, top-P (nucleus), and repetition penalty.
- **Batched Inference** — Continuous batching with iteration-level preemption and per-sequence KV cache isolation.
- **Streaming Output** — Real-time token streaming via Server-Sent Events (SSE) and WebSocket, in the WebUI. The engine's own REST API accepts a `"stream": true` field and answers with a `text/event-stream` of `chat.completion.chunk` objects (see [api.md](api.md)).
- **Jinja2 Chat Template Rendering** — Auto-applies `tokenizer.chat_template` from GGUF KV metadata with lightweight Jinja2 subset renderer (`{% for %}`, `{% if %}`, `{% set %}`, `{{ var }}`). Falls back to simple template when not available.
- **PromptCache (Radix-Tree Prefix Caching)** — Shared prompt prefix caching across requests for reduced TTFT. Populated after sequence completion in CPU, CUDA, and Metal engines.
- **Model Quantization** — `cmd/quantize` reads a GGUF model and writes a quantized one (`q4_k`, `q6_k`, `q8_0`, `f16`, `f32`), keeping the metadata and the token embedding.
- **Engine Health Audit** — Every forward pass audits its own logit distribution and reports NaNs, infinities, extreme values and flat distributions, in the log and in the Prometheus metrics.
- **Activation Logging** — `quarrel -activation-log <path>` writes the embedding, the per-layer activations (with NaN and Inf counts) and the final logits as JSON.
- **Speculative Decoding** — Multi-token draft forward pass with real logit output (not stubs). The standard `min(1, p_target/p_draft)` acceptance test, with a residual-distribution correction for rejected tokens. Enabled with `-draft-model <path>`, or per request through the REST API's `"speculative"` / `"draft_k"` fields.
- **LoRA Adapter Loading** — CPU merge-on-load strategy fusing A×B×α/r delta weights into CPUWeights at load time. Supports .gguf sidecar format. Also supported on Metal and CUDA.
- **Pluggable Math Backend** — All `math.*` calls route through `internal/mathx/`, switchable between stdlib `math` (default) and [`emlgo`](https://github.com/23skdu/emlgo) (SIMD batch) via `-tags emlgo`, with `pkg/fastmath` approximations behind `-tags emlfast`. Enables A/B performance testing from a single tree. See [roadmap.md](roadmap.md#planned-features).

---

## 2. Quantization and formats

The first three columns are what Quarrel does with the format: quantize
it, dequantize it, and run a matrix-vector product over the quantized
bytes without materializing them. The rest is which format each other
engine reads.

| Format | Quantize | Dequant | Zero-copy matvec | Notes | llama.cpp | vLLM | Ollama | TensorRT-LLM |
|--------|:---:|:---:|:---:|-------|-----------|------|--------|---------------|
| **FP32** | ✅ | ✅ | ✅ (`MatVecMul`) | | ✅ | - | ✅ | ✅ |
| **FP16** | ✅ | ✅ | ✅ | | ✅ | ✅ | ✅ | ✅ |
| **BF16** | ✅ | ✅ | ✅ (`MatVecMulBF16`) | | ✅ | ✅ | ✅ | ✅ |
| **Q8_0** | ✅ | ✅ | ✅ (`MatVecMulQ8_0`) | The default zero-copy format | ✅ | - | ✅ | ✅ |
| **Q6_K** | ✅ | ✅ | ✅ (`MatVecMulQ6_K`) | SIMD batch dequant | ✅ | - | ✅ | - |
| **Q5_K** | - | ✅ | ✅ (`MatVecMulQ5_K`) | | ✅ | - | ✅ | - |
| **Q5_0** | - | ✅ | ✅ (`MatVecMulQ5_0`) | | ✅ | - | ✅ | - |
| **Q4_K** | ✅ | ✅ | ✅ (`MatVecMulQ4_K`) | SIMD batch dequant | ✅ | - | ✅ | - |
| **Q4_0** | - | ✅ | ✅ (`MatVecMulQ4_0`) | | ✅ | - | ✅ | - |
| **Q3_K** | - | ✅ | ✅ (`MatVecMulQ3_K`) | | ✅ | - | ✅ | - |
| **Q2_K** | - | ✅ | ✅ (`MatVecMulQ2_K`) | | ✅ | - | ✅ | - |
| **FP8 (E4M3 / E5M2)** | ✅ | ✅ | - | Hopper Tensor Core format | ✅ | ✅ | - | ✅ |
| **TurboQuant** | ✅ | ✅ | - | 8× KV compression (PolarQuant + QJL) | - | - | - | - |
| **GPTQ** | - | - | - | | - | ✅ | - | ✅ |
| **AWQ** | - | - | - | | - | ✅ | - | ✅ |
| **GGUF (generic)** | ✅ | ✅ | ✅ | | ✅ | - | ✅ | - |

llama.cpp has the same zero-copy matrix-vector path over the K-quants
and the legacy quants; vLLM and TensorRT-LLM read FP8 and GPTQ/AWQ but
not the K-quants.

---

## 3. Hardware acceleration and backends

### NVIDIA CUDA
- Dynamic partial GPU layer offloading (`-ngl`) to bound VRAM consumption.
- Tensor Core WMMA support and cuDNN Flash Attention kernels.
- Paged KV cache directly in GPU VRAM.
- Multi-GPU tensor parallelism and pipeline parallelism — **not implemented**. The managers that existed were never used by the inference engine, which loads the whole model onto one device, and they have been deleted. `-devices 0,1` / `-tp-size 2` / `-pp-stages 2` exit with an error rather than pretending to shard.
- Automated compilation via `make nvidia` (`internal/device/libcuda_kernels.a`).

### Apple Silicon Metal
- 60+ custom Metal Shading Language (MSL) kernels (MatMul, RMSNorm, RoPE, SwiGLU, Flash Attention).
- Thread-safe async GPU command submission with tensor pooling and memory budgeting.

### CPU SIMD Subsystem
- **AVX-512**: Vectorized TurboQuant Step 3 inverse rotation using 16-lane `_mm512_fmadd_ps` with row-major linear combinations; QJL transforms.
- **AVX2**: Vectorized inverse rotation using 8-lane `_mm256_fmadd_ps`, SwiGLU, RMSNorm, and Softmax.
- **ARM NEON**: Vectorized inverse rotation and PolarQuant using `vfmaq_f32` with contiguous memory traversal.

---

## 4. KV cache management

- **Contiguous Cache** — Traditional sequential buffer allocation.
- **Paged Cache** — Virtual memory page table allocation for efficient memory use and fragmented requests.
- **Sliding Window Attention** — Fixed-window attention for long contexts (Mistral 4096, Gemma 4 hybrid).
- **TurboQuant Compression** — 8x KV cache compression with PolarQuant and QJL residual matrices.
- **Per-Sequence Tracking** — Independent cache position and sequence rollback support (`RollbackKV`).
- **PromptCache Integration** — Radix-tree prefix caching for repeated/shared prompts. Inserted after CompleteSequence in all engines.

---

## 5. Observability and telemetry

- **Prometheus Metrics** (`/metrics`):
  - `quarrel_gpu_layers_active`: Active GPU offloaded layer gauge.
  - `quarrel_cpu_layers_active`: Active CPU fallback layer gauge.
  - `quarrel_layer_offload_transfers_total`: Host-device activation transfer counter.
  - `quarrel_layer_offload_duration_seconds`: CPU layer execution latency histogram.
  - `inference_tokens_total`, `inference_duration_seconds`, `gpu_memory_allocated_bytes`.
  - `speculative_tokens_accepted_total`, `speculative_tokens_rejected_total`.
  - Paged KV cache hit/miss/eviction rates and out-of-bounds protection counters.
- **Quality & Numerical Stability Auditing**: NaN/Inf detection, activation health logging, and logit distribution validation.

---

## 6. Testing and quality assurance

- **Unit & Parity Tests**: Full coverage across device backends, engine pipelines, GGUF reader, and tokenizers.
- **Continuous Fuzz Test Suites**:
  - `FuzzApplyLayerCPU` — Validates CPU layer execution under randomized hidden states, dimensions, and layer indices.
  - `FuzzPolarQuant` & `FuzzQJLTransform` — Validates SIMD TurboQuant vector operations against mathematical boundaries.
  - `FuzzDequantizeQ4K_SIMD` & `FuzzDequantizeQ6K_SIMD` — Validates SIMD dequantization parity against scalar baselines.
  - `FuzzPagedKVCache_TurboQuant` & `FuzzSamplerQualityControl` — Validates the paged KV cache and the sampler against input edge cases.
  - `FuzzKVCacheUpdate` & `FuzzSharedKVCache` — Validates KV cache updates and the shared-cache path.
  - `FuzzMatMul` — Validates MatMul with VecDotF32 inner loop against reference implementation.
- **Security & Concurrency Audits**:
  - `go vet ./...` & `go vet -tags cuda ./...` — Clean (0 warnings).
  - `gosec` — Clean (0 vulnerabilities, 158 verified `#nosec` directives).
  - `go test -race` — Clean (0 data races across all packages).

---

## 7. The engines compared

| Engine | Language | License | Primary Backend | GPU Support |
|--------|----------|---------|-----------------|-------------|
| **Longbow-Quarrel** | Go | Open Source | CPU (SIMD) + CUDA + Metal | NVIDIA, Apple Silicon |
| **llama.cpp** | C/C++ | MIT | CPU (SIMD) + CUDA + Metal + Vulkan | NVIDIA, AMD, Apple, Intel |
| **vLLM** | Python/C++ | Apache 2.0 | CUDA (PagedAttention) | NVIDIA only |
| **Ollama** | Go | MIT | llama.cpp wrapper | NVIDIA, Apple |
| **TensorRT-LLM** | C++/Python | Apache 2.0 | CUDA (TensorRT) | NVIDIA only |
| **MLC-LLM** | C++/Python | Apache 2.0 | Multi-backend | NVIDIA, Apple, Vulkan |

---

## 8. Feature comparison

### 8.1 Core inference

| Feature | Quarrel | llama.cpp | vLLM | Ollama | TensorRT-LLM |
|---------|---------|-----------|------|--------|---------------|
| Continuous batching | ✅ | ✅ | ✅ | ❌ | ✅ |
| Paged KV cache | ✅ | ✅ | ✅ | ✅ | ✅ |
| Sliding window attention | ✅ | ✅ | ✅ | ✅ | ✅ |
| Prefix caching (PromptCache) | ✅ | ✅ | ✅ | ✅ | ✅ |
| Speculative decoding | ✅ | ✅ | ✅ | ❌ | ✅ |
| Multi-GPU (tensor parallel) | ❌ | ✅ | ✅ | ❌ | ✅ |
| Multi-GPU (pipeline parallel) | ❌ | ✅ | ❌ | ❌ | ✅ |
| Flash attention | ✅ | ✅ | ✅ | ✅ | ✅ |
| Chat template (Jinja2) | ✅ | ✅ | ✅ | ✅ | ✅ |
| LoRA adapter loading | ✅ | ✅ | ✅ | ✅ | ✅ |
| Grammar-constrained sampling | ✅ | ✅ | ✅ | ❌ | ❌ |
| Streaming output | ✅ | ✅ | ✅ | ✅ | ✅ |
| OpenAI-compatible API | ✅ | ✅ | ✅ | ✅ | ✅ |

> **Multi-GPU.** The tensor-parallel and pipeline-parallel managers
> were never used by the inference engine, which loads the whole model
> onto one device, and they have been deleted. `-devices 0,1`, `-tp-size 2` and `-pp-stages 2` exit with an
> error rather than pretending to shard the model. A single device can be
> selected with `-devices N`.

### 8.2 Quantization and memory

| Feature | Quarrel | llama.cpp | vLLM | Ollama | TensorRT-LLM |
|---------|---------|-----------|------|--------|---------------|
| Zero-copy inference (CPU) | ✅ | ✅ | ❌ | ✅ | ❌ |
| KV cache compression (TurboQuant) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Partial GPU offloading | ✅ | ✅ | ❌ | ✅ | ❌ |
| Memory-mapped weights | ✅ | ✅ | ❌ | ✅ | ❌ |
| Heap footprint (4B Q8_0) | <20 MB | <50 MB | N/A (GPU) | <50 MB | N/A (GPU) |

### 8.3 Hardware and platform

| Feature | Quarrel | llama.cpp | vLLM | Ollama | TensorRT-LLM |
|---------|---------|-----------|------|--------|---------------|
| CPU (x86_64) | ✅ AVX-512/AVX2 | ✅ AVX-512/AVX2/NEON | ❌ | ✅ | ❌ |
| CPU (ARM64) | ✅ NEON | ✅ NEON | ❌ | ✅ | ❌ |
| NVIDIA CUDA | ✅ | ✅ | ✅ | ✅ | ✅ |
| Apple Metal | ✅ | ✅ | ❌ | ✅ | ❌ |
| AMD ROCm | ❌ | ✅ | ✅ | ❌ | ❌ |
| Intel Arc | ❌ | ✅ (Vulkan) | ❌ | ❌ | ❌ |
| WebGPU | ❌ | ✅ (demo) | ❌ | ❌ | ❌ |

### 8.4 API and deployment

| Feature | Quarrel | llama.cpp | vLLM | Ollama | TensorRT-LLM |
|---------|---------|-----------|------|--------|---------------|
| REST API (OpenAI) | ✅ | ✅ (server) | ✅ | ✅ | ✅ |
| WebSocket streaming | ✅ | ❌ | ❌ | ❌ | ❌ |
| Arrow Flight RPC | ✅ (server) | ❌ | ❌ | ❌ | ❌ |
| Docker images | ✅ | ✅ (community) | ✅ | ✅ | ✅ |
| Kubernetes Helm chart | ✅ | ❌ | ✅ | ❌ | ✅ |
| Prometheus metrics | ✅ (133 metrics) | ✅ (basic) | ✅ | ❌ | ✅ |
| Grafana dashboard | ✅ | ❌ | ✅ | ❌ | ❌ |
| WebUI | ✅ (Go) | ❌ | ❌ | ✅ | ❌ |

### 8.5 Language and ecosystem

| Feature | Quarrel | llama.cpp | vLLM | Ollama | TensorRT-LLM |
|---------|---------|-----------|------|--------|---------------|
| Implementation language | Go | C/C++ | Python/C++ | Go | C++/Python |
| Build system | Make | CMake | pip/setuptools | Make | CMake |
| CI/CD pipeline | ✅ GitHub Actions | ✅ GitHub Actions | ✅ GitHub Actions | ✅ GitHub Actions | ✅ Jenkins |
| Fuzz testing | ✅ (600K+ execs) | ✅ | ❌ | ❌ | ❌ |
| Security audit (gosec) | ✅ (0 issues) | ❌ | ❌ | ❌ | ❌ |
| Race detector clean | ✅ | N/A (C++) | ❌ | N/A (C++) | N/A (C++) |

---

## 9. Performance comparison

### 9.1 CPU inference (4B Q8_0, single thread)

| Metric | Quarrel | llama.cpp | Notes |
|--------|---------|-----------|-------|
| Generation (tok/s) | ~8-12 | ~10-15 | llama.cpp has more mature SIMD |
| TTFT 512 tokens (ms) | ~80 | ~70 | Similar prefill speed |
| Peak RSS (MB) | <20 | <50 | Quarrel zero-copy advantage |
| Memory usage pattern | Flat | Linear with layers | Quarrel does not dequant to heap |

### 9.2 CPU inference (4B Q8_0, multi-core)

| Metric | Quarrel | llama.cpp | Notes |
|--------|---------|-----------|-------|
| Generation 8 threads (tok/s) | ~25-35 | ~30-40 | llama.cpp thread scaling slightly better |
| Generation 16 threads (tok/s) | ~40-55 | ~45-60 | Both benefit from parallelism |

### 9.3 GPU inference (NVIDIA CUDA, 7B Q4_K)

| Metric | Quarrel | llama.cpp | vLLM |
|--------|---------|-----------|------|
| Generation (tok/s) | ~45 | ~50 | ~80 (continuous batching) |
| VRAM usage (MB) | ~4200 | ~4100 | ~5000 |
| TTFT (ms) | ~30 | ~25 | ~15 |

### 9.4 Apple Silicon Metal (7B Q4_K)

| Metric | Quarrel | llama.cpp | Ollama |
|--------|---------|-----------|--------|
| Generation (tok/s) | ~30 | ~35 | ~33 |
| Unified memory (MB) | ~4100 | ~4000 | ~4200 |

**Key Takeaways:**
- **llama.cpp** leads in raw throughput due to years of SIMD optimization and community tuning
- **Quarrel** leads in memory efficiency (zero-copy) and developer tooling (133 Prometheus metrics, Grafana dashboard, Helm chart)
- **vLLM** leads in GPU throughput via PagedAttention and continuous batching, but requires Python and NVIDIA-only
- **Quarrel** is the only engine with a Go-native implementation, an Arrow Flight server, and WebSocket streaming

---

## What Quarrel does that the others do not

- **Zero-copy inference** — the weights stay memory-mapped and the dot
  products run over them directly, so the heap stays under 20 MB for a 4B
  model.
- **Go-native** — the CPU path has no CGo dependency, and the whole
  engine cross-compiles from one tree.
- **TurboQuant KV compression** — 8× KV cache compression, unique here.
- **133 Prometheus metrics** with a Grafana dashboard over all of them,
  and a Kubernetes Helm chart with health probes and a ServiceMonitor.
- **Fuzzing and a security gate** — `gosec` and `govulncheck` run in CI
  on two build configurations.

## What the others do that Quarrel does not

- **No AMD ROCm** — llama.cpp and vLLM support AMD GPUs.
- **No GPTQ/AWQ** — vLLM supports these popular quantization formats.
- **No Vulkan** — llama.cpp supports Vulkan for Intel/AMD GPUs.
- **No Python bindings** — llama.cpp has a large Python ecosystem.
- **Younger SIMD work** — llama.cpp has more mature AVX-512 and NEON
  kernels.
- **Single-device focus** — the engine loads the whole model onto one
  device; the multi-GPU managers that existed were never used and have
  been deleted.

---

## When to choose Quarrel

| Use Case | Recommended Engine |
|----------|-------------------|
| Memory-constrained deployment | **Quarrel** (zero-copy) |
| Go-native application integration | **Quarrel** |
| Kubernetes production deployment | **Quarrel** (Helm chart) |
| Maximum GPU throughput (NVIDIA) | vLLM |
| Maximum CPU throughput | llama.cpp |
| Broadest hardware support | llama.cpp |
| AMD GPU inference | llama.cpp or vLLM |
| Easiest setup | Ollama |
| Enterprise NVIDIA deployment | TensorRT-LLM |

---

## Future roadmap

| Feature | Quarrel | llama.cpp | vLLM |
|---------|---------|-----------|------|
| AMD ROCm support | Planned | ✅ | ✅ |
| GPTQ/AWQ support | Planned | ❌ | ✅ |
| Python bindings | Planned | ✅ | ✅ |
| Diffusion model support | ❌ | Planned | ❌ |
| Multi-modal (VLM) | ✅ (Metal) | ✅ | ✅ |
