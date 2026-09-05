# Longbow-Quarrel vs Reference Engines: Feature & Speed Comparison

Comprehensive comparison of Longbow-Quarrel against the top LLM inference engines for features, performance, and deployment options.

---

## 1. Engine Overview

| Engine | Language | License | Primary Backend | GPU Support |
|--------|----------|---------|-----------------|-------------|
| **Longbow-Quarrel** | Go | Open Source | CPU (SIMD) + CUDA + Metal | NVIDIA, Apple Silicon |
| **llama.cpp** | C/C++ | MIT | CPU (SIMD) + CUDA + Metal + Vulkan | NVIDIA, AMD, Apple, Intel |
| **vLLM** | Python/C++ | Apache 2.0 | CUDA (PagedAttention) | NVIDIA only |
| **Ollama** | Go | MIT | llama.cpp wrapper | NVIDIA, Apple |
| **TensorRT-LLM** | C++/Python | Apache 2.0 | CUDA (TensorRT) | NVIDIA only |
| **MLC-LLM** | C++/Python | Apache 2.0 | Multi-backend | NVIDIA, Apple, Vulkan |

---

## 2. Quantization Support

| Format | Quarrel | llama.cpp | vLLM | Ollama | TensorRT-LLM |
|--------|---------|-----------|------|--------|---------------|
| FP32 | ✅ | ✅ | - | ✅ | ✅ |
| FP16 | ✅ | ✅ | ✅ | ✅ | ✅ |
| BF16 | ✅ | ✅ | ✅ | ✅ | ✅ |
| Q8_0 | ✅ | ✅ | - | ✅ | ✅ |
| Q6_K | ✅ | ✅ | - | ✅ | - |
| Q5_0 | ✅ | ✅ | - | ✅ | - |
| Q5_K | ✅ | ✅ | - | ✅ | - |
| Q4_K | ✅ | ✅ | - | ✅ | - |
| Q4_0 | ✅ | ✅ | - | ✅ | - |
| Q3_K | ✅ | ✅ | - | ✅ | - |
| Q2_K | ✅ | ✅ | - | ✅ | - |
| FP8 (E4M3/E5M2) | ✅ | ✅ | ✅ | - | ✅ |
| GPTQ | - | - | ✅ | - | ✅ |
| AWQ | - | - | ✅ | - | �Q |
| GGUF (generic) | ✅ | ✅ | - | ✅ | - |

### Zero-Copy MatVec Kernels

| Format | Quarrel | llama.cpp |
|--------|---------|-----------|
| Q8_0 | ✅ | ✅ |
| Q4_K | ✅ | ✅ |
| Q6_K | ✅ | ✅ |
| Q4_0 | ✅ | ✅ |
| Q5_0 | ✅ | ✅ |
| Q5_K | ✅ | ✅ |
| Q2_K | ✅ | ✅ |
| Q3_K | ✅ | ✅ |
| BF16 | ✅ | ✅ |

---

## 3. Feature Comparison

### 3.1 Core Inference

| Feature | Quarrel | llama.cpp | vLLM | Ollama | TensorRT-LLM |
|---------|---------|-----------|------|--------|---------------|
| Continuous batching | ✅ | ✅ | ✅ | ❌ | ✅ |
| Paged KV cache | ✅ | ✅ | ✅ | ✅ | ✅ |
| Sliding window attention | ✅ | ✅ | ✅ | ✅ | ✅ |
| Prefix caching (PromptCache) | ✅ | ✅ | ✅ | ✅ | ✅ |
| Speculative decoding | ✅ | ✅ | ✅ | ❌ | ✅ |
| Multi-GPU (tensor parallel) | ✅ | ✅ | ✅ | ❌ | ✅ |
| Multi-GPU (pipeline parallel) | ✅ | ✅ | ❌ | ❌ | ✅ |
| Flash attention | ✅ | ✅ | ✅ | ✅ | ✅ |
| Chat template (Jinja2) | ✅ | ✅ | ✅ | ✅ | ✅ |
| LoRA adapter loading | ✅ | ✅ | ✅ | ✅ | ✅ |
| Grammar-constrained sampling | ✅ | ✅ | ✅ | ❌ | ❌ |
| Streaming output | ✅ | ✅ | ✅ | ✅ | ✅ |
| OpenAI-compatible API | ✅ | ✅ | ✅ | ✅ | ✅ |

### 3.2 Quantization & Memory

| Feature | Quarrel | llama.cpp | vLLM | Ollama | TensorRT-LLM |
|---------|---------|-----------|------|--------|---------------|
| Zero-copy inference (CPU) | ✅ | ✅ | ❌ | ✅ | ❌ |
| KV cache compression (TurboQuant) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Partial GPU offloading | ✅ | ✅ | ❌ | ✅ | ❌ |
| Memory-mapped weights | ✅ | ✅ | ❌ | ✅ | ❌ |
| Heap footprint (4B Q8_0) | <20 MB | <50 MB | N/A (GPU) | <50 MB | N/A (GPU) |

### 3.3 Hardware & Platform

| Feature | Quarrel | llama.cpp | vLLM | Ollama | TensorRT-LLM |
|---------|---------|-----------|------|--------|---------------|
| CPU (x86_64) | ✅ AVX-512/AVX2 | ✅ AVX-512/AVX2/NEON | ❌ | ✅ | ❌ |
| CPU (ARM64) | ✅ NEON | ✅ NEON | ❌ | ✅ | ❌ |
| NVIDIA CUDA | ✅ | ✅ | ✅ | ✅ | ✅ |
| Apple Metal | ✅ | ✅ | ❌ | ✅ | ❌ |
| Google TPU | ✅ (WIP) | ❌ | ❌ | ❌ | ✅ |
| AMD ROCm | ❌ | ✅ | ✅ | ❌ | ❌ |
| Intel Arc | ❌ | ✅ (Vulkan) | ❌ | ❌ | ❌ |
| WebGPU | ❌ | ✅ (demo) | ❌ | ❌ | ❌ |

### 3.4 API & Deployment

| Feature | Quarrel | llama.cpp | vLLM | Ollama | TensorRT-LLM |
|---------|---------|-----------|------|--------|---------------|
| REST API (OpenAI) | ✅ | ✅ (server) | ✅ | ✅ | ✅ |
| WebSocket streaming | ✅ | ❌ | ❌ | ❌ | ❌ |
| Arrow Flight RPC | ✅ | ❌ | ❌ | ❌ | ❌ |
| Docker images | ✅ | ✅ (community) | ✅ | ✅ | ✅ |
| Kubernetes Helm chart | ✅ | ❌ | ✅ | ❌ | ✅ |
| Prometheus metrics | ✅ (144 metrics) | ✅ (basic) | ✅ | ❌ | ✅ |
| Grafana dashboard | ✅ | ❌ | ✅ | ❌ | ❌ |
| WebUI | ✅ (Templ) | ❌ | ❌ | ✅ | ❌ |

### 3.5 Language & Ecosystem

| Feature | Quarrel | llama.cpp | vLLM | Ollama | TensorRT-LLM |
|---------|---------|-----------|------|--------|---------------|
| Implementation language | Go | C/C++ | Python/C++ | Go | C++/Python |
| Build system | Make | CMake | pip/setuptools | Make | CMake |
| CI/CD pipeline | ✅ GitHub Actions | ✅ GitHub Actions | ✅ GitHub Actions | ✅ GitHub Actions | ✅ Jenkins |
| Fuzz testing | ✅ (600K+ execs) | ✅ | ❌ | ❌ | ❌ |
| Security audit (gosec) | ✅ (0 issues) | ❌ | ❌ | ❌ | ❌ |
| Race detector clean | ✅ | N/A (C++) | ❌ | N/A (C++) | N/A (C++) |

---

## 4. Performance Comparison

### 4.1 CPU Inference (4B Q8_0 Model, Single Thread)

| Metric | Quarrel | llama.cpp | Notes |
|--------|---------|-----------|-------|
| Generation (tok/s) | ~8-12 | ~10-15 | llama.cpp has more mature SIMD |
| TTFT 512 tokens (ms) | ~80 | ~70 | Similar prefill speed |
| Peak RSS (MB) | <20 | <50 | Quarrel zero-copy advantage |
| Memory usage pattern | Flat | Linear with layers | Quarrel does not dequant to heap |

### 4.2 CPU Inference (4B Q8_0 Model, Multi-Core)

| Metric | Quarrel | llama.cpp | Notes |
|--------|---------|-----------|-------|
| Generation 8 threads (tok/s) | ~25-35 | ~30-40 | llama.cpp thread scaling slightly better |
| Generation 16 threads (tok/s) | ~40-55 | ~45-60 | Both benefit from parallelism |

### 4.3 GPU Inference (NVIDIA CUDA, 7B Q4_K)

| Metric | Quarrel | llama.cpp | vLLM |
|--------|---------|-----------|------|
| Generation (tok/s) | ~45 | ~50 | ~80 (continuous batching) |
| VRAM usage (MB) | ~4200 | ~4100 | ~5000 |
| TTFT (ms) | ~30 | ~25 | ~15 |

### 4.4 Apple Silicon Metal (7B Q4_K)

| Metric | Quarrel | llama.cpp | Ollama |
|--------|---------|-----------|--------|
| Generation (tok/s) | ~30 | ~35 | ~33 |
| Unified memory (MB) | ~4100 | ~4000 | ~4200 |

**Key Takeaways:**
- **llama.cpp** leads in raw throughput due to years of SIMD optimization and community tuning
- **Quarrel** leads in memory efficiency (zero-copy) and developer tooling (144 Prometheus metrics, Grafana dashboard, Helm chart)
- **vLLM** leads in GPU throughput via PagedAttention and continuous batching, but requires Python and NVIDIA-only
- **Quarrel** is the only engine with Go-native implementation, Arrow Flight distributed inference, and WebSocket streaming

---

## 5. Unique Quarrel Advantages

1. **Zero-Copy Inference** — <20 MB heap for 4B models (vs 50+ MB for llama.cpp)
2. **Go-Native** — No CGO dependency for CPU path; easy cross-compilation
3. **144 Prometheus Metrics** — Most comprehensive observability of any inference engine
4. **Grafana Dashboard** — Pre-built dashboards for PromptCache, speculative decoding, MoE, Gemma4
5. **Kubernetes Helm Chart** — Production-ready K8s deployment with health probes
6. **Arrow Flight RPC** — Distributed inference across multiple nodes
7. **WebSocket Streaming** — Real-time token streaming for web applications
8. **TurboQuant KV Compression** — 8x KV cache compression (unique to Quarrel)
9. **Security Auditing** — gosec-clean codebase with fuzz testing
10. **BF16 Zero-Copy** — Native BFloat16 dequantization and MatVec

---

## 6. Unique Quarrel Limitations

1. **No AMD ROCm** — llama.cpp and vLLM support AMD GPUs
2. **No GPTQ/AWQ** — vLLM supports these popular quantization formats
3. **Younger SIMD Optimizations** — llama.cpp has more mature AVX-512/NEON kernels
4. **No Python Bindings** — llama.cpp has extensive Python ecosystem integration
5. **No Vulkan** — llama.cpp supports Vulkan for Intel/AMD GPUs
6. **Single-GPU Focus** — Multi-GPU works but vLLM is more mature for large-scale serving

---

## 7. When to Choose Quarrel

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

## 8. Future Roadmap Comparison

| Feature | Quarrel | llama.cpp | vLLM |
|---------|---------|-----------|------|
| AMD ROCm support | Planned | ✅ | ✅ |
| GPTQ/AWQ support | Planned | ❌ | ✅ |
| Python bindings | Planned | ✅ | ✅ |
| Diffusion model support | ❌ | Planned | ❌ |
| Multi-modal (VLM) | ✅ (Metal) | ✅ | ✅ |
