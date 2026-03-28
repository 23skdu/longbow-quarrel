# Longbow-Quarrel vs vLLM Comparison

## Executive Summary

This document compares Longbow-Quarrel (a Go-based LLM inference engine) with vLLM (a Python-based high-performance LLM serving library). The comparison covers architecture, features, performance, and development priorities.

---

## Architecture Overview

### Longbow-Quarrel

- **Language:** Go with native GPU kernels (Metal/CUDA)
- **Architecture:** Monolithic Go application with embedded GPU kernels
- **GPU Backends:** Metal (Apple Silicon), CUDA (NVIDIA), CPU fallback
- **Model Format:** GGUF (llama.cpp format)
- **Deployment:** Docker, Kubernetes, native binaries

### vLLM

- **Language:** Python with PyTorch backend
- **Architecture:** Modular Python application with C++/CUDA kernels
- **GPU Backends:** CUDA (NVIDIA), ROCm (AMD), CPU (limited)
- **Model Format:** Hugging Face Safetensors, GGML
- **Deployment:** Docker, Kubernetes, cloud services

---

## Feature Comparison Matrix

| Feature | Longbow-Quarrel | vLLM |
|---------|-----------------|------|
| **Model Loading** | | |
| GGUF Format | ✅ Native | ⚠️ Via loader |
| Safetensors | ❌ Not supported | ✅ Native |
| GGML/GGUF Legacy | ✅ Native | ⚠️ Limited |
| **Quantization** | | |
| FP16/FP32 | ✅ | ✅ |
| INT8 | ✅ Q8_0 | ✅ |
| INT4 | ✅ Q4_K | ✅ AWQ/GPTQ |
| FP8 (Hopper) | ✅ E4M3/E5M2 | ✅ |
| **KV Cache** | | |
| Contiguous | ✅ | ✅ |
| Paged Attention | ✅ | ✅ (Pioneered) |
| Sliding Window | ✅ | ✅ |
| Prefix Caching | ⚠️ Shared cache | ✅ Hash-based |
| **Batching** | | |
| Static Batching | ✅ | ✅ |
| Continuous Batching | ⚠️ Basic | ✅ Advanced |
| Chunked Prefill | ❌ | ✅ |
| **Attention** | | |
| Flash Attention | ⚠️ CUDA only | ✅ |
| FlashInfer | ❌ | ✅ |
| Mamba/SSM | ✅ Experimental | ⚠️ Jamba limited |
| **Distributed** | | |
| Tensor Parallelism | ✅ Implemented | ✅ |
| Pipeline Parallelism | ✅ Implemented | ⚠️ V1 limited |
| Data Parallelism | ❌ | ✅ |
| Multi-LoRA | ❌ | ⚠️ V1 coming |
| **Advanced** | | |
| Speculative Decoding | ❌ | ✅ |
| Structured Output | ❌ | ✅ |
| Embedding Models | ❌ | ✅ |
| Multimodal | ❌ | ✅ VLM support |
| **API** | | |
| OpenAI Compatible | ✅ Basic | ✅ Full |
| WebSocket Streaming | ✅ | ❌ |
| gRPC | ❌ | ❌ |
| **Observability** | | |
| Prometheus Metrics | ✅ | ⚠️ V1 limited |
| Tracing | ⚠️ Activation tracing | ❌ |
| **Platforms** | | |
| NVIDIA GPU | ✅ CUDA | ✅ |
| AMD GPU | ❌ | ✅ ROCm |
| Apple Silicon | ✅ Metal | ❌ |
| Intel/Apple CPU | ✅ | ❌ |

---

## Performance Comparison

### Current Quarrel Performance (M3 Pro)

| Model | Quarrel | llama.cpp | Gap |
|-------|---------|-----------|-----|
| Smollm2 135M | 38.81 t/s | ~45 t/s | -14% |
| Granite 4B | 12.53 t/s | 45.38 t/s | -3.6x |
| Mistral 7B | 1.91 t/s | 25.29 t/s | -13.2x |

### vLLM Performance Targets (H100)

- Llama 3.1 8B: Up to 1.7x faster than vLLM V0
- Llama 3.3 70B: State-of-the-art throughput
- Prefix caching: <1% overhead when cache miss

### Performance Gap Analysis

**Quarrel disadvantages:**
1. **Metal synchronization overhead** - Excessive CPU/GPU sync calls
2. **Kernel batching** - Less aggressive than llama.cpp
3. **Python bindings** - No direct PyTorch integration
4. **Continuous batching** - Basic implementation vs vLLM's advanced scheduler

**Quarrel advantages:**
1. **Go runtime** - Lower memory overhead, better concurrency
2. **Native Metal** - No Python interpreter on Apple Silicon
3. **GGUF native** - Direct model parsing without conversion

---

## vLLM V1 Key Innovations (Jan 2025)

vLLM V1 introduced several architectural improvements that set new benchmarks:

1. **EngineCore** - Isolated execution loop with multiprocessing
2. **Persistent Batch** - Cached input tensors, minimal per-step recreation
3. **Zero-overhead Prefix Caching** - Hash-based with O(1) eviction
4. **Piecewise CUDA Graphs** - Overcomes traditional CUDA graph limitations
5. **FlashAttention 3 Integration** - Flexible attention for dynamic batching
6. **Multimodal Optimizations** - Async preprocessing, encoder cache

---

## Feature Gap Summary

### Critical (Blocking Production)

| Feature | Status | Priority |
|---------|--------|----------|
| Advanced Continuous Batching | Basic impl | P0 |
| Prefix Caching (hash-based) | Shared cache only | P0 |
| FlashAttention (Metal) | ❌ | P0 |
| Safetensors Support | ❌ | P0 |

### High Priority

| Feature | Status | Priority |
|---------|--------|----------|
| Chunked Prefill | ❌ | P1 |
| Speculative Decoding | ❌ | P1 |
| Structured Output | ❌ | P1 |
| Full OpenAI Compatibility | Basic | P1 |

### Medium Priority

| Feature | Status | Priority |
|---------|--------|----------|
| Embedding Models | ❌ | P2 |
| Multi-LoRA | ❌ | P2 |
| vLLM Python Integration | Export only | P2 |

### Low Priority

| Feature | Status | Priority |
|---------|--------|----------|
| Multimodal (VLM) | ❌ | P3 |
| AMD ROCm Support | ❌ | P3 |
| TPU Support | ❌ | P3 |

---

## Benchmarking Notes

To run proper comparisons:

```bash
# Quarrel benchmarks
go build -tags "darwin,metal" -o bin/metal_benchmark ./cmd/metal_benchmark
./bin/metal_benchmark -model <path> -tokens 100

# vLLM benchmarks
vllm serve <model> --dtype float16
# Use https://github.com/vllm-project/llmperf or similar
```

**Test Conditions:**
- Match hardware (M3 Pro vs equivalent NVIDIA)
- Same model quantization
- Same context length
- Same batch size
- Warm-up runs before measurement

---

## References

- [vLLM V1 Release](https://vllm-project.github.io/2025/01/27/v1-alpha-release.html)
- [vLLM Features](https://docs.vllm.ai/en/stable/features/)
- [PagedAttention Paper](https://blog.vllm.ai/2023/06/20/vllm.html)
- [Quarrel Features](./features.md)
- [Quarrel Performance](./performance.md)
