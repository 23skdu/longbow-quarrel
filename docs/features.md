# Longbow-Quarrel Features (v0.3.0)

> **Version scope.** This document describes release **v0.3.0**, the latest
> git tag. `main` carries additional unreleased work; see
> [roadmap.md](roadmap.md) for what has landed and what is planned.

A high-performance, memory-efficient LLM inference engine written in Go with native GPU acceleration and advanced SIMD vectorization.

---

## 1. Inference Engine

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

## 2. Quantization & Formats

| Format | Quantization | Dequantization | Zero-Copy MatVec | Notes |
|--------|--------------|----------------|------------------|-------|
| **Q8_0** | ✅ | ✅ | ✅ (`MatVecMulQ8_0`) | Primary zero-copy quantized format |
| **Q4_K** | ✅ | ✅ | ✅ (`MatVecMulQ4_K`) | SIMD unrolled batch dequant & zero-copy matvec |
| **Q6_K** | ✅ | ✅ | ✅ (`MatVecMulQ6_K`) | SIMD unrolled batch dequant & zero-copy matvec |
| **Q2_K** | - | ✅ | ✅ (`MatVecMulQ2_K`) | via matVecMulGeneric |
| **Q3_K** | - | ✅ | ✅ (`MatVecMulQ3_K`) | via matVecMulGeneric |
| **Q4_0** | - | ✅ | ✅ (`MatVecMulQ4_0`) | via matVecMulGeneric |
| **Q5_0** | - | ✅ | ✅ (`MatVecMulQ5_0`) | via matVecMulGeneric |
| **Q5_K** | - | ✅ | ✅ (`MatVecMulQ5_K`) | via matVecMulGeneric |
| **BF16** | ✅ | ✅ | ✅ (`MatVecMulBF16`) | BFloat16 — zero-copy parallel |
| **FP16 / FP32** | ✅ | ✅ | ✅ (`MatVecMul`) | Native floating point precision |
| **FP8 (E4M3 / E5M2)** | ✅ | ✅ | - | NVIDIA H100 Hopper Tensor Core format |
| **TurboQuant** | ✅ | ✅ | - | 8x KV cache compression (PolarQuant + QJL) |

---

## 3. Hardware Acceleration & Backends

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

## 4. KV Cache Management

- **Contiguous Cache** — Traditional sequential buffer allocation.
- **Paged Cache** — Virtual memory page table allocation for efficient memory use and fragmented requests.
- **Sliding Window Attention** — Fixed-window attention for long contexts (Mistral 4096, Gemma 4 hybrid).
- **TurboQuant Compression** — 8x KV cache compression with PolarQuant and QJL residual matrices.
- **Per-Sequence Tracking** — Independent cache position and sequence rollback support (`RollbackKV`).
- **PromptCache Integration** — Radix-tree prefix caching for repeated/shared prompts. Inserted after CompleteSequence in all engines.

---

## 5. Observability & Telemetry

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

## 6. Testing & Quality Assurance

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
