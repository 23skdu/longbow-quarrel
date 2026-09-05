# Longbow-Quarrel Architecture

Comprehensive architectural overview of Longbow-Quarrel (v0.2.0), covering execution pipelines, memory layout, hybrid layer offloading, and multi-backend acceleration.

---

## 1. System Topology

```
┌─────────────────────────────────────────────────────────────────┐
│                      Client Interfaces                          │
│         CLI (cmd/simple, cmd/quarrel)  |  Templ WebUI           │
│         OpenAI-Compatible REST API    |  WebSocket Stream       │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────┐
│                   Universal Model Resolver                      │
│      ~/.cache/llmfit/models/ | ~/.cache/llama.cpp/             │
│      ~/.cache/huggingface/   | ~/.ollama/models/               │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────┐
│                        Engine Layer                             │
│                  (internal/engine/interface.go)                 │
├────────────────────────────────┬────────────────────────────────┤
│          CUDA Engine           │           CPU Engine           │
│     (engine_cuda.go)           │     (engine_cpu.go)            │
│  - Fused GPU kernels           │  - Zero-copy quantized mmap    │
│  - Partial layer offloading    │  - SIMD batch dequantization   │
│  - KV Cache on VRAM            │  - ApplyLayerCPU pipeline      │
└───────────────┬────────────────┴────────────────┬───────────────┘
                │                                 │
                ▼                                 ▼
┌────────────────────────────────┐ ┌──────────────────────────────┐
│        NVIDIA CUDA GPU         │ │       Host CPU SIMD          │
│   (internal/device/cuda.go)    │ │   (internal/simd/)           │
│ - Tensor Cores / cuDNN         │ │ - AVX-512 / AVX2 / ARM NEON  │
│ - Flash Attention              │ │ - MatVecMul Q8_0, Q4_K, Q6_K │
│ - libcuda_kernels.a            │ │ - TurboQuant (Polar + QJL)   │
└────────────────────────────────┘ └──────────────────────────────┘
```

---

## 2. Memory-Efficient Zero-Copy Quantized Inference

In traditional CPU inference engines, weights are dequantized upfront into `[]float32` slices on the heap. For a 4B parameter model in Q8_0 (4.4 GB on disk), upfront dequantization allocates **17.9 GB of heap memory**, leading to swapping and Out-Of-Memory (OOM) failures.

Longbow-Quarrel eliminates this overhead via:

1. **Zero-Copy Matrix-Vector Multiplication:**
   - Evaluates direct dot products between activation vectors and raw quantized bytes directly from the memory-mapped file:
     - `gguf.MatVecMulQ8_0(data, vector, rows, cols)`
     - `gguf.MatVecMulQ4_K(data, vector, rows, cols)`
     - `gguf.MatVecMulQ6_K(data, vector, rows, cols)`
   - No floating-point weight buffers are allocated on the Go heap.
2. **On-Demand Token Embedding Lookup:**
   - Reads only the specific token's embedding vector (e.g. 2,560 elements = 10 KB) directly from mmap rather than materializing the full vocabulary embedding table (150,000 × 2,560 × 4 bytes = 1.5 GB).
3. **Outcome:**
   - Heap memory usage for a 4B parameter model drops from **17.9 GB to < 20 MB** (>99.9% reduction).

---

## 3. Partial GPU Layer Offloading Architecture

When a model exceeds available GPU VRAM, Longbow-Quarrel splits transformer layers dynamically:

```
Input Token → GPU Embedding
                 │
                 ▼
[Layers 0 .. numGPULayers - 1]  ───────► Runs on GPU VRAM
                 │
                 │ Activation Download (hidden.ToHostF32())
                 ▼
[Layers numGPULayers .. N - 1]  ───────► Runs on CPU RAM (ApplyLayerCPU)
                 │
                 │ Activation Upload (hidden.LoadFrom())
                 ▼
GPU Final RMSNorm & Output Head ───────► Runs on GPU
                 │
                 ▼
              Sampler
```

### Key Components:
- **`internal/engine/cpu_weights.go`**: Neutral layer execution pipeline shared between CPU and CUDA builds. Manages quantized and FP32 weight references, SwiGLU FFN projections, and hybrid attention.
- **`internal/device/cuda.go` (`NewCUDAModel`)**: Allocates GPU layer weights and KV cache positions only for layers $0 \le L < \text{numGPULayers}$, bounding VRAM footprint.
- **`internal/engine/engine_cuda.go`**: Manages host-device activation roundtrips and records latency histograms and transfer counters in Prometheus.

---

## 4. Multi-Architecture Dynamic Metadata Extraction

Quarrel dynamically parses model metadata across model families without hardcoded architecture keys:

- **Supported Architecture Identifiers:** `qwen35`, `qwen2`, `llama`, `mistral`, `gemma4`, `phi3`, `smollm2`, `granite`.
- **Hybrid Attention Handling (Qwen 3.5):**
  - Detects hybrid linear/full attention architectures (e.g. 32 layers where layers $L \equiv 3 \pmod 4$ use Full Self-Attention with per-head Q/K RMSNorm, and all other layers use GatedDeltaNet SSM linear projections).
  - Routes execution dynamically between `attentionCPU` and `ssm_conv1d`/`ssm_out`.

---

## 5. SIMD Vector Acceleration (`internal/simd/`)

| Vector ISA | Optimized Kernels | Strategy |
|------------|-------------------|----------|
| **AVX-512** | TurboQuant Step 3 Inverse Rotation, QJL Transform, Softmax | 16-lane `_mm512_fmadd_ps` with row-major linear combinations |
| **AVX2** | TurboQuant Inverse Rotation, SwiGLU, RMSNorm | 8-lane `_mm256_fmadd_ps` with contiguous memory traversal |
| **ARM NEON** | TurboQuant Inverse Rotation, PolarQuant | 4-lane `vfmaq_f32` with contiguous row-major loads |
| **Generic** | Portable pure Go fallbacks | Exact mathematical parity with SIMD implementations |

---

## 6. Directory Layout

```
internal/
  api/             # HTTP & Arrow Flight RPC servers
  config/          # Model and runtime configurations
  device/          # Hardware abstractions (CUDA, Metal, CPU)
  engine/
    cpu_weights.go # Neutral CPU layer pipeline & zero-copy matvec dispatch
    engine_cpu.go  # Pure CPU inference engine
    engine_cuda.go # CUDA engine with hybrid layer offloading
    interface.go   # Core engine interface contracts
    kv_cache.go    # Contiguous & sliding window KV cache
    kv_cache_paged.go # Paged virtual memory KV cache
  gguf/
    dequant.go     # Scalar reference dequantization & MatVecMulQ8_0
    dequant_simd.go# SIMD batch dequant & zero-copy MatVecMulQ4_K, MatVecMulQ6_K
    reader.go      # Binary GGUF parser & tensor mmap indexer
  models/
    resolver.go    # Universal multi-engine cache resolver
  simd/
    turboquant_avx512.c # AVX-512 TurboQuant kernel
    turboquant_avx2.c   # AVX2 TurboQuant kernel
    turboquant_neon.c   # ARM NEON TurboQuant kernel
  tokenizer/       # Fast BPE & SentencePiece tokenizer
```