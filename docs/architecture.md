# Longbow-Quarrel Architecture

Comprehensive architectural overview of Longbow-Quarrel (v0.3.0), covering execution pipelines, memory layout, hybrid layer offloading, and multi-backend acceleration.

---

## 1. System Topology

```
  CLI / REST API / WebSocket        the WebUI binary
            │
            ▼
  Universal Model Resolver           the model caches and the local paths
            │
            ▼
  Engine  (internal/engine)          batching, the KV caches, the audits
     ┌──────┴───────┐
     ▼              ▼
  CUDA engine    CPU engine         the same layers, two backends
     │              │
     ▼              ▼
  CUDA device     SIMD kernels      internal/device, internal/simd
```

---

## 2. Memory-Efficient Zero-Copy Quantized Inference

In traditional CPU inference engines, weights are dequantized upfront into `[]float32` slices on the heap. For a 4B parameter model in Q8_0 (4.4 GB on disk), upfront dequantization allocates **17.9 GB of heap memory**, leading to swapping and Out-Of-Memory (OOM) failures.

Longbow-Quarrel eliminates this overhead via:

1. **Zero-Copy Matrix-Vector Multiplication:** the dot product of an
   activation and a weight is evaluated directly over the quantized bytes
   of the memory-mapped file — `gguf.MatVecMulQ8_0`, `MatVecMulQ4_K`,
   `MatVecMulQ6_K` and the rest; the list of the formats is in
   [features.md](features.md#2-quantization-and-formats). No floating-point
   weight buffer is allocated on the Go heap.
2. **On-Demand Token Embedding Lookup:**
   - Reads only the specific token's embedding vector (e.g. 2,560 elements = 10 KB) directly from mmap rather than materializing the full vocabulary embedding table (150,000 × 2,560 × 4 bytes = 1.5 GB).
3. **Outcome:**
   - Heap memory usage for a 4B parameter model drops from **17.9 GB to < 20 MB** (>99.9% reduction).

---

## 3. Partial GPU Layer Offloading Architecture

When a model exceeds available GPU VRAM, Longbow-Quarrel splits transformer layers dynamically:

```
  token → GPU embedding
              │
              ▼
  layers 0 .. numGPULayers-1        on the GPU
              │  activation down
              ▼
  layers numGPULayers .. N-1         on the CPU
              │  activation up
              ▼
  output norm, output head, sampler  on the GPU
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

## 5. PromptCache & Chat Template Pipeline

Longbow-Quarrel implements a two-stage prompt preparation pipeline:

1. **Radix-Tree Prefix Caching (`prompt_cache.go`):**
   - Tokenized prompt segments are stored in a radix tree keyed by token ID sequences. Common prefixes (system prompts, repeated conversation turns) are cached so that only the delta of new tokens requires tokenization and prompt construction on subsequent requests.
   - Cache entries are evicted LRU when total cached token count exceeds the configurable budget. Subtrees are shared by reference, allowing zero-copy prefix reuse across concurrent requests.

2. **Jinja2 Chat Template Rendering (`prompt_wrapper.go`):**
   - Model chat templates are parsed and rendered using a lightweight Jinja2-compatible template engine. This replaces the legacy switch-case approach, enabling per-model template fidelity without hardcoding.
   - Template output (system prompt, user/assistant turn markers, stop tokens) is tokenized and fed into the prefix cache, completing the pipeline from raw chat messages to a cache-optimized token sequence.

---

## 6. SIMD Vector Acceleration (`internal/simd/`)

| Vector ISA | Optimized Kernels | Strategy |
|------------|-------------------|----------|
| **AVX-512** | TurboQuant Step 3 Inverse Rotation, QJL Transform, Softmax | 16-lane `_mm512_fmadd_ps` with row-major linear combinations |
| **AVX2** | TurboQuant Inverse Rotation, SwiGLU, RMSNorm | 8-lane `_mm256_fmadd_ps` with contiguous memory traversal |
| **ARM NEON** | TurboQuant Inverse Rotation, PolarQuant | 4-lane `vfmaq_f32` with contiguous row-major loads |
| **Generic** | Portable pure Go fallbacks | Exact mathematical parity with SIMD implementations |
| **Generic** | MatMul (VecDotF32 inner loop) | B transpose + VecDotF32 for cache-friendly column access |

---

## 7. Pluggable Math Backend (`internal/mathx/`)

The engine calls `math.*` ~700 times across ~100 files. To evaluate
[`emlgo`](https://github.com/23skdu/emlgo) as a drop-in math backend without
forking the tree, all math calls are routed through `internal/mathx/`, which has
two build-tagged implementations:

| File | Build tag | Backend |
|------|-----------|---------|
| `internal/mathx/math_stdlib.go` | `!emlgo` (default) | stdlib `math` |
| `internal/mathx/math_emlgo.go` | `emlgo` | `emlgo` scalar `pkg/*` (bit-exact) |
| `internal/mathx/math_emlgo.go` | `emlgo,emlfast` | `emlgo` `pkg/fastmath` (minimax approx) |

- The default build is **unchanged** (`!emlgo`); `emlgo` is compiled in only with
  `-tags emlgo`.
- emlgo's scalar functions delegate to stdlib `math` (bit-exact); `pkg/fastmath`
  uses minimax polynomial approximations (not bit-exact) and is gated separately
  behind `emlfast` so numerical drift can be isolated from SIMD batching gains.
- `IsNaN`, `IsInf`, `NaN`, `Inf`, `*bits`/`frombits`, `Copysign`, `MaxInt`,
  `MaxFloat32/64`, `SmallestNonzeroFloat32`, and `Pi` stay on stdlib — emlgo has
  no equivalent.
- **Prerequisite:** emlgo's `go.mod` declares `module github.com/emlgo/eml` while
  the repo is hosted at `github.com/23skdu/emlgo`; a fork with a corrected module
  path is required. See [dependencies.md](dependencies.md) and
  [roadmap.md](roadmap.md#planned-features).

---

## 8. Directory Layout

```
cmd/          the engine, the CLI, the WebUI, the tools
internal/
  api/         the HTTP listeners and the Arrow Flight server
  config/      the model and runtime configuration
  cpu/         the CPU feature detection
  device/      the backends: CUDA, Metal, CPU
  engine/      the inference engine, the KV caches, the batching
  gguf/        the GGUF reader and writer, the zero-copy dequant
  logger/      the structured logger
  mathx/       the pluggable math backend
  metrics/     the Prometheus instrumentation
  models/      the model resolver
  ollama/      the Ollama manifest store
  promtest/    the metric assertions the tests use
  sampler/     the samplers and the grammar constraints
  simd/        the SIMD kernels
  telemetry/   the OpenTelemetry spans
  testmodel/   the model builder the tests use
  tokenizer/   the BPE and SentencePiece tokenizers
  vector/      the vector search package
  vlm/         the vision pipeline
scripts/      the gates, the benchmarks and the drift checker
```

## 8.1 Element Sizes

The two GPU-free backends do not share an element-size convention, and
mixing them up was the source of a long series of defects:

| Quantity | CPU | CUDA | Metal |
|----------|-----|------|-------|
| The weights | float32 | float16 | float16 |
| The activations | float32 | float32 | float16 |
| The paged KV pool | float32 | float16 | float16 |
| The positions and the block tables | float32 | float32 | float32 |

The rule is that the *weights* are the only thing stored in half
precision, and everything the kernels compute with is in the element size
the kernels use. The device layer converts at the boundary: a float32
activation is converted to half precision for the matrix multiplication,
because cuBLAS cannot multiply a float32 activation by a half-precision
weight.

## 9. LoRA Adapter Architecture

Longbow-Quarrel supports Low-Rank Adaptation (LoRA) adapters with backend-specific load strategies:

- **CPU (`lora.go`):** Adapters are merged on load — the low-rank weight matrices are decompressed and fused into the base model's weight tensors at initialization time. This avoids runtime dispatch overhead and integrates seamlessly with the zero-copy matvec pipeline, since the merged weights are written back into the memory-mapped base model region.

- **Metal / CUDA:** Adapters are kept as separate weight tensors and dispatched at runtime. During forward pass, the LoRA projection (A→down, B→up) is executed as a fused matmul on the accelerator, and the output is added to the base model activation. This avoids re-mapping the full base model and enables hot-swapping adapters without reloading the base weights.

Adapter metadata (rank, alpha, target modules) is read from the GGUF file's `lora_adapter` metadata keys and validated against the base model's layer structure at load time.
