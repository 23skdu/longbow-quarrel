# Longbow-Quarrel - Next Steps & Roadmap

## 10-Part Architectural & Performance Improvement Plan (v0.4.0)

Based on recent architectural advances (coherence parity for Qwen 3.5 GatedDeltaNet SSM and Gemma 4, unified CPU/CUDA hybrid layer offloading, and memory reclamation), the following 10-part improvement plan outlines the next major milestones for performance, scalability, and multi-modal support:

### Part 1: Native CUDA Quantized Matrix Multiplication (Zero-Dequant GEMM)
- **Goal**: Eliminate host FP16 dequantization during model load by evaluating quantized weights (`Q4_K`, `Q8_0`, `Q6_K`, `Q2_K`, `Q3_K`) directly in CUDA kernels.
- **Implementation**:
  - Load raw quantized weight bytes directly into GPU VRAM (saving 50–70% VRAM and dropping load times from ~50s to <1s).
  - Implement warp-level dequantizing GEMM kernels that unpack quantized nibbles/bytes into shared memory/registers during dot product accumulation.
- **Files**: `internal/device/cuda.go`, `internal/device/cuda_kernels.cu`

### Part 2: CUDA Prefill Flash Attention with Sliding Window Support
- **Goal**: Accelerate multi-token prompt prefill on GPU by $5\times-10\times$.
- **Implementation**:
  - Replace sequential token decode loop with tiled FlashAttention-2 prefill kernel using online softmax in CUDA shared memory ($O(N)$ memory complexity).
  - Add native sliding-window masking inside the attention kernel for Gemma 4 (512-token local window) and Mistral architectures.
- **Files**: `internal/device/cuda_kernels.cu`, `internal/engine/engine_cuda.go`

### Part 3: Continuous Batching & Paged Attention Scheduling
- **Goal**: High-throughput multi-request serving with dynamic iteration-level scheduling.
- **Implementation**:
  - Implement vLLM-style iteration-level scheduler multiplexing prefill and decode requests in dynamic continuous batches.
  - Integrate paged block tables directly with GPU KV cache memory blocks to eliminate internal fragmentation and support dynamic request preemption.
- **Files**: `internal/engine/continuous_batching.go`, `internal/engine/kv_cache_paged.go`, `internal/engine/engine_cuda.go`

### Part 4: Cross-Engine & Asymmetric Speculative Decoding
- **Goal**: Double text generation tokens-per-second on consumer hardware.
- **Implementation**:
  - Support asymmetric draft/target pairs across engines (e.g. `Qwen3.5-0.8B` drafting on CPU or secondary GPU, while `Qwen3.5-4B` or `Gemma4` verifies candidate tokens in parallel on primary GPU).
  - Add tree-based speculation with dynamic draft length adjustment based on acceptance rate metrics.
- **Files**: `internal/engine/speculative.go`, `internal/engine/interface.go`

### Part 5: Gemma 4 End-to-End Vision-Language Pipeline (VLM)
- **Goal**: Support multi-modal image + text prompts using Gemma 4's native vision encoder.
- **Implementation**:
  - Complete the patch projection pipeline in `internal/vlm/` using `mmproj-*.gguf` weights (`patch_embed`, `proj.weight`, `inp_gate.weight`).
  - Wire image decoding into the OpenAI-compatible HTTP server (`/v1/chat/completions` with base64/URL image inputs).
- **Files**: `internal/vlm/encoder.go`, `internal/api/server.go`, `cmd/quarrel/main.go`

### Part 6: Grammar-Constrained Sampling Expansion (CFG & Regex)
- **Goal**: Deterministic, validated JSON schema and function-call token generation.
- **Implementation**:
  - Expand `internal/sampler/grammar.go` from basic JSON to an efficient pushdown automaton (PDA) supporting arbitrary EBNF grammars and regex constraints.
  - Implement token bitmask pre-filtering before softmax sampling to guarantee valid grammar transitions with zero rejection overhead.
- **Files**: `internal/sampler/grammar.go`, `internal/engine/sampler.go`

### Part 7: AVX-512 VNNI & AMX Quantized Dot Product Kernels
- **Goal**: 2x throughput boost for CPU-only inference on modern x86 hardware.
- **Implementation**:
  - Implement hardware-accelerated integer dot products (`vpdpbusd`) for `Q8_0` and `Q4_K` block accumulations.
  - Add runtime detection for Intel AMX (Advanced Matrix Extensions) with tile register matrix multiplications for high-core CPU servers.
- **Files**: `internal/simd/avx512.go`, `internal/simd/simd.go`

### Part 8: FP8 / Q8_0 Quantized Paged KV Cache
- **Goal**: 50% VRAM reduction for KV caches, enabling 64k–128k context lengths on 8GB GPUs.
- **Implementation**:
  - Implement FP8 (E4M3/E5M2) and Q8_0 page storage in `PagedKVCache` on both CPU and GPU.
  - Add dynamic scale tracking per page block and fused dequant-attention kernels.
- **Files**: `internal/engine/kv_cache_paged.go`, `internal/device/cuda_kernels.cu`

### Part 9: Distributed Pipeline & Tensor Parallelism over Arrow Flight
- **Goal**: Multi-node and multi-GPU sharding for models exceeding single-device memory.
- **Implementation**:
  - Connect `internal/engine/distributed.go` and `internal/device/nccl.go` with Apache Arrow Flight RPC for zero-copy tensor transfer between worker nodes.
  - Implement automatic layer splitting (pipeline parallelism) and column/row weight slicing (tensor parallelism).
- **Files**: `internal/engine/distributed.go`, `internal/engine/remote.go`, `internal/device/nccl.go`

### Part 10: Production Observability, Distributed Tracing & Memory Governors
- **Goal**: Enterprise-grade monitoring and proactive memory protection.
- **Implementation**:
  - Add OpenTelemetry spans tracking TTFT (Time To First Token), inter-token latency, and KV page allocation.
  - Implement proactive memory governors that automatically defragment or offload KV pages when host RAM or VRAM exceeds 85% capacity.
- **Files**: `internal/telemetry/telemetry.go`, `internal/metrics/metrics.go`, `internal/device/memory.go`

---

## Active Backlog & Operational Tasks

### Architecture & Extensibility
| Priority | Item | Files | Description |
|----------|------|-------|-------------|
| P1 | TPU/XLA backend validation | `cmd/quarrel/main_tpu.go` | Test and validate Google TPU inference path |

### Developer Experience & CI
| Priority | Item | Files | Description |
|----------|------|-------|-------------|
| P2 | Consolidate test scripts | `scripts/` | Merge redundant test/benchmark scripts into unified runner |
| P2 | Add coverage gates | `.github/workflows/ci.yml` | Enforce minimum coverage threshold on PRs |
| P2 | Release automation | `Makefile` | Single `make release` target with changelog, Docker push, Helm bump |

---

#### Last updated: September 2026 (v0.4.0 10-Part Architectural & Performance Improvement Plan)