# Longbow-Quarrel - Project Status

## Completed Features (v0.1.0)

All major features from the original 10-part plan have been implemented:

| Feature | Status | Location |
| -------- | -------- | ---------- |
| Metal GPU Backend | ✅ Complete | `internal/device/metal.go` |
| CUDA GPU Backend | ✅ Complete | `internal/device/cuda.go` |
| GGUF Model Loading | ✅ Complete | `internal/gguf/` |
| Sliding Window Attention | ✅ Complete | Mistral 4096 tokens |
| Gemma4 Hybrid Attention | ✅ Complete | 5 sliding + 1 full per 6 layers |
| OpenAI API Endpoints | ✅ Complete | `/v1/chat/completions`, `/v1/completions` |
| Benchmark Tool | ✅ Complete | `cmd/benchmark` |
| Output Validation | ✅ Complete | `compareTokenSequences()` |
| cuDNN Flash Attention | ✅ Complete | `internal/device/cudnn.go` |
| Fused Kernels | ✅ Complete | `cudaFusedAttention`, `cudaFusedRoPE` |

## 15-Step Roadmap: Arrow Integration & Elite Feature Parity

This roadmap is designed to elevate Longbow-Quarrel to compete directly with industry leaders (vLLM, llama.cpp, Ollama) while maintaining a strict focus on zero-copy Apache Arrow architectures for deep integration with the Longbow embedding engine.

### Phase 1: Zero-Copy Arrow Architecture & Longbow Integration

1. **True Zero-Copy Arrow Memory Foundation**: Refactor internal tensor and KV cache representations to natively adhere to the Apache Arrow columnar format in-memory, eliminating serialization overhead.
2. **Arrow Flight Embedding Engine**: Expand `internal/arrow_client/client.go` to provide a dedicated, high-throughput Flight stream, allowing Quarrel to push computed model embeddings directly into Longbow.
3. **Direct GPU-to-GPU Flight Transfers**: Adopt GPUDirect/Arrow GPU features to transfer dense embedding matrices straight from Metal/CUDA VRAM into Longbow via Arrow Flight, bypassing the CPU entirely.
4. **Arrow-backed Streaming Inference API**: Upgrade the standard Server-Sent Events (SSE) output to support Arrow Flight for real-time generation. This allows downstream consumers to receive highly structured, zero-copy inference tokens and metadata.

### Phase 2: Performance & Concurrency (Parity with vLLM)

1. **Continuous Batching & PagedAttention**: Implement true continuous batching paired with PagedAttention. This eliminates static batching bottlenecks and allows for hyper-efficient KV cache memory management in high-concurrency environments.
2. **Chunked Prefill & Prompt Caching**: ✅ Complete. Split long system and context prompts into optimal chunks to drastically reduce Time-To-First-Token (TTFT) and globally cache common prompt state across requests.
3. **Speculative Decoding Orchestration**: ✅ Complete. Integrate rejection sampling and sync mechanisms between Draft and Target KV caches to maximize the acceptance rate of speculative tokens.
4. **FlashAttention-2 Compatibility**: Evolve existing exact-match attention kernels into FlashAttention-2 (and Metal equivalent) implementations, leveraging threadgroup memory/shared memory to push hardware to its theoretical utilization limits.

### Phase 3: Advanced Features & Ecosystem (Parity with llama.cpp/Ollama)

1. **Speculative Decoding Pipelines**: Implement draft-model speculative decoding to accelerate token generation speeds by 2x-3x through multi-draft concurrent processing.
2. **Multi-LoRA Dynamic Serving**: Support loading and hot-swapping multiple Low-Rank Adaptations on the fly, enabling a single base model instance to serve disparate fine-tuned use-cases simultaneously.
3. **Structured Outputs & Grammar Sampling**: Integrate structured inference (JSON mode and formal grammar definition), enforcing model compliance to expected downstream application schemas.
4. **Vision-Language Model (VLM) Architectures**: Expand backend architecture to route multimodal inputs through vision encoders (e.g., CLIP, SigLIP) for leading open weights models like Llava and Qwen-VL.
5. **Comprehensive Quantization Support**: Round out GGUF compatibility to include the missing configurations (`Q5_K`, `Q2_K`, etc.) and upcoming low-bit floating point formats (e.g., GGUF v3).

### Phase 4: Stability & Enterprise Orchestration

1. **Multi-GPU Tensor Parallelism (NCCL/Metal)**: Implement distributed inference capable of sharding model weights across multiple PCIe/NVLink GPUs using NCCL, or multi-M-Max SoCs.
2. **Kubernetes-native Fault Tolerance**: Introduce strict memory budget enforcement (graceful degradation), liveness/readiness probes, and semantic routing features for deployment within large-scale container orchestration.

---

#### Last updated: April 2026

## Phase 1 Implementation Plan

### 1. True Zero-Copy Arrow Memory Foundation

**Objective:** Allow `device.Tensor` to be consumed directly by Apache Arrow ecosystem without CPU copy-loops.
- **Change:** Add `ToArrowArray(allocator memory.Allocator)` to `device.Tensor` (`metal.go`, `cuda.go`, `cpu.go`). 
- **Mechanism:** For Metal, when shared memory mode is used, we can directly map `buf.contents()` to an `arrow.Buffer` using the Arrow C-Data interface or raw pointer injection, wrapping it in an `array.Float32` or `array.FixedSizeList`.

### 2. Arrow Flight Embedding Engine Expansion

**Objective:** Pump embeddings from the model's final hidden state directly to the Longbow via Flight streaming.
- **Change:** Expand `FlightClient` in `internal/arrow_client/client.go` to include a `StreamEmbeddings` method that takes `*device.Tensor` rather than `[][]float32`.
- **Mechanism:** Using the zero-copy Arrow arrays created in Step 1, push them over the established gRPC Flight stream.

### 3. GPU-to-GPU Flight Transfers (Foundational)

**Objective:** Prepare the structures for CUDA GPUDirect / Metal to Arrow GPU integration.
- **Change:** Add GPU device ID and IPC handler pointers into the Arrow schemas/flight descriptors so Longbow knows the tensor lives in VRAM.

### 4. Arrow-backed Streaming Inference API

**Objective:** Expose generation as an Arrow Flight flight stream instead of traditional HTTP SSE.
- **Change:** Build `internal/api/flight_server.go` which runs alongside the REST API, offering a `DoGet` method for text generation tokens as a schema `[token_id: int32, logits: list<float32>]`.

### 5. Hotpath-Safe Prometheus Metrics

**Objective:** Ensure metrics (like bytes transferred via Arrow, flight stream errors) don't lock or allocate in fast-paths.
- **Change:** Add `atomic.Int64` counters inside `Context` or `FlightClient` that get periodically flushed to `prometheus.Counter` by a background goroutine, avoiding mutexes or `prometheus` library locks during generation.

## Phase 2 Implementation Plan

### 1. Continuous Batching & PagedAttention

**Objective:** Replace naive iteration loops with a continuous batching manager that supports concurrent sequence execution without pre-allocating contiguous memory per request.
- **Change:** Refactor `internal/engine/engine.go` and `internal/engine/kv_cache_paged.go`. The engine will maintain a `request_queue` and a `running_queue`. The inference loop will continuously pull from the `request_queue` up to the max token sequence limits.
- **Mechanism:** Modify `PagedKVCache` to support dynamic allocation at inference time, utilizing the `BlockTable` directly inside the new Attention kernels.

### 2. FlashAttention-2 Compatibility

**Objective:** Consolidate memory bandwidth by fusing the entire Q*K*V attention block inside shared threadgroup memory (SRAM), drastically improving memory-bound decoding speeds.
- **Change:** Introduce `Metal_FlashAttention2_F16` inside `internal/device/kernels.metal` and `internal/device/metal.go`.
- **Mechanism:** Implement block-tiling in Metal, keeping the Q array in threadgroup memory, and looping through the K and V blocks managed by the Paged BlockTable.

### 3. Chunked Prefill & Prompt Caching

**Objective:** Split large prompts into chunks over multiple iterations to prevent long prompts from starving decoding requests (Time-To-First-Token optimization).
- **Change:** Implement `RadixTree` inside `internal/engine/prompt_cache.go` to hash and identify prefix match chains in system prompts.
- **Mechanism:** Allocate KV Cache blocks globally for system prompts via a strict LRU cache policy, rather than sequence-scoped arrays.

### 4. TurboQuant KV Cache Compression

**Objective:** Expand context window capacity limits drastically by finalizing the TurboQuant custom KV cache logic natively inside PagedKVCache allocations.
- **Change:** Utilize `DataTypeTQ1_0` and `DataTypeTQ2_0` inside PagedKVCache buffers (`internal/device/utils.go`).
- **Mechanism:** During each dynamic memory block allocation, compress encoded elements using the fused PolarQuant + QJLTransform native Metal kernels (`turboquant_encode`), yielding robust low-bit residual caches pinned inside the BlockTable mappings.