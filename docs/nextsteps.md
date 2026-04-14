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
4. **FlashAttention-2 Compatibility**: ✅ Complete. Evolve existing exact-match attention kernels into FlashAttention-2 (and Metal equivalent) implementations, leveraging threadgroup memory/shared memory to push hardware to its theoretical utilization limits.

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
---

## Phase 5: Performance Hardening & Feature Completions

*The following 10 steps are derived from a deep code review. Each step identifies a concrete gap in the live implementation, with specific file and line references.*

---

### 5. Wire Real Rejection Sampling into Speculative Decoding

**Objective:** The speculative acceptance logic in `speculative.go` is a stub — all draft tokens are currently accepted unconditionally regardless of the target model's probability distribution.

- **Gap:** `internal/engine/speculative.go:93` — `// verify(targetLogits[i], candidates[p][i])` is commented out; `accepted++` fires unconditionally.
- **Change:** Implement `func rejectSample(targetLogits []float32, draftToken int, cfg SamplerConfig) (accepted bool, correctedToken int)` using the criterion `accept if u ~ Uniform(0,1) < min(1, p_target[t] / p_draft[t])`. On rejection, sample the correction token from the renormalized `(p_target - p_draft)+` distribution and truncate the accepted prefix at `i`.
- **Metrics:** Add `SpeculativeTokensAccepted` / `SpeculativeTokensRejected` counters to `internal/metrics/metrics.go` to track acceptance rate per model pair in production.

---

### 6. Copy-on-Write Physical Block Data in PagedKVCache

**Objective:** The CoW path in `PagedKVCache.Update()` allocates a new physical block but does not copy existing KV data into it — a documented correctness gap that silently corrupts forked sequences.

- **Gap:** `internal/engine/kv_cache_paged.go:399–404` — explicit comment: *"Note: We'd normally need to copy actual physical tensor data from physBlock to newPhys here!"*
- **Change:** Expose `ctx.CopyTensorRows(src, dst *device.Tensor, srcRowStart, dstRowStart, numRows int)` implemented as a device-side blit in `internal/device/metal.go` (and CUDA equivalent).
- **Wire In:** Call `ctx.CopyTensorRows(c.kPools[layer], c.kPools[layer], oldPhysOffset, newPhysOffset, c.blockSize)` for every layer immediately after the CoW block is allocated in `Update()`.

---

### 7. PromptCache LRU Eviction to Prevent OOM

**Objective:** `internal/engine/prompt_cache.go` has no eviction policy. Under memory pressure, cached prefix blocks are never released back to `PagedKVCache.freeBlocks`, eventually exhausting the block pool.

- **Gap:** `PromptCache.Insert` increments `RefCount` indefinitely with no complementary `Evict` path or maximum size bound.
- **Change:** Add `maxCachedBlocks int` and an LRU-ordered node list to `PromptCache`. Implement `func (pc *PromptCache) Evict(kvCache *PagedKVCache) (freed int)` that removes the LRU leaf, calls `kvCache.FreeSequence`, and returns the block count released.
- **Wire In:** Call `pc.Evict(kvCache)` from `ContinuousBatchManager.Step` when `kvCache.FreeBlocksCount()` drops below a configurable `LowWaterMark` field (default 32 blocks), before the admission loop runs.

---

### 8. Multi-LoRA Batch Dispatch — Per-Row Adapter Routing

**Objective:** `BatchDescriptor.AdapterIDs` records the active adapter per sequence, but the metal engine forward pass ignores it and applies (or skips) a single adapter for the full batch, making concurrent multi-LoRA serving semantically incorrect.

- **Gap:** In `engine.go` (metal build), the `runBatchLoop` forward pass does not inspect `desc.AdapterIDs` when calling `LoRAManager.GetWeights`.
- **Change:** Group sequences in `BatchDescriptor` by `AdapterID`. For each adapter group, apply its LoRA delta `(B·A) * (alpha/rank)` via a fused `Metal_LoRAForwardBatch` kernel accepting an `adapter_mask` index per row. Stage both A and B matrices into a pre-allocated adapter slab indexed by `(adapterID, layerIdx)` for O(1) lookup.
- **Metric:** Emit `lora_dispatch_groups_per_batch` histogram to measure adapter batching efficiency.

---

### 9. OpenTelemetry OTLP Exporter Wiring

**Objective:** `internal/telemetry/telemetry.go` creates a `TracerProvider` with `AlwaysSample()` but no exporter — every span is silently discarded, providing zero distributed tracing signal.

- **Gap:** `telemetry/telemetry.go:35` — `sdktrace.NewTracerProvider` has no `WithBatcher` or `WithSpanProcessor` argument.
- **Change:** Conditionally wire an OTLP gRPC exporter when `OTEL_EXPORTER_OTLP_ENDPOINT` is set; fall back to `stdouttrace.New()` for local development.
- **Wire In:** Stop blanking the `ctx` return value in `api/server.go:123` and `api/adapters.go:27` (`_ = ctx`). Propagate the span-enriched context through to `engine.Infer` so nested engine spans nest correctly under the HTTP handler span, resolving both outstanding IDE warnings in one pass.

---

### 10. Min-P Filtering and Mirostat v2 Adaptive Sampling

**Objective:** `internal/engine/sampler.go` is missing two widely-used sampling modes available in llama.cpp and Ollama: Min-P (relative probability floor) and Mirostat v2 (entropy-targeting adaptive temperature).

- **Gap:** `internal/engine/sampler_config.go` has no `MinP`, `MirostatTau`, or `MirostatEta` fields.
- **Change (Min-P):** Add `MinP float64` to `SamplerConfig`. After temperature softmax, filter candidates where `c.prob < MinP * candidates[0].prob`. Cheaper than Top-P and avoids incoherence at low temperatures.
- **Change (Mirostat v2):** Add `MirostatTau`, `MirostatEta float64` fields and per-`Sampler` `mirostatMu float64` state. After sampling, compute surprise `s = -log2(p_sampled)`, update `mu -= eta * (s - tau)`, clip logits above `mu` before the next step.
- **Expose:** Add corresponding JSON fields to `CompletionRequest` in `internal/api/server.go`.

---

### 11. VLM Encoder — Real Pixel Preprocessing

**Objective:** `internal/vlm/encoder.go` decodes the image container but fills the pixel tensor with all zeros, so every vision embedding is a meaningless zero vector regardless of the image content.

- **Gap:** `vlm/encoder.go:57` — `pixels := make([]float32, TargetW*TargetH*Channels)` is never populated. The decoded `image.Image` is discarded.
- **Change:** Use `golang.org/x/image/draw.CatmullRom.Scale` to bicubic-resize to 224×224. Fill pixels as `(float32(channel_value)/255.0 - mean[c]) / std[c]` using per-architecture normalization constants loaded from GGUF metadata (`clip.vision.image_mean`, `clip.vision.image_std`) into a new `VisionEncoderConfig` struct.
- **Test:** Add a test in `vlm/encoder_test.go` loading a 4×4 solid-color PNG and asserting the output patch tensor is non-zero.

---

### 12. MasterDistributedEngine — Real Tensor Parallel All-Reduce

**Objective:** `internal/engine/distributed_master.go` stubs all `ForwardBatch` calls by forwarding entirely to `shards[0]`. Remaining shards receive no computation, making the distributed engine non-functional for Tensor Parallelism.

- **Gap:** `distributed_master.go:56` — `// Stub: currently delegating to primary`.
- **Change (Phase A):** Add `ForwardShardedLayer(layerIdx int, colStart, colEnd int, input *device.Tensor) (*device.Tensor, error)` to the `DistributedEngine` interface. `RemoteWorkerEngine` implements it via a bidirectional Arrow Flight RPC: `DoPut` receives input activations, `DoGet` returns partial output activations, reusing the `InferenceFlightServer` scaffold in `internal/api/flight_server.go`.
- **Change (Phase B):** In `MasterDistributedEngine.ForwardBatch`, fan out one `ForwardShardedLayer` goroutine per shard via `errgroup`, then call `ctx.ConcatTensors` (column axis) to reconstruct the full hidden state before proceeding to the next layer.

---

### 13. Eliminate Per-Layer Block Table CPU Round-Trips

**Objective:** `PagedKVCache.Get()` unconditionally converts the block table to `[]float32` and calls `LoadFrom` (a CPU→GPU upload) on every call. For a 32-layer model with batch size 8, this is 256 unnecessary host-device transfers per forward pass.

- **Gap:** `internal/engine/kv_cache_paged.go:559–565` — `goTable := make([]float32, len(table))` + `tableDevice.LoadFrom(goTable)` fires unconditionally inside the hot path.
- **Change (short-term):** Track `dirty map[string]bool` in `PagedKVCache`. Only call `LoadFrom` when `dirty[seqID]` is true (set by `Update`, `Allocate`, `AttachPrefixBlocks`), then clear the flag.
- **Change (long-term):** Unify `blockTablesDevice` as a single 2D tensor `[maxBatchSize, maxBlocksPerSeq]` updated in-place with a `ctx.SetRow` primitive, passed once per forward pass to the attention kernel via `GetBatch`.

---

### 14. CPU Engine Full Layer Stack via Accelerate/BLAS GEMM

**Objective:** `internal/engine/engine_cpu.go` `forward()` only copies the last token's embedding and returns it without executing any Attention or FFN layers, making the CPU engine produce semantically incorrect output for all real models.

- **Gap:** `engine_cpu.go:406–421` — the function body is a pure embedding lookup stub with no matrix multiplications.
- **Change:** Implement the full transformer loop using `simd.MatMulF32` backed by `cblas_sgemm` (Apple `Accelerate.framework` on macOS, `OpenBLAS` on Linux via cgo): `embed → [RMSNorm, QKV-GEMM, Softmax, AttnO-GEMM, residual] × L → [RMSNorm, Gate/Up-GEMM, SwiGLU, Down-GEMM, residual] × L → OutputNorm → LogitGEMM`.
- **Benchmark:** Add `BenchmarkCPUForwardLayer` to `engine_cpu_unit_test.go`, gating merges on ≥2 tok/s for a single 7B-class attention layer (dim=4096, heads=32) on Apple Silicon.
- **Impact:** A working CPU engine enables `SpeculativeManager` to use CPU as a zero-VRAM draft model for any Metal/CUDA target, unlocking speculative decoding without a second GPU.

---

## Additional Stubbed/Incomplete Code Found in Codebase

### 1. RemoteWorkerEngine Flight RPC Not Implemented
- **Location:** `internal/engine/remote.go:45,55,59`
- `ForwardShard()`, `InferWithCallback()`, `ForwardBatch()` all return not-implemented errors

### 2. StoreKVQuantized Panics on CUDA
- **Location:** `internal/device/cuda.go:558`, `internal/device/cpu.go:685`
- Returns panic for unimplemented quantized KV cache storage

### 3. CPU VisionPatchEmbed Stubs  
- **Location:** `internal/device/cpu.go:690-697`
- CPU stubs for VLM patch embedding

### 4. Multi-GPU NCCL Stubs
- **Location:** `internal/device/multi_gpu.go:11`
- Placeholder for distributed training NCCL integration

### 5. CUDA Engine SwapModel Not Implemented
- **Location:** `internal/engine/engine_cuda.go:288`
- Returns error that swap is not implemented

### 6. Q5_K Falls Back to Q4_K
- **Location:** `internal/engine/engine.go:543-544`
- Q5_K quantization falls back to Q4_K instead of full implementation

### 7. CPU ForwardDraft Not Implemented
- **Location:** `internal/engine/engine_cpu.go:511`
- Returns error for speculative decoding on CPU

### 8. Grammar Initialization Stubbed
- **Location:** `internal/api/server.go:166`
- Grammar field is ignored, not enforced

---

## Validation of Existing Next Steps

The following items from the original document were checked against current codebase:

| Item | Status | Verified Location |
|------|--------|-------------------|
| 5. Rejection Sampling | ⚠️ Still stubbed | `internal/engine/speculative.go:93` - verify() commented out |
| 6. CoW Physical Block Copy | ⚠️ Still stubbed | `internal/engine/kv_cache_paged.go:399-404` - comment confirms missing copy |
| 7. PromptCache LRU Eviction | ⚠️ Not implemented | No Evict method found in prompt_cache.go |
| 8. Multi-LoRA Batch Dispatch | ⚠️ Not implemented | No AdapterIDs handling in ForwardBatch |
| 9. OTLP Exporter | ⚠️ Still stubbed | `internal/telemetry/telemetry.go:35` - no WithBatcher |
| 10. Min-P / Mirostat | ⚠️ Not implemented | No fields in `sampler_config.go` |
| 11. VLM Encoder Pixels | ⚠️ Still stubbed | `internal/vlm/encoder.go:57` - pixels created but zeroed |
| 12. Tensor Parallel All-Reduce | ⚠️ Still stubbed | `internal/engine/distributed_master.go:56` - delegates to shard[0] |
| 13. Block Table CPU Round-trips | ⚠️ Still stubbed | `internal/engine/kv_cache_paged.go:559-565` - unconditional LoadFrom |
| 14. CPU Engine Full Stack | ⚠️ Still stubbed | `internal/engine/engine_cpu.go:389-405` - only embedding lookup |
