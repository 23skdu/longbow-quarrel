# Longbow-Quarrel Development Roadmap

## Executive Summary

| Feature | Status | Priority |
|---------|--------|----------|
| **Metal Backend (Apple)** | ✅ RESTORED | - |
| **CUDA Backend (Linux)** | ✅ IMPLEMENTED | - |
| **Test Coverage** | ✅ IMPROVED (36.8% → 46.8%) | - |
| **WebUI Service** | ✅ COMPLETE | - |
| **Production Integration** | ✅ COMPLETE | - |
| **cuDNN Integration** | ✅ COMPLETE | - |
| **FP8 Support (H100)** | ✅ COMPLETE | - |
| **All High/Medium Priority Items** | ✅ COMPLETE | - |

---

## Completed Features

### ✅ Production Integration
- **Status:** COMPLETE
- **Files:** `cmd/webui/`, `internal/engine/`
- **Features:** Engine adapter, hot-swapping, per-sequence KV cache tracking

### ✅ Metal Backend (Apple Silicon)

### ✅ Metal Backend (Apple Silicon)
- **Status:** RESTORED and verified
- **Files:** `internal/device/kernels.metal`, `internal/device/metal_backend.m`
- **Coverage:** 61 GPU kernels for FP16, Q3_K, Q4_K, Q6_K, Q8_0

### ✅ CUDA Backend (Linux)
- **Status:** IMPLEMENTED
- **Files:** `internal/device/cuda.go`, `internal/device/cuda_kernels.cu`
- **Features:** Tensor Core WMMA, Flash Attention, Paged KV Cache

### ✅ CPU SIMD Optimizations
- **Status:** COMPLETE
- **AVX2:** Softmax, SwiGLU, FP16→FP32 conversion
- **AVX-512:** Zen 4+, Ice Lake+ support

### ✅ Branchless Quantization
- **Status:** COMPLETE
- **Files:** `internal/gguf/dequant.go`
- **Optimizations:** Q4K/Q6K dequantization without branch mispredictions

### ✅ Test Coverage Improvements
- **Status:** IMPROVED (+10% overall)
- `internal/config`: 0% → 100%
- `internal/logger`: 0% → 100%
- `internal/ollama`: 0% → 41%
- `internal/cpu`: 76.1% → 81.8%
- `internal/simd`: 65.9% → 76.8%
- `internal/metrics`: 4.9% → 88.2%

### ✅ WebUI Service
- **Status:** COMPLETE (commit 9d8fbda)
- **Files:** `cmd/webui/` (19 files, +2363 lines)
- **Features:** Templ-based UI, WebSocket streaming, Docker support

---

## Pending Features (Backlog)

### Medium Priority

#### cuDNN Integration
- [x] Add cuDNN for additional optimization on NVIDIA GPUs
- [x] Leverage cuDNN's optimized attention kernels
- [x] Support grouped convolutions for MOE models

#### FP8 Support (H100)
- [x] Implement FP8 E4M3/E5M2 quantization
- [x] Tensor Core FP8 support on Hopper architecture
- [x] FP8 dequantization kernels

### Low Priority

#### Multi-GPU Support
- **Status:** ✅ IMPLEMENTED
- **Files:** `internal/device/cuda.go`, `internal/device/multi_gpu.go`, `internal/device/cuda_kernels.cu`
- **Tensor Parallelism:**
  - `TensorParallelManager` - Manages tensor-parallel operations
  - AllReduce, AllGather, Broadcast collective operations
  - Weight splitting across GPUs
- **Pipeline Parallelism:**
  - `PipelineParallelManager` - Manages pipeline stages
  - `PipelineStage` - Individual stage with layer ranges
  - Micro-batch scheduling with configurable depth
- **Cross-GPU Communication:**
  - `CrossGPUCommunicator` - Peer-to-peer memory access
  - CUDA P2P API for direct GPU-to-GPU transfers
  - Peer access matrix for bandwidth estimation
- **Hybrid Manager:**
  - `HybridParallelismManager` - Unified multi-GPU coordination
  - Automatic layer distribution across devices
  - Collective operation coordination
- **NCCL Collective Kernels:**
  - AllReduce, AllGather, ReduceScatter, Broadcast
  - Tensor parallelism reduction kernels
  - Pipeline send/receive kernels

#### vLLM Integration
- **Status:** COMPLETE (Export operators package implemented, commit `bbcc523`)
- **Files:** `cmd/vllm_export/`, `internal/device/cuda_export.go`
- **Export Operators:**
  - `Init()` - CUDA context initialization
  - `GetDeviceCount()` - Query available CUDA devices
  - `GetDeviceName()` - Get device name
  - `GetMemoryInfo()` - Query GPU memory
  - `DequantizeQ8_0/4_K/6_K()` - Weight dequantization
  - `RMSNorm()` - RMS normalization
  - `SwiGLU()` - SwiGLU activation
  - `RoPE()` - Rotary positional encoding
  - `Attention()` - Multi-head attention with KV cache
  - `MatMul()` - Matrix multiplication
  - `Synchronize()` - Stream synchronization
- **Export Package:** `internal/device/cuda_export.go` - Exported CUDA functions as C-shared library
- **Note:** PyTorch custom op registration and batch scheduler remain as future enhancements

---

## Build Commands Reference

```bash
# Metal backend (macOS Apple Silicon)
CGO_ENABLED=1 go build -tags metal ./...

# CUDA backend (Linux NVIDIA)
go build -tags=cuda,amd64 ./...

# AVX2 only (no GPU)
go build -tags=amd64 ./...

# AVX-512 (Zen 4+, Ice Lake+)
go build -tags=amd64,avx512 ./...

# All optimizations
go build -tags=cuda,amd64,avx512 ./...

# CUDA with NCCL (Multi-GPU)
go build -tags=cuda,amd64,nccl ./...
```

---

## Testing Commands

```bash
# Run all tests with coverage
go test -coverprofile=coverage.out ./...

# Run specific package tests
go test ./internal/cpu/...
go test ./internal/simd/...
go test ./internal/metrics/...

# Metal-specific tests (macOS)
go test -tags=metal ./internal/device/...
```

---

## Incomplete Features & Code TODOs

### High Priority (Blocking Functionality)

#### Model Hot-Swapping (cmd/webui/engine/adapter_*.go)
- **Status:** ✅ FIXED
- **Issue:** `UnloadModel()` exists but no hot-swap logic for active requests
- **Fix:** Implemented `HotSwapModel()` in both CUDA and Metal adapters with active request waiting
- **Location:** `cmd/webui/engine/adapter_cuda.go:283-349`, `cmd/webui/engine/adapter_metal.go:197-240`

#### KV Cache Sharing Between Requests (internal/engine/engine.go)
- **Status:** ✅ IMPLEMENTED
- **Issue:** Current sequential processing, single CachePos
- **Fix:** Added SequenceManager for per-sequence position tracking; replaced single CachePos with per-sequence tracking
- **Changes:**
  - Added Sequence struct and SequenceManager in types.go
  - Modified inferInternal to accept SequenceID from SamplerConfig
  - Replaced all "seq-0" with e.SeqIDStr(seq.ID) for proper sequence isolation
  - Each sequence now has its own cache position tracked independently
- **Location:** `internal/engine/types.go`, `internal/engine/engine.go`

#### Mamba Layer Detection (internal/engine/mamba.go:91)
- **Status:** ✅ FIXED
- **Issue:** `IsMambaLayer()` relied on weight-based nil check - needed proper config-based detection
- **Fix:** Implemented config-based detection with multiple patterns:
  - `"all"` - Pure Mamba model (all layers are Mamba)
  - `"none"` - Pure Transformer (no Mamba layers)
  - `"even"` - Hybrid model with Mamba on even layers (0, 2, 4, ...)
  - `"odd"` - Hybrid model with Mamba on odd layers (1, 3, 5, ...)
  - Weight-based fallback for custom patterns
- **Changes:**
  - Added `IsHybrid` and `MambaLayerPattern` fields to Config
  - Added `detectMambaLayers()` method to detect pattern from GGUF metadata
  - Added `CountMambaLayers()` helper method
  - Updated `IsMambaLayer()` with config-based detection priority
- **Location:** `internal/engine/mamba.go`, `internal/config/config.go`

#### Tensor Zero() Method (internal/engine/engine.go:1191)
- **Status:** ✅ FIXED
- **Issue:** `// TODO: Add Zero() to tensor` - SSM state initialization incomplete
- **Fix:** Added `ZeroInit()` calls for ConvState and SSMState tensors

#### Quantization Support (internal/gguf/quantize.go:6)
- **Status:** ✅ IMPLEMENTED
- **Issue:** `QuantizeWeightsToQ4K()` returns "not implemented"
- **Fix:** Implemented Q4_K quantization encoder consistent with existing DequantizeQ4K decoder
- **Changes:**
  - Added `Float32ToFloat16()` helper in `internal/gguf/quantize.go`
  - Implemented `QuantizeWeightsToQ4K()` with per-block min/max computation, float16 roundtrip for d/dmin, per-group scale packing (inverse of decoder), and 4-bit weight quantization
  - Updated `TestQuantizeWeightsToQ4KNotImplemented` → `TestQuantizeWeightsToQ4K` in `internal/gguf/gguf_test.go`
  - Added roundtrip test in `internal/gguf/quantization_test.go`
- **Location:** `internal/gguf/quantize.go`

### Medium Priority (Quality/Performance)

#### RoPE Attention Causal Mask Test (internal/device/rope_attention_test.go:122)
- **Status:** ✅ FIXED
- **Issue:** `// TODO: Call attention kernel with causal mask`
- **Fix:** Updated test to call `q.Attention()` kernel with proper parameters

#### Perplexity Calculation (internal/engine/engine.go:113)
- **Status:** ✅ FIXED
- **Issue:** Simplified implementation - `// This is just a placeholder - real perplexity requires model probabilities`
- **Fix:** Implemented `CalculatePerplexityFromLogits()` on Engine that runs the forward pass at each token position, collects logits, computes log probabilities via log-sum-exp, and returns real model-based perplexity
- **Changes:**
  - Added `Engine.CalculatePerplexityFromLogits(tokens []int) PerplexityResult` in `internal/engine/engine.go`
  - Updated `calculatePerplexityForTokens()` in `cmd/smoke_test/moe_regression_test.go` to use real logits
  - Added `TestEngine_CalculatePerplexityFromLogits` edge case test in `internal/engine/quality_test.go`
- **Location:** `internal/engine/engine.go:131`, `cmd/smoke_test/moe_regression_test.go:311`

#### Quality Evaluator Tests (internal/engine/smollm2_zero_logits_test.go:17,23)
- **Status:** DOCUMENTED
- **Issues:**
  - Skipped tests documented with implementation notes
  - Tests require loaded model + tokenizer which isn't available in unit tests

### Low Priority (Future/Backlog)

#### CUDA Coherence Tests (cmd/smoke_test/cuda_coherence_test.go)
- **Status:** ✅ IMPLEMENTED
- **Issue:** `t.Skip("CUDA engine not implemented - this is a placeholder for CUDA coherence tests")`
- **Fix:** Implemented 4 CUDA coherence tests using synthetic GGUF model and CUDA engine
- **Tests:**
  - `TestCUDACoherenceWrapping` — Context window wrapping with 32-position KV cache
  - `TestCUDAMultiTokenCoherence` — Multiple sequential inference prompts
  - `TestCUDASelfConsistency` — Two engines produce identical output with temperature=0
  - `TestCUDAKVCacheCorrectness` — Consecutive inferences with different inputs (no crashes)
- **Location:** `cmd/smoke_test/cuda_coherence_test.go`

#### Inference String Method (internal/engine/engine.go:1260)
- **Status:** ✅ FIXED
- **Issue:** Uses `inputTokens := []int{1, 2, 3} // Placeholder tokenization`
- **Fix:** Updated to use `e.Tokenizer.Encode(prompt)` and `e.Tokenizer.Decode(tokens)`

#### IQ1_M Quantization (internal/gguf/structs.go)
- **Status:** ✅ IMPLEMENTED
- **Issue:** `t.Errorf("IQ1_M SizeBytes() = %d, want 0 (not implemented)", got)`
- **Fix:** Added IQ1_M block size: 56 bytes per 256 elements

#### Fused QKV + RoPE Kernel (internal/device/cuda_kernels.cu:1158)
- **Status:** ✅ IMPLEMENTED
- **Issue:** `// TODO: Add fused QKV + RoPE kernel with precomputed frequencies`
- **Fix:** Implemented proper fused RoPE kernel that applies rotary positional encoding to pre-computed Q and K projections using precomputed cos/sin frequency tables in a single kernel launch
- **Changes:**
  - Rewrote `fused_qkv_rope_kernel` in `internal/device/cuda_kernels.cu` to apply RoPE rotation to Q and K with precomputed cos/sin tables, passing V through unchanged
  - Updated `cudaFusedQKVRope` C export function with correct signature and parameters
  - Added `CUDAContext.FusedQKVRope()` Go method in `internal/device/cuda.go` with cos/sin table precomputation
  - Added `TestCUDAFusedQKVRope` test verifying RoPE application, V passthrough, and norm preservation
- **Location:** `internal/device/cuda_kernels.cu:991`, `internal/device/cuda.go:574`

---

## API Endpoints Summary

| Endpoint | Method | Description | Status |
|----------|--------|-------------|--------|
| `/health` | GET | Health check | ✅ Complete |
| `/healthz` | GET | Simple liveness | ✅ Complete |
| `/readyz` | GET | Readiness probe | ✅ Complete |
| `/version` | GET | Version info | ✅ Complete |
| `/metrics` | GET | Prometheus metrics | ✅ Complete |
| `/api/models` | GET | List models | ✅ Complete |
| `/api/generate` | POST | Generate text (sync) | ✅ Complete |
| `/api/stream` | POST | Stream text (SSE) | ✅ Complete |
| `/ws` | WebSocket | Real-time inference | ✅ Complete |

---

## Plan: Feature Parity with vLLM & Performance Leadership

This section outlines the roadmap to achieve feature equivalence with vLLM and surpass it in performance on the same models.

### Phase 1: Performance Foundation (Immediate)

**Goal:** Close the 3-14x performance gap with llama.cpp/vLLM on existing hardware.

| Task | Target | Expected Impact |
|------|--------|-----------------|
| Reduce Metal sync overhead | -60% sync calls | 2-3x speedup |
| Batch kernel dispatch | Combine ops | 1.5x speedup |
| KV cache optimization | Paged cache default | 1.2x speedup |
| Quantized kernel tuning | Q4_K optimization | 1.5x speedup |

**Deliverables:**
- [ ] Implement kernel fusion pipeline (QKV + RoPE + Attention)
- [ ] Add Persistent Batch pattern (cache input tensors)
- [ ] Optimize Metal memory allocator (reduce fragmentation)
- [ ] Implement Flash Attention for Metal backend
- [ ] Add continuous batching scheduler

### Phase 2: Feature Parity (Q2 2025)

**Goal:** Implement missing critical features from vLLM.

#### P0 - Critical

| Feature | Implementation | Dependencies |
|---------|-----------------|--------------|
| Hash-based Prefix Caching | O(1) LRU cache with content hashing | KV cache refactor |
| Chunked Prefill | Split large prompts across steps | Batching scheduler |
| Safetensors Support | Add Safetensors loader | Model loading |
| Advanced Continuous Batching | vLLM-style scheduler | Request queue redesign |

#### P1 - High Priority

| Feature | Implementation | Dependencies |
|---------|-----------------|--------------|
| Speculative Decoding | Draft-verifier pattern | Sampling layer |
| Structured Output | JSON/schema validation | Sampling layer |
| Full OpenAI API | /embeddings, /completions | API layer |

### Phase 3: Advanced Features (Q3 2025)

**Goal:** Implement differentiated capabilities.

| Feature | Target | Differentiation |
|---------|--------|-----------------|
| Multimodal (VLM) | Image input support | Native Go implementation |
| Multi-LoRA | Multiple adapter support | Model loader |
| Embedding Models | Encoder support | Model runner |
| Speculative Decoding | Medusa/Eagle style | Advanced sampling |

### Phase 4: Performance Leadership (Q4 2025)

**Goal:** Surpass vLLM on key metrics.

#### Performance Targets

| Metric | Current | vLLM Reference | Target |
|--------|---------|----------------|--------|
| Throughput (7B) | 1.9 t/s | ~50 t/s (H100) | >60 t/s |
| Throughput (Smollm2) | 38.8 t/s | N/A | >60 t/s |
| Prefix cache overhead | N/A | <1% | <0.5% |
| Cold start time | ~10s | ~8s | <5s |
| Memory efficiency | Baseline | Baseline | -20% |

#### Differentiation Strategy

1. **Go Runtime Advantages:**
   - Lower memory footprint (no Python interpreter)
   - Better concurrency for multi-request handling
   - Faster cold starts

2. **Native Metal Backend:**
   - Exclusive Apple Silicon optimization
   - No PyTorch overhead on M-series chips
   - Target: Beat vLLM on M3 Pro/M4

3. **Architecture Innovations:**
   - Implement vLLM V1-style EngineCore pattern in Go
   - Zero-copy tensor management
   - goroutine-based request scheduling

### Technical Implementation Details

#### 1. Kernel Fusion Pipeline

```go
// Target fused kernel structure
type FusedOperation struct {
    QKVProjection  // Combined Q, K, V projection
    RoPE           // Rotary positional encoding
    Attention      // Flash attention with causal mask
    Softmax        // Combined with attention
    OutputProjection
}

// Benefits: Single GPU kernel launch, minimal memory traffic
```

#### 2. Persistent Batch Pattern

```go
// Cache input tensors across inference steps
type PersistentBatch struct {
    inputCache map[RequestID]*CachedInput  // Persist token tensors
    diffs      map[RequestID][]int        // Incremental updates only
}

// Benefits: Reduces per-step tensor allocation by ~70%
```

#### 3. Hash-Based Prefix Caching

```go
type PrefixCache struct {
    hashIndex map[uint64]CacheEntry  // O(1) lookup
    lru       list.List              // O(1) eviction
    lock      sync.RWMutex
}

// Benefits: Near-zero overhead prefix matching
```

#### 4. Continuous Batching Scheduler

```go
type Scheduler struct {
    pending    *PriorityQueue  // Waiting requests
    running    *TokenBudget    // Active sequences
    maxTokens  int             // Budget per iteration
    prefillRatio float64       // Prefill/decode balance
}

// Benefits: Maximize GPU utilization, minimize latency
```

### Resource Requirements

| Phase | Engineers | Timeline | Key Dependencies |
|-------|-----------|----------|------------------|
| Phase 1 | 1-2 | 6 weeks | Metal/CUDA kernels |
| Phase 2 | 2 | 12 weeks | KV cache, scheduler |
| Phase 3 | 2 | 12 weeks | Model loader, sampling |
| Phase 4 | 1-2 | 8 weeks | All prior phases |

### Risk Mitigation

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Metal kernel complexity | High | Start with CUDA, port later |
| Feature scope creep | Medium | Strict phase gates |
| Performance targets | Medium | Monthly benchmarking |
| vLLM V1 moves fast | High | Track monthly releases |

### Success Metrics

- **Phase 1:** Match llama.cpp throughput on Metal (265+ t/s TinyLlama)
- **Phase 2:** 80% feature parity with vLLM
- **Phase 3:** Full OpenAI compatibility
- **Phase 4:** Beat vLLM on M-series GPU benchmarks

---

*Last updated: March 2026*
*See also: [Comparison with vLLM](./comparison.md)*
