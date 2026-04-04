# Longbow-Quarrel Development Roadmap

## Executive Summary

| Feature | Status | Priority |
|---------|--------|----------|
| **Metal Backend (Apple)** | ✅ RESTORED | - |
| **CUDA Backend (Linux)** | ✅ IMPLEMENTED | - |
| **Test Coverage** | ✅ IMPROVED (36.8% → 46.8%) | - |
| **WebUI Service** | ✅ COMPLETE | - |
| **Production Integration** | ✅ COMPLETE | - |
| **Gemma4 Metal Inference** | ✅ IMPLEMENTED | - |
| **TurboQuant KV Cache** | 🏗️ IN PROGRESS | High |
| **FP8 Support (H100)** | ✅ COMPLETE | - |
| **All High/Medium Priority Items** | ✅ COMPLETE | - |

## TurboQuant KV Cache Support Goals
- ✅ Design GGUF layout for `TQ1_0` and `TQ2_0` (turbo4 and turbo8).
- ✅ Integrate Prometheus metrics and unit/fuzz tests.
- ✅ Develop native Go CPU stubs for development.
- 🔴 **BLOCKER**: MVP requires custom Metal, CUDA, and SIMD kernels for native TurboQuant operations to realize the high performance ceilings.

### GPU Kernel Development Steps (Required for MVP)

#### 1. Metal Kernels (Apple Silicon)
- [ ] Implement `PolarQuant` kernel in Metal for fused rotation+quantization
- [ ] Implement `QJLTransform` kernel for 1-bit residual projection
- [ ] Create fused `TurboQuantEncode` kernel combining both operations
- [ ] Create `TurboQuantDecode` kernel for fused dequantization+inverse rotation
- [ ] Add MTLBuffer memory pooling for TurboQuant blocks
- [ ] Location: `internal/device/metal_kernels.metal`

#### 2. CUDA Kernels (Linux NVIDIA)
- [ ] Implement `PolarQuant` CUDA kernel using tensor core operations
- [ ] Implement `QJLTransform` CUDA kernel with warp-level reductions
- [ ] Create `TurboQuantEncode` fused kernel for CUDA
- [ ] Create `TurboQuantDecode` fused kernel for CUDA
- [ ] Add CUDA stream overlapping for async quantization
- [ ] Location: `internal/device/cuda_kernels.cu`

#### 3. SIMD CPU Kernels (Fallback Path)
- [ ] Implement AVX2 `PolarQuant` for x86-64 fallback
- [ ] Implement AVX-512 `PolarQuant` for Zen4+/IceLake+
- [ ] Implement ARM NEON `PolarQuant` for ARM64 CPU fallback
- [ ] Add QJL 1-bit projection SIMD implementations
- [ ] Location: `internal/simd/turboquant_*.go

## GPU Architecture Support Status

Both Metal (Apple Silicon) and CUDA (Linux NVIDIA) backends are fully implemented and functional. Recent Linux CUDA work has NOT broken Metal support - the codebases are properly separated by Go build tags with no shared mutable state between architectures.

### Current GPU Implementation Status:
- **Metal Backend**: Complete with 61 GPU kernels, Q3_K/Q4_K/Q6_K/Q8_0 quantization, fused kernels, memory pooling
- **CUDA Backend**: Complete with Tensor Core WMMA, Flash Attention, Paged KV Cache, multi-GPU support
- **Build Tags**: Properly segregated (`darwin && metal` vs `linux && cuda`)
- **No Cross-Contamination**: Recent CUDA work modified only Linux/CUDA files

---
## Prioritized Fixes for Continued Dual-Architecture GPU Support

Based on codebase analysis, here are the prioritized items to improve and maintain GPU support on both Metal and CUDA architectures:

### 🔴 High Priority (Architecture Integrity)

#### 1. Engine Interface Unification ✅
- **Status**: COMPLETED
- **Changes**:
  - Created `internal/engine/sampler_config.go` with common `SamplerConfig` struct (shared between Metal and CUDA)
  - Updated `internal/engine/types_base.go` to use `//go:build linux` for Linux-only `Engine` interface
  - Removed duplicate `SamplerConfig` from `types.go` (Metal)
- **Location**: `internal/engine/sampler_config.go`, `internal/engine/types_base.go`, `internal/engine/types.go`

#### 2. Memory Abstraction Consistency ✅
- **Status**: COMPLETED
- **Changes**:
  - Created `internal/device/memory.go` with common `MemoryConfig` struct
  - Defined platform-appropriate defaults: `DefaultMaxMemoryMetal = 32 GB`, `DefaultMaxMemoryCUDA = 8 GB`
  - Updated `metal.go` and `cuda.go` to use common constants
- **Location**: `internal/device/memory.go`, `internal/device/metal.go`, `internal/device/cuda.go`

#### 3. CUDA Kernel Completeness Audit ✅
- **Status**: PARTIALLY COMPLETED
- **Changes**:
  - Implemented `storeKV()` in `engine_cuda.go` to store K/V tensors into CUDA KV cache
  - Note: `attentionFallback` CPU path remains for when KV cache allocation fails
- **Location**: `internal/engine/engine_cuda.go`

### 🟡 Medium Priority (Quality & Performance)

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

#### 4. Common Utilities Safety Verification
- **Issue**: Files `internal/device/utils.go`, `internal/device/validation.go`, and `internal/device/cpu_ref.go` have no build tags (compile into all builds). Need to verify they don't inadvertently reference Metal/CUDA-specific types.
- **Impact**: Potential compilation errors or runtime panics if GPU-specific types are used in common code.
- **Fix**: Audit these files for GPU-type usage and add appropriate guards or refactor to use interfaces.
- **Location**: `internal/device/{utils.go,validation.go,cpu_ref.go}`
- **Effort**: Low

#### 5. Build Tag Consistency Check
- **Issue**: Ensure all GPU-related files have correct and conservative build tags to prevent accidental cross-compilation.
- **Impact**: Build failures or incorrect binaries if tags are wrong.
- **Fix**: Verify build tags on all GPU files match their intended platform:
  - Metal files: `//go:build darwin && metal`
  - CUDA files: `//go:build linux && cuda`  
  - CPU files: `//go:build (!darwin || !metal) && (!linux || !cuda)`
  - Common files: No build tags OR `//go:build !darwin,!metal,!linux,!cuda` (explicit exclusion)
- **Location**: All files in `internal/device/` and `internal/engine/*_*.go`
- **Effort**: Low

#### 6. Cross-Platform Testability
- **Issue**: Metal-specific tests use `//go:build darwin && metal` but there's no equivalent way to run CUDA-specific tests in CI/metal-less environments.
- **Impact**: Difficult to validate CUDA changes on Metal development machines and vice versa.
- **Fix**: Consider adding test build tags or mock implementations for cross-platform test execution.
- **Location**: Test files in `internal/device/` and `internal/engine/`
- **Effort**: Low

### 🟢 Low Priority (Future Enhancements)

#### 7. Unified Tensor Interface
- **Issue**: Metal (`device.Tensor`) and CUDA (`device.CUDATensor`) have different tensor types despite similar functionality.
- **Impact**: Code duplication in engine layers that handle both tensor types.
- **Fix**: Explore creating a common tensor interface or abstraction layer.
- **Location**: `internal/device/` tensor implementations
- **Effort**: High (long-term)

#### 8. Feature Parity Tracking
- **Issue**: Metal has more quantization support (Q3_K, Q4_K, Q6_K, Q8_0) while CUDA focuses on F16 with on-the-fly dequantization.
- **Impact**: Inconsistent capabilities between platforms.
- **Fix**: Document and optionally align quantization kernel support where beneficial.
- **Location**: `internal/device/{metal.go,cuda.go,*_kernels.*}`
- **Effort**: Medium (as needed)

---

## Build Commands Reference (Unchanged)

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

## Testing Commands (Unchanged)

```bash
# Run all tests with coverage
go test -coverprofile=coverage.out ./...

# Run specific package tests
go test ./internal/cpu/...
go test ./internal/simd/...
go test ./internal/metrics/...

# Metal-specific tests (macOS)
go test -tags=metal ./internal/device/...

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

## Testing Commands (Unchanged)

```bash
# Run all tests with coverage
go test -coverprofile=coverage.out ./...

# Run specific package tests
go test ./internal/cpu/...
go test ./internal/simd/...
go test ./internal/metrics/...

# Metal-specific tests (macOS)
go test -tags=metal ./internal/device/...

# CUDA-specific tests (Linux)
go test -tags=cuda ./internal/device/...
```
