# Longbow-Quarrel 15-Part Improvement Roadmap

## Executive Summary
Longbow-Quarrel is a high-performance Metal-accelerated LLM inference engine for Apple Silicon. This roadmap addresses core correctness, performance, architecture, and developer experience improvements based on deep codebase analysis.

---

## Phase 1: Correctness & Output Quality (Items 1-5)

### 1. [HIGH] Mistral Coherence Fix - RoPE & KV Cache Validation
**Priority: P0**
**Status: In Progress**

The Mistral model currently produces incoherent outputs due to:
- RoPE implementation verification needed (`rope_f16` kernel calling pattern)
- KV cache indexing with sliding window (4096) requires validation
- Position encoding for grouped query attention (GQA) ratio handling

**Actions:**
- Verify `rope_f16` is called correctly for Q and K projections at each position
- Validate `CachePos` modulo arithmetic for sliding window attention
- Audit GQA ratio `heads/kv_heads` in attention score computation
- Test with simple prompts ("The capital of France is") and verify coherent completion

**Files:** `internal/device/kernels.metal:586-619`, `internal/engine/engine.go:815-868`

---

### 2. [HIGH] Activation Flow Audit - Layer-by-Layer Integrity
**Priority: P0**
**Status: Pending**

Debug logs reveal activation collapse or saturation occurring between layers 0-31 during inference. Need systematic verification of:
- RMSNorm precision (F16 vs F32 path selection)
- SwiGLU intermediate value ranges (SmolLM2 produces 50-60 range values)
- Residual connection integrity
- NaN/Inf propagation detection

**Actions:**
- Implement enhanced `ScanMax` tracking across all 32 layers
- Add activation statistics logging with min/max/mean/RMS per layer
- Verify FP32 FFN path for small models (SmolLM2) is being used correctly
- Add numerical stability checks before and after each major operation

**Files:** `internal/engine/engine.go:585-606`, `internal/device/activation_stats.go`

---

### 3. [HIGH] Logit Distribution Analysis & Sampling Fixes
**Priority: P0**
**Status: Pending**

Current issues with output quality stem from:
- Raw logit distribution showing flatness or extreme values
- Output layer quantization effects (Q6_K output weight handling)
- Temperature/Top-P interaction issues in sampler

**Actions:**
- Audit raw logits before softmax - check range and distribution
- Verify `LinearToFP32_Into` correctly handles Q6_K output weights
- Add logit entropy calculation for quality estimation
- Review sampling temperature defaults (0.7 may be suboptimal for some models)

**Files:** `internal/engine/engine.go:623-661`, `internal/engine/sampler.go`

---

### 4. [MEDIUM] Quantization Accuracy Verification
**Priority: P1**
**Status: Partial**

K-quantization (Q3_K, Q4_K, Q6_K) needs end-to-end accuracy verification:
- CPU-side dequantization comparison with reference outputs
- GPU kernel quantization accuracy
- Per-channel vs per-block scale handling

**Actions:**
- Compare dequantized weights against llama.cpp reference for each quantization type
- Verify Q4_K minimum scale (dmin) handling in `linear_q4k_f16`
- Add quantization error metrics to validation suite
- Test with smaller models (SmolLM2) where quantization effects are more visible

**Files:** `internal/gguf/dequant.go`, `internal/device/kernels.metal:120-401`

---

### 5. [MEDIUM] Tensor Type Promotion Consistency
**Priority: P1**
**Status: Pending**

Norm weights are promoted from F16 to FP32 for precision, but:
- Inconsistent handling across different code paths
- Some paths bypass promotion and use F16 directly
- Output weight sharing with token embedding needs validation

**Actions:**
- Audit all norm weight usages to verify FP32 promotion is applied
- Add validation checks for tensor type compatibility
- Ensure consistent dtype handling in fused kernels
- Test with both F16 and FP32 norm weight paths

**Files:** `internal/engine/engine.go:307-322`

---

## Phase 2: Performance Optimization (Items 6-9)

### 6. [HIGH] Kernel Fusion & Memory Optimization
**Priority: P1**
**Status: In Progress**

Current architecture has many small kernel dispatches. Fusion opportunities:
- Fused RoPE + StoreKV for reduced memory bandwidth
- Combined QKV projection (already exists as `RMSNormQKV`)
- FFN fusion improvements for K-quantized weights

**Actions:**
- Profile kernel dispatch overhead with Metal instruments
- Implement fused RoPE+StoreKV kernel for single-token processing
- Optimize `att_fused_f16` kernel for longer context windows
- Reduce scratch buffer allocation frequency

**Files:** `internal/device/kernels.metal:761-907`, `internal/device/metal.go:1027-1323`

---

### 7. [MEDIUM] KV Cache Optimization
**Priority: P1**
**Status: Partial**

Sliding window attention implementation needs tuning:
- Cache indexing modulo arithmetic efficiency
- Memory access patterns for rolling buffer
- Prefill vs decode phase cache handling

**Actions:**
- Benchmark KV cache access patterns for different window sizes
- Optimize cache_idx calculation (currently `t % windowSize`)
- Consider cache prefetching for sequential token generation
- Add cache hit/miss metrics

**Files:** `internal/engine/engine.go:833-868`, `internal/device/kernels.metal:735-758`

---

### 8. [MEDIUM] Memory Pool & Allocation Strategy
**Priority: P2**
**Status: Partial**

Tensor pooling exists but could be improved:
- Heap-backed scratch buffers may leak (comment in code)
- Finalizer-based cleanup is unreliable for Metal buffers
- Memory budget tracking could be more aggressive

**Actions:**
- Implement explicit LayerScratch cleanup pattern
- Add memory pressure callbacks for pool clearing
- Profile memory allocation patterns during inference
- Consider memory-mapped weights for very large models

**Files:** `internal/device/metal.go:869-1025`, `internal/engine/engine.go:542-545`

---

### 9. [LOW] Thread Configuration & GPU Utilization
**Priority: P2**
**Status: Pending**

Metal kernel threadgroup configuration may not be optimal:
- Current 32-thread SIMD groups may underutilize GPU
- Grid dimensions could be tuned for different model sizes
- Attention kernel thread count requires tuning

**Actions:**
- Profile GPU utilization with Metal debugger
- Test different threadgroup sizes (64, 128, 256 threads)
- Optimize grid launch dimensions for each kernel
- Add GPU utilization metrics

**Files:** `internal/device/kernels.metal` (various kernel launches)

---

## Phase 3: Architecture & Extensibility (Items 10-12)

### 10. [MEDIUM] Architecture Abstraction Layer
**Priority: P1**
**Status: Pending**

Current code tightly couples Llama/Mistral/SmolLM handling:
- Architecture-specific heuristics scattered throughout
- Sliding window detection via RopeTheta heuristic
- Hardcoded attention patterns per model type

**Actions:**
- Create architecture descriptor struct with model-specific parameters
- Abstract RoPE, attention, and FFN patterns per architecture
- Support YAML/json model configuration files
- Enable easier addition of new architectures (Qwen, Gemma, etc.)

**Files:** `internal/engine/types.go:20-36`, `internal/engine/engine.go:143-172`

---

### 11. [MEDIUM] Multi-Format Support Expansion
**Priority: P2**
**Status: Pending**

Current GGUF reader could support more quantization types:
- Q2_K, Q5_K, Q8_0 quantization formats
- GPTQ and AWQ converted models (if GGUF-compatible)
- Mixed quantization within single model

**Actions:**
- Add Q2_K and Q5_K kernel support
- Implement weight format detection and routing
- Add tensor type validation during model load
- Support for model-specific weight arrangements

**Files:** `internal/gguf/structs.go`, `internal/device/metal.go:85-94`

---

### 12. [LOW] Batching & Concurrent Inference
**Priority: P3**
**Status: Not Started**

Current implementation is single-sequence only:
- No batch processing support
- No KV cache sharing across requests
- No request queueing

**Actions:**
- Design batched inference API (dynamic batching)
- Implement KV cache sharing for concurrent requests
- Add request scheduling with priority support
- Profile batch efficiency vs latency tradeoffs

**Files:** `internal/engine/engine.go:494-813` (main inference loop)

---

## Phase 4: Developer Experience (Items 13-15)

### 13. [MEDIUM] Debugging & Observability Improvements
**Priority: P1**
**Status: In Progress**

Current debugging is ad-hoc:
- Multiple debug log files with different formats
- No structured logging system
- Activation dumping is experimental

**Actions:**
- Implement structured logging with levels (DEBUG, INFO, WARN, ERROR)
- Create unified debug artifact format (JSON lines)
- Add web-based visualization for activation patterns
- Integrate with existing Prometheus metrics

**Files:** `internal/engine/activation_logger.go`, `internal/metrics/metrics.go`

---

### 14. [MEDIUM] Test Suite & Validation Framework
**Priority: P1**
**Status: Pending**

No comprehensive test suite exists:
- Only manual verification against llama.cpp
- No unit tests for critical path functions
- No golden output validation

**Actions:**
- Create unit tests for tokenizer, sampler, GGUF parsing
- Add golden output tests for key model architectures
- Implement automatic correctness regression detection
- Add benchmark comparison against reference implementations

**Files:** `cmd/` (various verification utilities)

---

### 15. [LOW] Documentation & Examples
**Priority: P2**
**Status: Partial**

Documentation exists but needs expansion:
- Usage guide incomplete (ends abruptly)
- No API documentation for internal packages
- Missing architecture diagrams

**Actions:**
- Complete usage.md documentation
- Add architecture overview with data flow diagrams
- Create example applications (chat CLI, API server)
- Document kernel API for custom extension

**Files:** `docs/`, `README.md`

---

## Priority Matrix

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| P0 | 1. Mistral Coherence Fix | High | Critical |
| P0 | 2. Activation Audit | Medium | High |
| P0 | 3. Logit Analysis | Medium | High |
| P1 | 4. Quantization Verification | Medium | High |
| P1 | 5. Tensor Type Consistency | Low | Medium |
| P1 | 6. Kernel Fusion | High | High |
| P1 | 7. KV Cache Optimization | Medium | Medium |
| P1 | 13. Observability | Medium | Medium |
| P1 | 14. Test Suite | High | High |
| P2 | 8. Memory Pool | Medium | Medium |
| P2 | 9. Thread Tuning | Low | Low |
| P2 | 10. Architecture Abstraction | High | High |
| P2 | 11. Multi-Format Support | Medium | Medium |
| P2 | 12. Batching | High | Medium |
| P2 | 15. Documentation | Low | Medium |

---

## Immediate Actions (Next Sprint)

1. **Complete Mistral coherence debugging** - Focus on RoPE and KV cache integration
2. **Add structured logging** - Replace scattered printf with unified logging
3. **Implement activation scan across all layers** - Enable full visibility into inference
4. **Create minimal test suite** - Verify tokenizer and sampler correctness
5. **Document current state** - Complete usage.md and add architecture overview

---

*Generated: 2026-01-22 | Based on deep analysis of ~/REPOS/longbow-quarrel*
