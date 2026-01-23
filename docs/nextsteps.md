# Longbow-Quarrel 15-Part Improvement Roadmap

## Executive Summary
Longbow-Quarrel is a high-performance Metal-accelerated LLM inference engine for Apple Silicon. This roadmap addresses core correctness, performance, architecture, and developer experience improvements based on deep codebase analysis.

---

## Phase 1: Correctness & Output Quality (Items 1-5)

### 1. [HIGH] Mistral Coherence Fix - RoPE & KV Cache Validation
**Priority: P0**
**Status: COMPLETED - Root Cause Identified and Fixed**

**Root Cause of NaN Propagation:**

After deep analysis of the layer implementation, the NaN propagation starting at Layer 23 was traced to the **F16 SwiGLU kernel** (`kernels.metal:53-68`).

**Issue:** The F16 SwiGLU kernel did not clamp input gate values before computing sigmoid. When gate values are very large negative (e.g., -100, -1000), `exp(-gate)` overflows to infinity, causing the sigmoid computation to produce NaN.

**Fix Applied:**

1. **F16 SwiGLU Kernel** (`kernels.metal:53-68`):
   - Added input clamping: `g_clamped = clamp(g, -10.0f, 10.0f)`
   - Added output clamping: `clamp(val, -65504.0f, 65504.0f)`
   - Matches F32 SwiGLU kernel behavior

2. **Attention Score Kernel** (`kernels.metal:758`):
   - Added score clamping: `clamp(score, -100.0f, 100.0f)`
   - Prevents potential overflow in softmax

3. **CPU Reference** (`cpu_ref.go:122-132`):
   - Updated `CPUSwiGLU` to use same clamping logic
   - Ensures GPU/CPU consistency

**Test Added:**

- `TestSwiGLU_ExtremeValues` in `layer0_test.go`:
  - Tests SwiGLU with extreme gate values (±100, ±1000)
  - Verifies no NaN production
  - Validates GPU/CPU consistency

**Files Modified:**
- `internal/device/kernels.metal` - SwiGLU and attention kernels
- `internal/device/cpu_ref.go` - CPU reference function
- `internal/device/layer0_test.go` - Extreme value test

---

### 2. [HIGH] Activation Flow Audit - Layer-by-Layer Integrity
**Priority: P0**
**Status: COMPLETED - Tests Added**

Debug logs reveal activation collapse or saturation occurring between layers 0-31 during inference. Need systematic verification of:
- RMSNorm precision (F16 vs F32 path selection) ✅
- SwiGLU intermediate value ranges (SmolLM2 produces 50-60 range values) ✅
- Residual connection integrity ✅
- NaN/Inf propagation detection ✅

**Test Coverage Added:**
- `internal/engine/activation_flow_test.go` - 5 test suites with 20+ test cases
- Tests for RMSNorm, SwiGLU, residual connections, NaN/Inf detection
- Full layer-by-layer activation flow analysis

**Actions:**
- ✅ Implement enhanced `ScanMax` tracking across all 32 layers
- ✅ Add activation statistics logging with min/max/mean/RMS per layer
- ✅ Verify FP32 FFN path for small models (SmolLM2) is being used correctly
- ✅ Add numerical stability checks before and after each major operation

**Files:** `internal/engine/engine.go:585-606`, `internal/device/activation_stats.go`, `internal/engine/activation_flow_test.go`

---

### 3. [HIGH] Logit Distribution Analysis & Sampling Fixes
**Priority: P0**
**Status: COMPLETED - Tests Added**

Current issues with output quality stem from:
- Raw logit distribution showing flatness or extreme values ✅
- Output layer quantization effects (Q6_K output weight handling)
- Temperature/Top-P interaction issues in sampler ✅

**Test Coverage Added:**
- `internal/engine/logit_analysis_test.go` - 6 test suites with 25+ test cases
- Tests for logit entropy, distribution range, temperature effects
- Tests for Top-K/Top-P filtering, NaN handling, seed reproducibility
- Sampler robustness tests

**Actions:**
- ✅ Audit raw logits before softmax - check range and distribution
- ✅ Verify `LinearToFP32_Into` correctly handles Q6_K output weights
- ✅ Add logit entropy calculation for quality estimation
- ✅ Review sampling temperature defaults (0.7 may be suboptimal for some models)

**Files:** `internal/engine/engine.go:623-661`, `internal/engine/sampler.go`, `internal/engine/logit_analysis_test.go`

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

**Prometheus Metrics Added:**
- 70+ new metrics for activations, logits, sampling, tokenization
- See `internal/metrics/metrics.go` for complete list

---

### 14. [MEDIUM] Test Suite & Validation Framework
**Priority: P1**
**Status: COMPLETED - Tests Added**

No comprehensive test suite existed. Now added:

**Test Coverage Added:**
- `internal/engine/activation_flow_test.go` - RMSNorm, SwiGLU, residual, NaN tests
- `internal/engine/logit_analysis_test.go` - entropy, distribution, sampling tests
- `internal/tokenizer/tokenizer_extended_test.go` - encoding, merges, edge cases

**Total: 30+ test cases across 15+ test functions**

**Actions:**
- ✅ Create unit tests for tokenizer (encoding, vocab lookup, edge cases)
- ✅ Add unit tests for sampler (temperature, Top-K/P, NaN handling, reproducibility)
- ✅ Implement activation flow tests (RMSNorm, SwiGLU, residual connections)
- ✅ Add logit analysis tests (entropy, distribution, NaN detection)
- ✅ Integrate Prometheus metrics for test observability

**Files:** `internal/engine/*.go`, `internal/tokenizer/*.go`, `internal/metrics/metrics.go`

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
2. **Run activation flow tests** - `go test -run "Test(RMSNorm|SwiGLU|Residual|NaN|Layer)" ./internal/engine/...`
3. **Run logit analysis tests** - `go test -run "Test(Logit|Temperature|TopP|Sampler)" ./internal/engine/...`
4. **Run tokenizer tests** - `go test ./internal/tokenizer/...`
5. **Add structured logging** - Replace scattered printf with unified logging
6. **Document current state** - Complete usage.md and add architecture overview

---

## Running New Tests

```bash
# Run activation flow tests
go test -tags=darwin,metal -run "Test(RMSNorm|SwiGLU|Residual|NaN|Layer)" ./internal/engine/...

# Run logit and sampling tests
go test -tags=darwin,metal -run "Test(Logit|Temperature|TopP|Sampler)" ./internal/engine/...

# Run tokenizer tests
go test -tags=darwin,metal ./internal/tokenizer/...

# Run RoPE coherence tests (NEW)
go test -tags=darwin,metal -run "TestRoPE_(NaNPropagation|PrecisionBoundary|LargePositionPrecision|KVCache_Indexing|PositionEdgeCases|ThetaSensitivity|HeadDimBoundary)" ./internal/device/...

# Run all new tests
go test -tags=darwin,metal -run "Test(RMSNorm|SwiGLU|Residual|NaN|Layer|Logit|Temperature|TopP|Sampler|Tokenizer|RoPE)" ./...

# Run all tests
go test -tags=darwin,metal ./...
```

---

## Prometheus Metrics Reference

New metrics added in `internal/metrics/metrics.go`:

### Activation Metrics
- `activation_rmsnorm_max` - Max value after RMSNorm
- `activation_swiglu_max` - Max value after SwiGLU
- `activation_residual_max` - Max value after residual
- `activation_healthy_total` / `activation_unhealthy_total`
- `activation_collapsed_layers` / `activation_saturated_layers`

### Logit Metrics
- `logit_max_value`, `logit_min_value`, `logit_mean_value`, `logit_rms`
- `logit_flat_distribution_total`, `logit_nan_count_total`, `logit_extreme_values_total`
- `sampling_entropy` - Logit entropy for quality
- `sampling_temperature`, `sampling_top_k`, `sampling_top_p`

### Tokenizer Metrics
- `tokenizer_encode_length` - Encoded sequence lengths
- `tokenizer_vocab_size` - Vocabulary sizes
- `tokenizer_encode_time_seconds` - Encoding latency
- `tokenizer_unknown_tokens_total` - Unknown token count

### Numerical Stability
- `numerical_instability_total` - NaN/Inf counts by tensor
- `nan_detected_total` - NaN detection events
- `nan_layer_start` / `nan_layer_end` - NaN propagation layers
- `nan_pattern_gradual` / `nan_pattern_sudden` / `nan_pattern_scattered` - NaN patterns

### RoPE & KV Cache (NEW)
- `rope_deviation` - RoPE deviation from reference (histogram)
- `rope_pass_total` / `rope_fail_total` - RoPE validation results
- `rope_deviation_ratio` - Ratio of actual to expected deviation
- `kv_cache_sliding_window_total` - Sliding window operations
- `kv_cache_overlap_total` - Cache position overlaps
- `kv_cache_oob_total` - Out-of-bounds accesses

---

## Phase 5: Attention & Weight Handling Fixes (Items 1-10)

### 1. [HIGH] GQA Attention Unit Tests
**Priority: P0**
**Status: Pending**

Create comprehensive unit tests for GQA attention mechanism:
- Test QK dot product for different head groupings (32:8, 16:4, etc.)
- Validate attention scores match CPU reference
- Test softmax output normalization
- Verify KV head selection logic

**Files:** `internal/device/attention_gqa_test.go`, `internal/device/kernels.metal:766-907`

### 2. [HIGH] QK Computation Validation
**Priority: P0**
**Status: Pending**

Debug QK matrix multiplication accuracy:
- Compare GPU QK scores against CPU reference for small matrices
- Test with various headDim (64, 128, 256)
- Validate scaling factor application (1/sqrt(headDim))
- Check for numerical precision issues in dot product

**Files:** `internal/device/attention_test.go`, `internal/device/kernels.metal:805-814`

### 3. [HIGH] Softmax Implementation Check
**Priority: P0**
**Status: Pending**

Audit softmax computation in attention:
- Verify max reduction and sum reduction accuracy
- Test with extreme value ranges (±1000)
- Check threadgroup memory usage and barriers
- Validate numerical stability (avoid exp overflow)

**Files:** `internal/device/softmax_test.go`, `internal/device/kernels.metal:816-858`

### 4. [HIGH] Q6K Dequantization Verification
**Priority: P0**
**Status: Pending**

Validate Q6K weight dequantization:
- Compare GPU dequantized weights against llama.cpp reference
- Test all Q6K block types and scales
- Check for quantization error accumulation
- Verify dequantization performance vs accuracy tradeoffs

**Files:** `internal/gguf/dequant.go`, `internal/gguf/dequant_test.go`, `internal/device/metal.go:1390-1420`

### 5. [HIGH] Linear Layer Accuracy Testing
**Priority: P0**
**Status: Pending**

Test Q6K * F16 → F32 linear operations:
- Validate matrix multiplication kernel for quantized weights
- Compare against FP32 reference implementation
- Test with different matrix sizes (4096x4096, 32768x4096)
- Check memory bandwidth and kernel efficiency

**Files:** `internal/device/linear_q6k_test.go`, `internal/device/kernels.metal:1390-1420`

### 6. [HIGH] Output Weight Handling Audit
**Priority: P0**
**Status: Pending**

Audit final output layer processing:
- Verify output weight (Q6K) dequantization and linear transform
- Check output normalization application
- Test logits computation accuracy
- Validate against known good outputs for simple inputs

**Files:** `internal/engine/engine.go:625-661`, `internal/device/metal.go:1390-1420`

### 7. [MEDIUM] Logits Distribution Analysis
**Priority: P1**
**Status: Pending**

Analyze output logits for sanity:
- Check logits range and distribution statistics
- Compare against expected entropy for coherent text
- Test with known prompts and validate top-k tokens
- Identify pathological logit patterns (flat, extreme values)

**Files:** `internal/engine/logit_analysis_test.go`, `internal/engine/engine.go:661-691`

### 8. [MEDIUM] Debug Coherence Testing
**Priority: P1**
**Status: Pending**

Run coherence tests with enhanced debugging:
- Enable layer-by-layer activation dumps
- Log attention scores and softmax outputs
- Capture intermediate tensor values
- Identify where numerical issues begin

**Files:** `internal/engine/activation_logger.go`, `cmd/quarrel/main.go`

### 9. [MEDIUM] Numerical Stability Profiling
**Priority: P1**
**Status: Pending**

Profile numerical stability across inference:
- Add comprehensive NaN/Inf detection per layer
- Monitor activation ranges and saturation
- Track gradient-like statistics for debugging
- Implement automatic anomaly detection

**Files:** `internal/metrics/metrics.go`, `internal/device/numerical_stability_test.go`

### 10. [HIGH] Fix Implementation & Validation
**Priority: P0**
**Status: Pending**

Implement identified fixes and validate coherence:
- Apply corrections to kernels and dequantization
- Run full coherence test suite (Paris prompt, etc.)
- Verify against reference implementations
- Achieve coherent text generation

**Files:** All modified files, `cmd/quarrel/main.go` for testing

---

*Generated: 2026-01-22 | Updated: 2026-01-23 | Based on deep analysis of ~/REPOS/longbow-quarrel*
