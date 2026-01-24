# Longbow-Quarrel 15-Part Improvement Roadmap

## Executive Summary
Longbow-Quarrel is a high-performance Metal-accelerated LLM inference engine for Apple Silicon. This roadmap addresses core correctness, performance, architecture, and developer experience improvements based on deep codebase analysis.

---

## Phase 1: Correctness & Output Quality (Items 1-5)

### 1. [HIGH] Mistral Coherence Fix - RoPE & KV Cache Validation
**Priority: P0**
**Status: COMPLETED - Root Cause Identified and Fixed**

### 2. [HIGH] Activation Flow Audit - Layer-by-Layer Integrity
**Priority: P0**
**Status: COMPLETED - Tests Added**

### 3. [HIGH] Logit Distribution Analysis & Sampling Fixes
**Priority: P0**
**Status: COMPLETED - Tests Added**

### 4. [MEDIUM] Quantization Accuracy Verification
**Priority: P1**
**Status: Partial**

### 5. [MEDIUM] Tensor Type Promotion Consistency
**Priority: P1**
**Status: Pending**

## Phase 2: Performance Optimization (Items 6-9)

### 6. [HIGH] Kernel Fusion & Memory Optimization
**Priority: P1**
**Status: COMPLETED - Q4K Embedding Kernel Optimized**

### 7. [MEDIUM] KV Cache Optimization
**Priority: P1**
**Status: Partial**

### 8. [MEDIUM] Memory Pool & Allocation Strategy
**Priority: P2**
**Status: Partial**

### 9. [LOW] Thread Configuration & GPU Utilization
**Priority: P2**
**Status: Pending**

## Phase 3: Architecture & Extensibility (Items 10-12)

### 10. [MEDIUM] Architecture Abstraction Layer
**Priority: P1**
**Status: Pending**

### 11. [MEDIUM] Multi-Format Support Expansion
**Priority: P2**
**Status: Pending**

### 12. [LOW] Batching & Concurrent Inference
**Priority: P3**
**Status: Not Started**

## Phase 4: Developer Experience (Items 13-15)

### 13. [MEDIUM] Debugging & Observability Improvements
**Priority: P1**
**Status: In Progress**

### 14. [MEDIUM] Test Suite & Validation Framework
**Priority: P1**
**Status: COMPLETED - Tests Added**

### 15. [LOW] Documentation & Examples
**Priority: P2**
**Status: Partial**

---
## Phase 5: Attention & Weight Handling Fixes (Items 1-10)

### 1. [HIGH] GQA Attention Unit Tests
**Priority: P0**
**Status: COMPLETED**

### 2. [HIGH] QK Computation Validation
**Priority: P0**
**Status: Pending**

### 3. [HIGH] Softmax Implementation Check
**Priority: P0**
**Status: Pending**

### 4. [HIGH] Q6K Dequantization Verification
**Priority: P0**
**Status: Pending**

### 5. [HIGH] Linear Layer Accuracy Testing
**Priority: P0**
**Status: Pending**

### 6. [HIGH] Output Weight Handling Audit
**Priority: P0**
**Status: Pending**

### 7. [MEDIUM] Logits Distribution Analysis
**Priority: P1**
**Status: Pending**

### 8. [MEDIUM] Debug Coherence Testing
**Priority: P1**
**Status: Pending**

### 9. [MEDIUM] Numerical Stability Profiling
**Priority: P1**
**Status: Pending**

### 10. [HIGH] Fix Implementation & Validation
**Priority: P0**
**Status: Pending**
---
## Immediate Actions (Next Sprint)

1. **Complete Mistral coherence debugging** - Focus on RoPE and KV cache integration
2. **Run activation flow tests** - `go test -run "Test(RMSNorm|SwiGLU|Residual|NaN|Layer)" ./internal/engine/...`
3. **Run logit analysis tests** - `go test -run "Test(Logit|Temperature|TopP|Sampler)" ./internal/engine/...`
4. **Run tokenizer tests** - `go test ./internal/tokenizer/...`
5. **Add structured logging** - Replace scattered printf with unified logging
6. **Document current state** - Complete usage.md and add architecture overview
7. **Update GQA testing documentation** - Mark GQA testing as complete in docs/nextsteps.md


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
