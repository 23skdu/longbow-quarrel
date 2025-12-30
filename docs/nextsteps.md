# Quarrel Improvement Roadmap

This document outlines the next steps to improve the performance, correctness, and capabilities of the `Longbow-Quarrel` inference engine.

## Immediate Next 10 Steps (Priority Order)

### Step 1: Debug Mistral Attention Output Magnitude ⚠️ CRITICAL

**Status**: Blocking coherent text generation

- [ ] Investigate why attention outputs are extremely small (~0.07-0.25 max)
- [ ] Compare attention scores distribution with reference implementation
- [ ] Verify attention projection weights are being applied correctly
- [ ] Check if attention output scaling factor is missing
- [ ] Add layer-by-layer comparison with llama.cpp for same input

**Why**: Attention outputs are 10-100x smaller than expected, suggesting a fundamental issue in the attention mechanism that's causing downstream layers to receive weak signals.

### Step 2: Validate Output Head Linear Transformation

**Status**: Suspected numerical precision issue

- [ ] Verify `Metal_LinearF16ToF32` kernel correctness with unit tests
- [ ] Compare output head weights between Q4K and F16 versions
- [ ] Check if logit scaling is applied correctly (temperature, etc.)
- [ ] Validate that F16->F32 conversion preserves precision
- [ ] Test with known-good input activations to isolate kernel vs upstream issues

**Why**: The output head is the final transformation before token selection. Even small errors here can cause completely wrong token predictions.

### Step 3: Fix LinearF32_Into Weight Offset Bug

**Status**: Critical bug found during debugging

- [ ] Add `C.int(weight.Offset)` to `Metal_MatMul_Q4K_F32` call in `LinearF32_Into`
- [ ] Add `C.int(weight.Offset)` to `Metal_MatMul_F16_F32_F32` call in `LinearF32_Into`
- [ ] Add unit test to verify F32 linear transformations use correct weight offsets
- [ ] Test SmolLM2 inference to ensure F32 path still works correctly

**Why**: The same offset bug that affected F16 paths also exists in the F32 linear transformation path used by SmolLM2.

### Step 4: Implement Comprehensive Kernel Unit Tests

**Status**: Essential for debugging and regression prevention

- [ ] Create reference outputs from llama.cpp for each kernel type
- [ ] Add unit tests for `linear_q4k_f16`, `linear_f16`, `linear_f16_f32`
- [ ] Add unit tests for `att_scores_f16`, `att_softmax_f16`, `att_values_f16`
- [ ] Add unit tests for `rope_f16` with various theta values
- [ ] Add unit tests for `rmsnorm_f32_f16` and `rmsnorm_f32`

**Why**: Without kernel-level unit tests, it's impossible to isolate whether bugs are in kernels, weight loading, or inference logic.

### Step 5: Add Layer-by-Layer Activation Logging

**Status**: Needed for systematic debugging

- [ ] Implement configurable activation logging (env var or flag)
- [ ] Log Q, K, V projections before and after RoPE
- [ ] Log attention scores, softmax outputs, and attention results
- [ ] Log FFN gate/up activations and final outputs
- [ ] Create comparison script to diff against llama.cpp outputs

**Why**: Systematic layer-by-layer comparison is the only way to pinpoint exactly where the inference diverges from reference.

### Step 6: Optimize Fused Attention Kernel

**Status**: Performance improvement opportunity

- [ ] Remove the `p < -1` debug condition and re-enable fused path
- [ ] Profile fused vs unfused attention performance
- [ ] Extend fused kernel to support all sequence lengths (currently disabled)
- [ ] Verify numerical correctness matches unfused path
- [ ] Benchmark memory bandwidth improvement

**Why**: Fused attention can provide 2-3x speedup by reducing memory traffic, but it's currently disabled for debugging.

### Step 7: Implement Zero-Allocation Inference

**Status**: Performance and stability improvement

- [ ] Audit all allocations in hot path (use profiler)
- [ ] Expand scratch buffer system to cover all temporary tensors
- [ ] Pre-allocate all buffers during model initialization
- [ ] Verify no allocations occur during token generation phase
- [ ] Measure latency improvement from reduced allocation overhead

**Why**: Allocations during inference cause unpredictable latency spikes and potential memory fragmentation.

### Step 8: Add Multi-Token Prefill

**Status**: Major performance improvement for long prompts

- [ ] Refactor `Layer` to accept batch dimension for parallel token processing
- [ ] Update Metal kernels to handle `(Batch, Seq, Head, Dim)` tensors
- [ ] Implement parallel KV cache storage for multiple positions
- [ ] Benchmark prompt processing speed (tokens/sec) improvement
- [ ] Verify correctness with various batch sizes (1, 4, 8, 16)

**Why**: Current single-token prefill is extremely slow for long prompts. Batched prefill can provide 10-50x speedup.

### Step 9: Implement KV Cache Quantization

**Status**: Memory optimization for long contexts

- [ ] Add 8-bit and 4-bit KV cache storage options
- [ ] Update attention kernels to dequantize K/V on-the-fly
- [ ] Measure memory savings (expect 2-4x reduction)
- [ ] Benchmark accuracy vs memory tradeoff
- [ ] Add configuration option for KV cache precision

**Why**: KV cache is the primary memory bottleneck for long contexts. Quantization can enable 2-4x longer contexts with minimal accuracy loss.

### Step 10: Add Llama 3.x and Qwen 2.5 Support

**Status**: Expand model compatibility

- [ ] Verify GQA (Grouped Query Attention) implementation
- [ ] Add RoPE scaling support for extended context
- [ ] Test with Llama 3.1 8B and Qwen 2.5 7B models
- [ ] Create architecture-specific test suites
- [ ] Document model compatibility matrix

**Why**: Supporting popular open models increases the utility and adoption of Quarrel.

---

## Recent Accomplishments ✅

- Fixed critical tensor offset bug affecting all linear transformations
- Fixed Q4_K embedding lookup kernel (GPU-side dequantization)
- Fixed RoPE convention mismatch (Adjacent -> Half-Half for GGUF compatibility)
- Fixed KV cache storage heap corruption (added buffer offsets)
- Fixed prompt-token overwrite bug in inference loop
- Fixed head indexing in `att_fused_f16` kernel
- Fixed value cache mix-up in fused attention kernel
- Verified Mistral architecture activation stability
- Added comprehensive logit debugging (top-5 candidates, statistics)
- Added KV cache integrity checks during prefill

## Long-Term Roadmap (Steps 11-20)

1. **Flash Attention-2 Implementation** - Tiled attention for memory efficiency
2. **Speculative Decoding** - Draft model + verification for 2-3x speedup
3. **Continuous Batching** - Request queue and slot-based KV cache management
4. **Paged Attention** - Virtual memory for KV cache to handle fragmentation
5. **GPU-Side Sampling** - Move ArgMax/TopK/TopP to GPU to reduce latency
6. **Mixed Precision Strategies** - Per-layer precision configuration
7. **Q3_K and Q5_K Support** - Additional quantization formats
8. **Context Length Extensions** - RoPE scaling and ALiBi support
9. **Production Monitoring** - Metrics, health checks, graceful shutdown
10. **Comprehensive Documentation** - Deployment guide, API reference, examples

## Notes

- Steps 1-5 are **critical for correctness** and block all other work
- Steps 6-10 focus on **performance and capabilities**
- Steps 11-20 are **advanced features** for production deployment
- Each step should include unit tests and benchmarks
- Maintain backward compatibility where possible
