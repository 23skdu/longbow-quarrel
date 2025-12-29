# Quarrel Improvement Roadmap

This document outlines the next 20 steps to improve the performance, correctness, and capabilities of the `Longbow-Quarrel` inference engine. Priorities are based on recent debugging findings and production readiness requirements.

## Priority 1: Correctness & Stability

### Step 1: Resolve Mistral Q4_K Precision Issues ⚠️ CRITICAL

- [ ] Test Mistral with F16 (non-quantized) weights to isolate Q4_K precision as root cause
- [ ] Compare layer-by-layer activations with llama.cpp reference implementation
- [ ] Implement activation clipping/scaling if needed (clip to ±20 after RMSNorm)
- [ ] Add numerical stability checks for extreme activation ranges
- [ ] Document Q4_K precision limitations for models with tiny embeddings

### Step 2: Fix Sequential KV Cache Overwrites

- [ ] Debug why `TestKVCacheSequential` shows cache corruption across positions
- [ ] Verify `store_kv_f16` kernel doesn't overwrite previous positions
- [ ] Add comprehensive KV cache integrity tests
- [ ] Validate cache behavior with multi-token prefill

### Step 3: Objective-C ARC Compliance

- [ ] Fix 6 `__bridge_retained`/`__bridge_transfer` warnings in `metal_backend.m`
- [ ] Enable ARC or remove unnecessary bridge casts
- [ ] Ensure proper memory management for Metal objects

### Step 4: Complete CPU Scan Safety

- [ ] Fix `ScanZeroes` and `ScanOCD` to handle DataTypeQ4K (currently incomplete)
- [ ] Add unit tests for all Scan* functions with Q4_K tensors
- [ ] Ensure no debug probes can crash on any data type

### Step 5: Q6_K Support Validation

- [ ] Verify Q6_K dequantization correctness (currently untested)
- [ ] Add unit tests for Q6_K embedding and linear kernels
- [ ] Test with Q6_K quantized models

## Priority 2: Performance Optimization

### Step 6: Fused Attention Kernels

- [ ] Profile current `AttScores`, `Softmax`, `AttValues` separately
- [ ] Extend `att_fused_f16` to support all sequence lengths (currently limited to <1024)
- [ ] Benchmark fused vs unfused attention performance
- [ ] Validate numerical correctness with edge cases

### Step 7: Flash Attention-2 Implementation

- [ ] Study Flash Attention-2 tiling and memory access patterns
- [ ] Implement forward pass tiling loop in Metal
- [ ] Optimize threadgroup memory usage for K/V blocks
- [ ] Test with long context sequences (>2048 tokens)
- [ ] Measure memory bandwidth improvement

### Step 8: SIMD Optimization in Softmax

- [ ] Verify `simd_sum` and `simd_max` correctness across all threadgroup sizes
- [ ] Ensure threadgroup size is always 32 for proper SIMD reduction
- [ ] Add numerical stability tests for extreme input ranges
- [ ] Profile softmax performance improvement

### Step 9: Multi-Token Prefill

- [ ] Refactor `Layer` to process batches of tokens in parallel
- [ ] Update Metal kernels to handle `(Batch, Seq, Head, Dim)` tensors
- [ ] Benchmark prompt processing speed (tokens/sec) for long prompts
- [ ] Verify correctness with batch sizes 1, 4, 8, 16

### Step 10: Zero-Allocation Inference Path

- [ ] Audit all allocations in hot path (Layer, Attention, FFN)
- [ ] Expand scratch buffer system to cover all temporary tensors
- [ ] Profile memory allocations during inference
- [ ] Achieve zero allocations per token in generation phase

## Priority 3: Advanced Features

### Step 11: KV Cache Quantization

- [ ] Implement 8-bit and 4-bit KV cache storage
- [ ] Update attention kernels to dequantize K/V on-the-fly
- [ ] Measure memory savings vs accuracy tradeoff
- [ ] Add configuration option for KV cache precision

### Step 12: Speculative Decoding

- [ ] Integrate draft model (SmolLM2-135M) alongside main model
- [ ] Implement draft generation loop (k steps)
- [ ] Implement verification step (k+1 tokens in parallel)
- [ ] Add rejection sampling logic
- [ ] Benchmark acceptance rate and wall-clock speedup

### Step 13: Continuous Batching

- [ ] Design request queue and scheduling loop
- [ ] Implement slot-based KV cache memory management
- [ ] Separate cache management from model execution
- [ ] Test with concurrent requests (simulated load)
- [ ] Measure throughput improvement

### Step 14: Paged Attention

- [ ] Implement BlockManager for physical vs virtual blocks
- [ ] Update attention kernels to use block tables
- [ ] Verify memory handling under high concurrency
- [ ] Test with fragmented memory scenarios

### Step 15: Sampling Optimizations

- [ ] Implement GPU-side `ArgMax`, `TopK`, `TopP` kernels
- [ ] Move sampling logic from CPU to GPU
- [ ] Only transfer final token ID to CPU
- [ ] Verify distribution matches CPU implementation
- [ ] Measure latency reduction

## Priority 4: Model Support & Quality

### Step 16: Additional Model Architectures

- [ ] Add Llama 3.x support (verify GQA, RoPE scaling)
- [ ] Add Qwen 2.5 support
- [ ] Add Gemma 2 support
- [ ] Create architecture-specific test suites
- [ ] Document model compatibility matrix

### Step 17: Mixed Precision Strategies

- [ ] Implement configurable precision per layer type
- [ ] Test F32 attention + F16 FFN combinations
- [ ] Measure accuracy vs performance tradeoffs
- [ ] Add precision configuration to model config

### Step 18: Quantization Improvements

- [ ] Implement Q3_K support (currently stubbed)
- [ ] Add Q5_K support
- [ ] Optimize Q4_K dequantization for tiny activations
- [ ] Add per-layer quantization configuration

### Step 19: Context Length Extensions

- [ ] Implement RoPE scaling for extended context
- [ ] Add ALiBi positional encoding support
- [ ] Test with 8K, 16K, 32K context lengths
- [ ] Optimize memory usage for long contexts

### Step 20: Production Readiness

- [ ] Add comprehensive error handling and recovery
- [ ] Implement request timeout and cancellation
- [ ] Add detailed performance metrics and logging
- [ ] Create production deployment guide
- [ ] Add health check and monitoring endpoints
- [ ] Implement graceful shutdown and resource cleanup

## Recent Accomplishments ✅

- Fixed Q4_K embedding lookup kernel (GPU-side dequantization)
- Fixed KV cache storage pipeline (wrong pipeline bug)
- Fixed CPU scan safety (DataTypeQ4K handling)
- Added comprehensive unit tests (RoPE, attention, KV cache)
- Verified Mistral architecture (pre-norm, GQA, attention scaling)
- Documented Mistral debugging findings and root cause hypothesis

## Notes

- Steps 1-5 are critical for correctness and should be prioritized
- Steps 6-10 focus on performance optimization
- Steps 11-15 enable advanced inference features
- Steps 16-20 expand model support and production readiness
- Each step should include unit tests and benchmarks
- Maintain backward compatibility where possible
