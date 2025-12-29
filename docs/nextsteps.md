# Performance Optimization Roadmap

This document outlines the next 10 steps to improve the performance, efficiency, and scalability of the `Longbow-Quarrel` inference engine.
Each step is broken down into granular sub-tasks to ensure steady progress and avoid complexity overload.

## Step 1: Fused Attention Kernels

- [ ] Analysis & Profiling
  - [ ] Profile current `AttScores`, `Softmax`, and `AttValues` kernels separately to establish a baseline.
  - [ ] Identify memory bandwidth bottlenecks in the current split implementation.
- [ ] Implementation
  - [ ] Create a new Metal kernel `fused_attention_v1` combining `AttScores` and `Softmax`.
  - [ ] Extend the kernel to include `AttValues` (generating `fused_attention_full`).
- [ ] Verification
  - [ ] Validate correctness against the split kernels using `TestAttentionLayer`.
  - [ ] Benchmark `fused_attention_full` and record speedup in `benchmark.log`.

## Step 2: SIMD Reduction in Softmax

- [ ] Research
  - [ ] Review Metal Shading Language documentation for `simd_sum` and `simd_max`.
- [ ] Implementation
  - [ ] Modify `Softmax` kernel to use threadgroup SIMD operations.
  - [ ] Adjust threadgroup sizes to align with SIMD width (32 threads).
- [ ] Verification
  - [ ] Verify numerical stability with edge case inputs (very large/small values).
  - [ ] Confirm no regressions in perplexity or output quality.

## Step 3: Flash Attention-2 Implementation

- [ ] Planning
  - [ ] Study Flash Attention-2 tiling and memory access patterns.
- [ ] Implementation
  - [ ] Implement the forward pass tiling loop in Metal.
  - [ ] Optimize shared memory (threadgroup memory) usage for K and V blocks.
- [ ] Verification
  - [ ] Test with long context sequences (> 2048 tokens).
  - [ ] Verify memory usage reduction compared to standard attention.

## Step 4: Speculative Decoding

- [ ] Setup
  - [ ] Integrate a smaller "draft" model (e.g., SmolLM2-135M) alongside the main model.
- [ ] Implementation
  - [ ] Implement the "draft" generation loop (k steps).
  - [ ] Implement the "verify" step in the main model (processing k+1 tokens in parallel).
  - [ ] Add rejection sampling logic.
- [ ] Verification
  - [ ] Measure acceptance rate of draft tokens.
  - [ ] Benchmark end-to-end wall clock speedup.

## Step 5: KV Cache Quantization

- [ ] Implementation
  - [ ] Define 8-bit and 4-bit types for K/V cache storage.
  - [ ] Update `AttScores` and `AttValues` kernels to dequantize K/V on the fly.
- [ ] Verification
  - [ ] Verify potential accuracy loss (perplexity check).
  - [ ] Measure memory savings (GBs saved).

## Step 6: Multi-Token Prefill

- [ ] Refactoring
  - [ ] Modify `Layer` signature to accept a batch of tokens/embeddings.
- [ ] Implementation
  - [ ] Update Metal kernels to handle `(Batch, Seq, Head, Dim)` tensors.
- [ ] Verification
  - [ ] Verify prompt processing speed (tokens/sec) for long prompts.

## Step 7: Continuous Batching

- [ ] Scheduler Design
  - [ ] Design a request queue and scheduling loop (iteration-level).
- [ ] Implementation
  - [ ] Separate the KV cache management from the model execution.
  - [ ] Implement "slot" based memory management for active requests.
- [ ] Verification
  - [ ] Test with concurrent requests (simulated load).

## Step 8: MPS Graph Integration

- [ ] Prototype
  - [ ] Create a standalone `MPSGraph` prototype for the MLP (FeedForward) block.
- [ ] Integration
  - [ ] Integrate the MPSGraph MLP into the main inference loop.
- [ ] Benchmarking
  - [ ] Compare `MPSGraph` performance vs custom Metal kernels for FFN.

## Step 9: Sampling Optimizations

- [ ] Kernels
  - [ ] Implement parallel `ArgMax`, `TopK`, and `TopP` kernels in Metal.
- [ ] Integration
  - [ ] Move sampling logic from Go (CPU) to Metal (GPU).
  - [ ] Only transfer the final selected token ID back to CPU.
- [ ] Verification
  - [ ] Verify distribution of generated tokens matches the CPU implementation.

## Step 10: Paged Attention

- [ ] Memory Manager
  - [ ] Implement a `BlockManager` to handle physical vs virtual blocks.
  - [ ] Update Paged Attention kernels to use block tables.
- [ ] Verification
  - [ ] Verify memory handling under high concurrency (no OOM with fragmentation).
