# Performance Optimization Roadmap

This document outlines the next 10 steps to improve the performance, efficiency, and scalability of the `Longbow-Quarrel` inference engine.

## Step 1: Fused Attention Kernels

Implement a fused Metal kernel for `AttScores + Softmax + AttValues`. This reduces memory bandwidth by keeping intermediate scores in GPU threadgroup memory (SRAM) instead of writing them back to VRAM.

## Step 2: SIMD Reduction in Softmax

Current `Softmax` max/sum calculation uses basic atomic or sequential reductions. Moving to `simd_max` and `simd_sum` within Metal threadgroups will significantly speed up the normalization pass.

## Step 3: Flash Attention-2 Implementation

Port the Flash Attention-2 tiling algorithm to Metal. This will enable handling much larger context lengths with minimal memory overhead and improved throughput.

## Step 4: Speculative Decoding

Implement speculative decoding by using a smaller model (e.g., SmolLM2-135M) to "draft" tokens and the main model to verify them. This can provide a 2-3x speedup on high-end Apple Silicon.

## Step 5: KV Cache Quantization

Quantize the KV cache to 4-bit or 8-bit to reduce memory footprint and bandwidth during the autoregressive phase, allowing for larger batch sizes and longer sequences.

## Step 6: Multi-Token Prefill

Optimize the `Layer` method to process multiple prompt tokens in a single Metal dispatch. Currently, prompt tokens are processed sequentially; batching them will saturate GPU compute units more effectively.

## Step 7: Continuous Batching

Implement continuous batching (iteration-level scheduling) to handle concurrent user requests efficiently, reducing wait times and increasing overall system throughput.

## Step 8: MPS Graph Integration

Explore `MPSGraph` for the FFN block. While custom kernels are fast, `MPSGraph` may offer better fusion and scheduling for complex operations like SwiGLU on newer Apple Silicon chips.

## Step 9: Sampling Optimizations

Implement GPU-side sampling (Top-P, Top-K, Temperature) to avoid transferring large logit tensors back to the CPU for every token.

## Step 10: Paged Attention

Implement Paged Attention (similar to vLLM) to manage KV cache memory with zero fragmentation, enabling massive concurrency for multi-tenant deployments.
