# Performance Improvement Plan (20-Part)

This document outlines the strategic roadmap for significantly enhancing the performance of `longbow-quarrel`.

## Phase 1: Kernel & Dispatch Optimizations

1. **SIMD Unrolling in Metal Kernels**: Explicitly unroll loops in `kernels.metal` (e.g., Q4_K dot products) to maximize instruction-level parallelism.
2. **Threadgroup Memory (LDS) Optimization**: Fine-tune the use of `threadgroup` memory in attention and softmax kernels to reduce global memory bandwidth usage.
3. **Warp-Level Shuffles**: Utilize SIMD-group shuffles instead of threadgroup memory for reductions within a warp (32 threads).
4. **Asynchronous Command Buffer Pipelining**: Overlap CPU-side token processing and GGUF parsing with GPU-side execution by pipelining command buffers.
5. **Pre-compiled Metal Libraries**: Switch from runtime kernel compilation to pre-compiled `.metallib` files to reduce initial startup latency.

## Phase 2: Structural Architecture Improvements

1. **Paged Attention**: Implement paged KV cache management (similar to vLLM) to significantly reduce memory fragmentation and increase batch capacity.
2. **Fused RMSNorm + Linear**: Create a single fused kernel for RMSNorm followed by the first Linear projection to save memory round-trips.
3. **Fused SwiGLU + Down Projection**: Combine the SwiGLU activation and the subsequent down-projection into one kernel call.
4. **Continuous Batching**: Implement continuous batching to process multiple requests with varying sequence lengths simultaneously.
5. **Speculative Decoding**: Implement speculative decoding using a smaller drafter model (e.g., SmolLM2-135M) to accelerate the generation of a larger target model (e.g., Mistral-7B).

## Phase 3: Advanced Quantization & Memory

1. **Q4_0 and Q8_0 Support**: Add support for standard Q4_0 and Q8_0 quantization formats for broader model compatibility.
2. **Weight-Only Quantization (Int4/Int8)**: Optimize for weight-only quantization where activations stay in FP16/BF16 but weights are dequantized on-the-fly.
3. **KV Cache Compression**: Implement FP8 or Int8 quantization for the KV cache to double the effective context window.
4. **DirectStorage / zero-copy Weight Loading**: Use macOS `mmap` more aggressively with `MTLResourceStorageModeShared` to minimize copying between CPU and GPU.
5. **Tensor Pooling Enhancements**: Refine the tensor pool to use a slab-based allocator to further reduce allocation overhead.

## Phase 4: Scaling & Hardware Utilization

1. **Multi-GPU Support**: Distribute model layers across multiple Apple Silicon GPUs (if available in Mac Studio/Pro configurations).
2. **Tensor Parallelism**: Split large weight matrices across GPUs for parallel execution of single-layer computations.
3. **MPS Graph Integration**: Evaluate using the `Metal Performance Shaders Graph` framework for complex higher-level operations to leverage Apple's built-in optimizations.
4. **Numerical Stability (BF16 Support)**: Optimize kernels for BF16 where supported (M2/M3 chips) to improve training-inference parity and stability.
5. **Automatic Profile-Guided Optimization (PGO)**: Utilize Go's PGO features along with GPU profiling data to optimize the overall binary for common execution paths.

---

This plan should be implemented iteratively, with correctness verification at each step.
