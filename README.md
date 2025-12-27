
<img width="2784" height="1536" alt="quarrel_logo" src="https://github.com/user-attachments/assets/e1ab45ae-f4de-4f68-91a5-fe931a720c21" />

# longbow-quarrel
Longbow-quarrel is a high-performance LLM inference engine written in Go with Metal GPU acceleration for Apple Silicon. It's designed to run Llama 3 architecture models in GGUF format with maximum throughput and correctness.

Key Features
Performance:

Metal GPU backend using custom kernels and MPS (Metal Performance Shaders)
Achieves ~298 tokens/second on smollm2-135M (116% of llama.cpp reference)
Fused kernel optimizations (RMSNorm+Linear fusion)
Asynchronous GPU dispatch with tensor pooling
CPU profiling support via pprof
Architecture Support:

Llama 3 model architecture
Grouped Query Attention (GQA)
KV cache for autoregressive generation
RoPE positional embeddings
SwiGLU activation
Correctness & Reliability:

Comprehensive error checking and validation
Zero-initialized KV cache to prevent garbage data
Token bounds checking
Output validation tests
Validity checking against llama.cpp reference
Model Format:

GGUF format support (FP16 and FP32)
Compatible with Ollama-downloaded models
Tested with smollm2-135M/360M and Llama 3.2 models
Technical Stack
Language: Go with CGO for Metal interop
GPU: Metal framework (macOS/Apple Silicon only)
Kernels: Custom Metal compute kernels (MatMul, RMSNorm, RoPE, SwiGLU, Attention)
Dependencies: Minimal - Metal, Foundation, MetalPerformanceShaders, Accelerate
Current Status
The project is in active development (release/0.0.1-rc2) with:

Complete Llama 3 inference pipeline
Validated correctness against llama.cpp
Optimized performance exceeding reference implementation
Comprehensive benchmarking and testing infrastructure
It's part of the Longbow ecosystem, with longbow-fletcher handling embeddings and longbow-quarrel focused on text generation inference.
