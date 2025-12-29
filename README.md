
<img width="2784" height="1536" alt="quarrel_logo" src="https://github.com/user-attachments/assets/e1ab45ae-f4de-4f68-91a5-fe931a720c21" />

# longbow-quarrel

Longbow Quarrel is a high-performance LLM inference engine written in Go
with Metal GPU acceleration for Apple Silicon. It's designed to run Llama 3
architecture models in GGUF format with maximum throughput and correctness.

## Key Features

### Performance

- Metal GPU backend using custom kernels for Llama architecture
- Native support for **K-Quantization** (Q3_K, Q4_K, Q6_K)
- Achieves high-performance inference on Apple Silicon
- Fused kernel optimizations (RMSNorm+Linear fusion, attention scaling)
- Thread-safe asynchronous GPU dispatch with tensor pooling and memory budget
- CPU profiling support via pprof

### Architecture Support

- Llama 3 model architecture
- Grouped Query Attention (GQA)
- KV cache for autoregressive generation
- RoPE positional embeddings
- SwiGLU activation

### Correctness & Reliability

- Comprehensive error checking and validation
- Zero-initialized KV cache to prevent garbage data
- Token bounds checking
- Output validation tests
- Validity checking against llama.cpp reference

### Model Format

- GGUF format support (FP16, FP32, Q3_K, Q4_K, Q6_K)
- **Native Ollama model support** - use model names like `mistral:latest`
- Automatically resolves Ollama models from `~/.ollama/models`
- Backward compatible with direct GGUF file paths
- Tested and validated with SmolLM2 (135M/360M), Llama 3.2, and Mistral models

## Technical Stack

- **Language**: Go with CGO for Metal interop
- **GPU**: Metal framework (macOS/Apple Silicon only)
- **Kernels**: Custom Metal compute kernels
  (MatMul, RMSNorm, RoPE, SwiGLU, Attention)
- **Dependencies**: Minimal - Metal, Foundation,
  MetalPerformanceShaders, Accelerate

The project is in active development (release/0.1.0) with:

- Complete Llama 3 inference pipeline with Metal acceleration
- Validated correctness against `llama.cpp` for multiple quantization levels
- Robust memory management with pooling and global budget
- Comprehensive benchmarking and testing infrastructure

It's part of the Longbow ecosystem, with longbow-fletcher handling
embeddings and longbow-quarrel focused on text generation inference.
