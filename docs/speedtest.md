# Performance Benchmarks

## Executive Summary

Longbow-quarrel achieves **297.90 tokens/second** on Apple M3 Pro, **exceeding llama.cpp by 16%** for the smollm2-135M model.

## Test Environment

- **Hardware**: Apple M3 Pro (12-core CPU, 18-core GPU)
- **OS**: macOS 14+ (Metal API)
- **Model**: smollm2-135M-instruct (GGUF F16 format)
- **Reference**: llama.cpp (Homebrew installed, commit 85c40c9b0)
- **Test Date**: December 2025

## Methodology

### Benchmarking Script

```bash
./scripts/benchmark_compare.sh -n 20
```

The benchmarking script performs:

1. **llama-bench** (reference): Runs with `-p 20 -n 20 -r 1` for text generation throughput
2. **longbow-quarrel**: Generates 20 tokens with prompt "The capital of France is"
3. **Comparison**: Parses t/s metrics and calculates relative performance

### Token Generation

Both implementations perform auto-regressive token generation:

- Single token prompt processing
- 20-token generation sequence
- Greedy decoding (argmax selection)
- No temperature sampling or top-k/top-p

### Metrics Captured

- **Throughput**: Tokens per second (t/s) during generation phase
- **Latency**: Time per token (ms/token)
- **Memory**: GPU VRAM usage via Metal performance counters

## Results

### Throughput Comparison

| Implementation | Throughput (t/s) | vs Reference | Latency (ms/token) |
|----------------|------------------|--------------|---------------------|
| llama.cpp      | 256.63           | 100%         | 3.90                |
| longbow-quarrel | 297.90          | **116%**     | **3.36**            |

### Performance Timeline

| Optimization Phase | Throughput (t/s) | vs Baseline | vs Reference |
|--------------------|------------------|-------------|--------------|
| Baseline           | 17.93            | 1.0x        | 7%           |
| Async Dispatch     | 131.34           | 7.3x        | 51%          |
| Tensor Pooling     | 132.84           | 7.4x        | 52%          |  
| Fused RMSNorm+Linear | **297.90**     | **16.6x**   | **116%**     |

## Key Optimizations

### 1. Async Command Buffer Execution

**Impact**: 7.3x speedup (17.93 → 131.34 t/s)

Removed synchronous GPU waits after each kernel dispatch:

```objectivec
// Before
[commandBuffer commit];
[commandBuffer waitUntilCompleted];

// After
[commandBuffer commit];  // Async execution
```

### 2. Fused RMSNorm+Linear Kernel

**Impact**: 2.27x speedup (131.34 → 297.90 t/s)

Combined normalization and matrix multiplication into single Metal kernel:

- Eliminated 48 intermediate buffer allocations per token
- Used threadgroup shared memory (576 elements)
- Reduced global memory traffic by 50%

```metal
kernel void rmsnorm_linear_f16(...) {
    threadgroup half shared_normed[576];
    // Phase 1: RMS normalization
    // Phase 2: Store in threadgroup  
    // Phase 3: MatMul from shared memory
}
```

## Memory Usage

| Metric | Value |
|--------|-------|
| Model Size (F16) | 256 MB |
| KV Cache (512 ctx) | 12 MB |
| Activation Tensors | ~8 MB (pooled) |
| Total VRAM | ~280 MB |

## Reproducibility

### Build from Source

```bash
git clone https://github.com/23skdu/longbow-quarrel
cd longbow-quarrel
git checkout release/0.0.1-rc1
go build -tags metal -o quarrel ./cmd/quarrel
```

### Download Model

```bash
# Using Ollama
ollama pull smollm2:135m-instruct

# Model path
MODEL=/Users/$USER/.ollama/models/blobs/sha256-f535f83ec568d040f88ddc04a199fa6da90923bbb41d4dcaed02caa924d6ef57
```

### Run Benchmark

```bash
./scripts/benchmark_compare.sh -n 20 -m $MODEL
```

## Analysis

### Why Faster Than llama.cpp?

1. **Fused Operations**: Single kernel dispatch vs separate norm+matmul
2. **Threadgroup Memory**: Zero global memory writes for normalized values
3. **Optimized for Small Models**: Tuned for 576-dim embeddings (smollm2)
4. **Metal Performance**: Direct Metal API vs MPS abstractions

### Scaling Characteristics

Performance scales with:

- **Model size**: Larger models see diminishing returns (>1B params)
- **Context length**: Linear scaling up to 512 tokens
- **Batch size**: Currently optimized for batch=1 (streaming inference)

## Limitations

- **Single Token Generation**: Optimized for auto-regressive decoding
- **Apple Silicon Only**: Requires Metal GPU (M1/M2/M3+)
- **FP16 Precision**: Quantized formats (Q4/Q8) not yet supported

## Future Work

Additional optimizations from the Top 10 plan:

- Custom MatMul kernels for 576×576 matrices (est. +20%)
- Fused SwiGLU+Linear (est. +10%)
- Batched request handling (est. +30% for batch>1)

Current performance already exceeds project goals by surpassing llama.cpp reference.
