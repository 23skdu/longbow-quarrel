# Performance & Benchmarking

Longbow-Quarrel is designed to be a high-performance LLM inference engine for Apple Silicon. This document outlines how to measure its performance and compares it against references like `llama.cpp`.

## Quick Performance Summary

| Metric | Target | Current Status | Notes |
| :--- | :--- | :--- | :--- |
| **Throughput (FP16)** | >100 t/s | **~75 t/s** | Tested on M3 Pro, smollm2-135M |
| **Throughput (Q4_K)** | >250 t/s | *Optimization Pending* | K-Quant support in progress |
| **Correctness** | >95% Match | **>99%** | Validated against llama.cpp |

> [!NOTE]
> Performance can vary based on hardware and thermal state. The "Speed Test" claiming 297 t/s (in `docs/archive/speedtest.md`) was likely a specific peak burst or different measuring methodology. We track sustainable throughput here.

## Running Benchmarks

We provide several scripts to validate performance and correctness.

### 1. `bin/metal_benchmark` (Preferred)

This is the compiled Go benchmark tool for Metal inference.

```bash
# Build
go build -tags "darwin,metal" -o bin/metal_benchmark ./cmd/metal_benchmark

# Run
./bin/metal_benchmark -model <path_to_gguf> -tokens 100
```

### 2. `scripts/benchmark_compare.sh`

A wrapper script that runs both `quarrel` and `llama.cpp` (if installed) to provide a direct comparison.

```bash
./scripts/benchmark_compare.sh -p "The quick brown fox" -n 64
```

### 3. `scripts/validity_check.py`

Ensures that the model is generating *correct* text, not just fast gibberish. It compares output similarity with `llama-completion`.

```bash
python3 scripts/validity_check.py
```

## Profiling

To investigate performance bottlenecks (e.g., stalling on Metal synchronization), use the built-in profiling support:

```bash
# Enable CPU profiling
./scripts/benchmark_compare.sh --profile

# Analyze results
go tool pprof -http=:8080 cpu.pprof
```

## Comparisons

### vs. llama.cpp

**Data from Jan 2026 runs on M3 Pro**

- **llama.cpp**: ~225 t/s (token generation)
- **longbow-quarrel**: ~65-75 t/s (end-to-end)

**Analysis**:
The current gap is primarily due to:

1. **Metal Synchronization**: We currently synchronize between CPU and GPU more often than necessary.
2. **Kernel Dispatch**: `llama.cpp` batches kernels more aggressively.
3. **Quantization**: We are currently fastest in FP16; quantized kernels are still being optimized.

See [docs/nextsteps.md](nextsteps.md) for our optimization roadmap.
