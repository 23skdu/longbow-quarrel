# Performance Benchmarks: longbow-quarrel vs. llama.cpp

This document provides a performance comparison between `longbow-quarrel` and `llama.cpp` for LLM inference on Apple Silicon.

## Benchmark Setup

-   **System**: Apple M3 Pro
-   **Model**: `smollm2:135m-instruct-fp16` (134.52M parameters, F16)
-   **Prompt**: "The quick brown fox" (4 tokens)
-   **Generation Length**: 256 tokens

## Benchmark Results

| Engine | Test | Throughput (t/s) | Notes |
| :--- | :--- | :--- | :--- |
| **longbow-quarrel** | End-to-End | **~63.5 t/s** | Includes prompt processing + generation |
| **llama.cpp** | Prompt Processing | `595.73 t/s` | `pp4` (4 prompt tokens) |
| **llama.cpp** | Token Generation | `225.29 t/s` | `tg256` (256 generated tokens) |

### Summary

-   `llama.cpp`'s token generation is approximately **3.5x faster** than `longbow-quarrel`'s overall throughput (`225.29 / 63.5`).
-   `longbow-quarrel`'s performance is respectable but shows a significant gap compared to the highly optimized `llama.cpp`.
-   The primary difference lies in the efficiency of the token generation loop.

## Commands Used

### longbow-quarrel

```bash
# Run benchmark 3 times to get a stable average
for i in {1..3}; do
  echo "--- Run $i ---"
  ./quarrel -model smollm2:135m-instruct-fp16 -prompt "The quick brown fox" -n 256 2>&1 | grep "Inference complete"
done
```
**Output:**
-   Run 1: `62.07 t/s`
-   Run 2: `64.86 t/s`
-   Run 3: `63.65 t/s`

### llama.cpp

```bash
# Build llama.cpp with Metal support
cmake -B build
cmake --build build --config Release -j

# Run benchmark
MODEL_PATH="/Users/rsd/.ollama/models/blobs/sha256-f535f83ec568d040f88ddc04a199fa6da90923bbb41d4dcaed02caa924d6ef57"
./build/bin/llama-bench -m "$MODEL_PATH" -p 4 -n 256 -ngl 99 -r 3 -o md
```

## Next Steps

The performance gap highlights several areas for optimization in `longbow-quarrel`, as outlined in the [Next Steps Roadmap](nextsteps.md):

1.  **GPU Kernel Optimization (P1)**: Investigate and optimize the Metal compute kernels, particularly for matrix multiplication and attention.
2.  **Memory Management (P1)**: Improve KV-cache management and tensor pooling to reduce overhead.
3.  **Concurrency (P2)**: Implement multi-threading and parallel processing to improve overall throughput.
