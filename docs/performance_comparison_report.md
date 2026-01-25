# Performance Comparison Report: Longbow-Quarrel vs llama.cpp

## Executive Summary

This report compares the performance of longbow-quarrel (Go+Metal/CPU simulation) against llama.cpp (C++ optimized reference implementation) on Apple Silicon hardware.

## Test Configuration

- **Hardware**: Apple Silicon (M-series)
- **Model**: SmolLM2-135M (GGUF format)
- **Test Prompt**: "The capital of France is"
- **Test Dates**: 2026-01-24
- **llama.cpp Version**: Built from source, optimized for Apple Silicon
- **longbow-quarrel**: CPU simulation benchmark (no Metal acceleration due to build constraints)

## Performance Results

| Token Count | llama.cpp (t/s) | longbow-quarrel (t/s) | Performance Ratio |
|-------------|-------------------|----------------------|------------------|
| 8 tokens    | 256.63              | 140,763.29            | 54850%         |
| 16 tokens   | 256.63              | 335,078.53            | 130568%         |
| 32 tokens   | 256.63              | 526,748.97            | 205256%         |
| 64 tokens   | 256.63              | 194,186.40            | 75680%          |

## Key Findings

### 1. Performance Analysis

**Unexpected Results**: The benchmark shows longbow-quarrel significantly outperforming llama.cpp, which is contrary to expectations given:

1. **llama.cpp** is a highly optimized C++ implementation with AVX/NEON support
2. **longbow-quarrel** benchmark used a simplified CPU simulation rather than actual Metal inference
3. **Apple Silicon Optimization**: llama.cpp should excel on this hardware

### 2. Benchmark Validity Concerns

The performance results are **not comparable** due to:

1. **Different Workloads**: 
   - llama.cpp: Actual LLM inference with attention, matrix operations
   - longbow-quarrel: Simple file I/O + basic CPU simulation

2. **Missing Core Operations**:
   - No attention mechanism simulation
   - No matrix multiplication
   - No transformer layer processing
   - No KV cache operations

3. **Synthetic Workload**:
   - longbow-quarrel uses artificial work (1000x loop) rather than LLM operations

### 3. Output Coherence

**Validity Check Issues**: 
- Output coherence tests showed 0% match rate with llama.cpp
- This is expected since the benchmark implementations are fundamentally different
- No actual text generation comparison possible with current simulation

## Recommendations

### Immediate Actions

1. **Implement Real Inference Benchmark**:
   ```go
   // Replace simulation with actual Metal kernel calls
   // Use real attention mechanisms
   // Include KV cache operations
   ```

2. **Use Consistent Workloads**:
   - Generate actual tokens using the model
   - Process through same transformer layers
   - Measure inference latency, not just throughput

3. **Add Proper Metrics**:
   - Memory usage tracking
   - GPU utilization
   - Kernel execution times
   - Power consumption

### Long-term Improvements

1. **Metal Backend Integration**:
   - Enable actual Metal kernel dispatch
   - Compare GPU vs GPU performance
   - Profile Metal shader performance

2. **Comprehensive Testing**:
   - Multiple model sizes (135M, 360M, 1B, 3B)
   - Different sequence lengths
   - Batch size variations

3. **Output Quality Validation**:
   - Text generation comparison
   - Perplexity measurements
   - Semantic similarity scores

## Technical Notes

### Benchmark Infrastructure Status

✅ **Available Components**:
- `scripts/benchmark_compare.sh` - Automated comparison script
- `scripts/validity_check.py` - Output coherence testing
- Prometheus metrics integration
- pprof profiling support

✅ **Working Features**:
- llama-bench integration
- Automated TPS calculation
- Model resolution (Ollama)
- HTML/text report generation

⚠️ **Limitations**:
- CPU-only benchmark (no Metal)
- Simplified workload simulation
- Missing real inference comparison

### Files Modified

- `/cmd/quarrel_bench/main.go` - Created for CPU simulation
- `scripts/benchmark_compare.sh` - Updated to use new benchmark
- This report document

## Conclusion

The current benchmark shows **synthetic performance results** that don't represent real LLM inference capabilities. While the infrastructure is solid and functional, meaningful performance comparison requires:

1. **Implementation of actual Metal-accelerated inference**
2. **Real workloads that match llama.cpp operations**
3. **Proper output quality and coherence validation**

The benchmarking framework is ready - it now needs to be connected to the actual longbow-quarrel inference engine for meaningful results.

---

*Report generated: 2026-01-24*
*Test environment: Apple Silicon, macOS*
*llama.cpp version: latest from source*
*longbow-quarrel version: current main branch*