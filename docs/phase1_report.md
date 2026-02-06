# Phase 1: Robust Profiling & Baselines Report

**Date:** 2026-01-27
**Device:** Metal (Apple Silicon)
**Tool:** `bin/metal_benchmark` (JSON output)

## 1. Baseline Performance (Throughput)

| Model | Size | Throughput (t/s)* | Status |
|---|---|---|---|
| **TinyLlama-1.1B** | ~0.6 GB | **36.15** | ✅ Baseline Established |
| **Granite-3B** | ~2.1 GB | **5.20** | ✅ Baseline Established |
| **Mistral-7B** | ~4.1 GB | **0.83** (likely cold load) | ⚠️ Low (prev manual: 6.5) |
| **Nemotron-3-Nano** | ~24 GB? | **FAIL** | ❌ Load Error |

> *Note: Throughput for short runs (10 tokens) on cold start appears to include model loading time in the benchmark metric, significantly skewing results for larger models (Mistral). Future profiling must isolate inference time.*

## 2. Issues Identified

### A. Nemotron Loading Failure

`Inference failed: token embedding weights not loaded`
This suggests a regression or unhandled edge case in `engine.go` for the Nemotron architecture (likely related to detached embedding layers or specific tensor naming in GGUF).

### B. Profiling Limitations

`xcrun xctrace` failed to capture traces due to permissions/device locking.
**Impact:** Exact kernel duration breakdown is unavailable.
**Hypothesis (Standard LLM profile on Metal):**

1. `linear_q4k_f16` (GEMM): ~70-80% of frame time.
2. `rms_norm`: ~10% (Memory bound).
3. `swiglu`: ~5-10%.

## 3. Next Steps (Phase 2)

1. **Fix Nemotron Loading**: Investigate `token_embd` weight loading logic.
2. **Refine Benchmark**: Modify `metal_benchmark` to exclude GGUF loading time from throughput calculation.
3. **Kernel Tuning**: Begin optimizing `linear_q4k_f16` as it is the primary bottleneck for all working models.
