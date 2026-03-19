# Longbow-Quarrel Development Roadmap

## Executive Summary

| Feature | Status | Priority | Next Action |
|---------|--------|----------|-------------|
| **Metal Backend (Apple)** | ✅ RESTORED | - | - |
| **CUDA Backend (Linux)** | ✅ IMPLEMENTED | - | - |
| **Test Coverage** | ✅ IMPROVED (36.8% → 46.8%) | - | - |
| **WebUI Service** | ✅ COMPLETE | - | - |
| **Production Integration** | 🔄 IN PROGRESS | High | Connect engine adapter to real inference |
| **cuDNN Integration** | ⏳ PENDING | Medium | Add cuDNN for additional optimization |
| **FP8 Support (H100)** | ⏳ PENDING | Medium | Full FP8 E4M3/E5M2 support |
| **Multi-GPU Support** | ⏳ PENDING | Low | Model parallelism across GPUs |
| **vLLM Integration** | ⏳ PENDING | Low | Export operators for vLLM compatibility |

---

## Active Tasks

### Production Integration (Priority: High)

**Objective:** Complete engine.go integration and prepare for production deployment

#### Engine Integration
- [x] Connect `cmd/webui/engine/adapter.go` to real `internal/engine/engine.go`
- [x] Add model hot-swapping support (UnloadModel exists, but no hot-swap logic for active requests)
- [x] Implement KV cache sharing between requests (current: sequential processing, single CachePos)

#### Production Readiness
- [x] Add API key authentication
- [x] Implement rate limiting
- [x] Add OpenAPI documentation (docs/openapi.yaml created)
- [x] Configure CORS for cross-origin requests

#### Load Testing
- [x] Create load test script (100+ concurrent connections) (scripts/load_test.py created)
- [x] Benchmark throughput (tokens/second)
- [x] Measure latency percentiles (p50, p95, p99) — P50, P95, P99 all implemented

---

## Completed Features

### ✅ Metal Backend (Apple Silicon)
- **Status:** RESTORED and verified
- **Files:** `internal/device/kernels.metal`, `internal/device/metal_backend.m`
- **Coverage:** 61 GPU kernels for FP16, Q3_K, Q4_K, Q6_K, Q8_0

### ✅ CUDA Backend (Linux)
- **Status:** IMPLEMENTED
- **Files:** `internal/device/cuda.go`, `internal/device/cuda_kernels.cu`
- **Features:** Tensor Core WMMA, Flash Attention, Paged KV Cache

### ✅ CPU SIMD Optimizations
- **Status:** COMPLETE
- **AVX2:** Softmax, SwiGLU, FP16→FP32 conversion
- **AVX-512:** Zen 4+, Ice Lake+ support

### ✅ Branchless Quantization
- **Status:** COMPLETE
- **Files:** `internal/gguf/dequant.go`
- **Optimizations:** Q4K/Q6K dequantization without branch mispredictions

### ✅ Test Coverage Improvements
- **Status:** IMPROVED (+10% overall)
- `internal/config`: 0% → 100%
- `internal/logger`: 0% → 100%
- `internal/ollama`: 0% → 41%
- `internal/cpu`: 76.1% → 81.8%
- `internal/simd`: 65.9% → 76.8%
- `internal/metrics`: 4.9% → 88.2%

### ✅ WebUI Service
- **Status:** COMPLETE (commit 9d8fbda)
- **Files:** `cmd/webui/` (19 files, +2363 lines)
- **Features:** Templ-based UI, WebSocket streaming, Docker support

---

## Pending Features (Backlog)

### Medium Priority

#### cuDNN Integration
- [x] Add cuDNN for additional optimization on NVIDIA GPUs
- [x] Leverage cuDNN's optimized attention kernels
- [ ] Support grouped convolutions for MOE models

#### FP8 Support (H100)
- [x] Implement FP8 E4M3/E5M2 quantization
- [x] Tensor Core FP8 support on Hopper architecture
- [x] FP8 dequantization kernels

### Low Priority

#### Multi-GPU Support
- Model parallelism across multiple GPUs
- Tensor parallelism for large models (>70B)
- Pipeline parallelism support

#### vLLM Integration
- Export operators for vLLM compatibility
- Paged Attention API alignment
- Batch scheduler integration

---

## Build Commands Reference

```bash
# Metal backend (macOS Apple Silicon)
CGO_ENABLED=1 go build -tags metal ./...

# CUDA backend (Linux NVIDIA)
go build -tags=cuda,amd64 ./...

# AVX2 only (no GPU)
go build -tags=amd64 ./...

# AVX-512 (Zen 4+, Ice Lake+)
go build -tags=amd64,avx512 ./...

# All optimizations
go build -tags=cuda,amd64,avx512 ./...
```

---

## Testing Commands

```bash
# Run all tests with coverage
go test -coverprofile=coverage.out ./...

# Run specific package tests
go test ./internal/cpu/...
go test ./internal/simd/...
go test ./internal/metrics/...

# Metal-specific tests (macOS)
go test -tags=metal ./internal/device/...
```

---

## Incomplete Features & Code TODOs

### High Priority (Blocking Functionality)

#### Model Hot-Swapping (cmd/webui/engine/adapter_*.go)
- **Status:** INCOMPLETE
- **Issue:** `UnloadModel()` exists but no hot-swap logic for active requests
- **Impact:** Cannot switch models while requests are in-flight
- **Location:** `cmd/webui/engine/types.go:35-36`

#### KV Cache Sharing Between Requests (internal/engine/engine.go)
- **Status:** INCOMPLETE
- **Issue:** Current sequential processing, single CachePos
- **Impact:** Cannot batch multiple requests efficiently
- **Location:** `cmd/webui/handlers/inference.go`

#### Mamba Layer Detection (internal/engine/mamba.go:91)
- **Status:** TODO
- **Issue:** `IsMambaLayer()` relies on nil check - needs proper config-based detection
- **Details:** `// TODO: Implement logic based on model config. For Nemotron-3-Nano, pattern detection needed`

#### Tensor Zero() Method (internal/engine/engine.go:1191)
- **Status:** TODO
- **Issue:** `// TODO: Add Zero() to tensor` - SSM state initialization incomplete
- **Impact:** SSM states may contain dirty data between inference runs

#### Quantization Support (internal/gguf/quantize.go:6)
- **Status:** NOT IMPLEMENTED
- **Issue:** `QuantizeWeightsToQ4K()` returns "not implemented"
- **Impact:** Cannot quantize models to Q4_K format at runtime
- **Related:** Tests in `internal/gguf/quantization_q4k_test.go` skip due to missing implementation

### Medium Priority (Quality/Performance)

#### RoPE Attention Causal Mask Test (internal/device/rope_attention_test.go:122)
- **Status:** TODO (Test Incomplete)
- **Issue:** `// TODO: Call attention kernel with causal mask`
- **Details:** Test documents expected behavior but kernel not called

#### Perplexity Calculation (internal/engine/engine.go:113)
- **Status:** PLACEHOLDER
- **Issue:** Simplified implementation - `// This is just a placeholder - real perplexity requires model probabilities`
- **Impact:** Quality metrics not accurate without proper token probability computation

#### Quality Evaluator Tests (internal/engine/smollm2_zero_logits_test.go:17,23)
- **Status:** TODO
- **Issues:**
  - `// TODO: Implement with full engine + tokenizer setup`
  - `// TODO: Implement by comparing tokenizer vocab with model expectations`

### Low Priority (Future/Backlog)

#### CUDA Coherence Tests (cmd/smoke_test/cuda_coherence_test.go:94,102,110)
- **Status:** SKIPPED (Placeholder tests)
- **Issue:** `t.Skip("CUDA engine not implemented - this is a placeholder for CUDA coherence tests")`
- **Impact:** No CUDA coherence validation

#### Inference String Method (internal/engine/engine.go:1260)
- **Status:** PLACEHOLDER
- **Issue:** Uses `inputTokens := []int{1, 2, 3} // Placeholder tokenization`
- **Impact:** Cannot properly test string-based inference

#### IQ1_M Quantization (internal/gguf/gguf_test.go:149)
- **Status:** NOT IMPLEMENTED
- **Issue:** `t.Errorf("IQ1_M SizeBytes() = %d, want 0 (not implemented)", got)`

---

## API Endpoints Summary

| Endpoint | Method | Description | Status |
|----------|--------|-------------|--------|
| `/health` | GET | Health check | ✅ Complete |
| `/healthz` | GET | Simple liveness | ✅ Complete |
| `/readyz` | GET | Readiness probe | ✅ Complete |
| `/version` | GET | Version info | ✅ Complete |
| `/metrics` | GET | Prometheus metrics | ✅ Complete |
| `/api/models` | GET | List models | ✅ Complete |
| `/api/generate` | POST | Generate text (sync) | ✅ Complete |
| `/api/stream` | POST | Stream text (SSE) | ✅ Complete |
| `/ws` | WebSocket | Real-time inference | ✅ Complete |
