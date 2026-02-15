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
- [ ] Connect `cmd/webui/engine/adapter.go` to real `internal/engine/engine.go`
- [ ] Add model hot-swapping support
- [ ] Implement KV cache sharing between requests

#### Production Readiness
- [ ] Add API key authentication
- [ ] Implement rate limiting
- [ ] Add OpenAPI documentation
- [ ] Configure CORS for cross-origin requests

#### Load Testing
- [ ] Create load test script (100+ concurrent connections)
- [ ] Benchmark throughput (tokens/second)
- [ ] Measure latency percentiles (p50, p95, p99)

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
- Add cuDNN for additional optimization on NVIDIA GPUs
- Leverage cuDNN's optimized attention kernels
- Support grouped convolutions for MOE models

#### FP8 Support (H100)
- Implement FP8 E4M3/E5M2 quantization
- Tensor Core FP8 support on Hopper architecture
- FP8 dequantization kernels

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
