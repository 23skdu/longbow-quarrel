# Longbow-Quarrel Release 0.1.0

## Summary

Longbow-Quarrel is a high-performance LLM inference engine built in Go, supporting GGUF model format with Metal (Apple Silicon) and CUDA (NVIDIA) acceleration.

## What's New in 0.1.0

### Code Changes
- Added `bin/` to `.gitignore` to prevent committing compiled binaries
- Cleaned up deleted directories from git history

### Previously Completed Features (in prior releases)
- CUDA GPU acceleration with fused kernels
- Metal (Apple Silicon) backend
- GGUF model loading (Q4_K, Q6_K, Q8_0, F16, F32)
- Sliding window attention for Mistral/Gemma4
- Gemma4 hybrid attention (5 sliding + 1 full per 6 layers)
- OpenAI-compatible API endpoints (`/v1/chat/completions`, `/v1/completions`)
- Benchmark command (`cmd/benchmark`)
- Output validation with llama.cpp comparison
- cuDNN flash attention integration

## Docker Images

| Platform | Image | Status |
|----------|-------|--------|
| Linux (CPU) | `ghcr.io/23skdu/longbow-quarrel:latest` | ✅ Pushed |
| Linux (CUDA) | `ghcr.io/23skdu/longbow-quarrel:cuda-latest` | ⚠️ Needs cuda_kernels |

## Quick Start

```bash
# Run with Docker
docker run ghcr.io/23skdu/longbow-quarrel:latest --model model.gguf --prompt "Hello"

# Build from source
go build -o quarrel ./cmd/simple
```

## Known Issues
- CUDA build requires `cuda_kernels` library (build from `internal/device/cuda_kernels.cu`)
- Metal build requires macOS with Apple Silicon

## Repo Size
~549MB git repository (includes full history)

---

*Released: April 2026*