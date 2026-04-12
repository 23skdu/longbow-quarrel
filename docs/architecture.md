# Longbow-Quarrel Architecture

## Overview

Longbow-Quarrel is a Go-based LLM inference engine with pluggable GPU backends.

```
┌─────────────────────────────────────────────┐
│               CLI / WebUI                   │
└─────────────────┬─────────────────────────┘
                  │
┌─────────────────▼─────────────────────────┐
│          Engine Interface               │
│  (Infer, InferWithCallback, SwapModel)   │
└─────────────────┬─────────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    ▼             ▼             ▼
┌────────┐  ┌────────┐  ┌──────────┐
│ Metal  │  │ CUDA   │  │   CPU    │
│Engine │  │Engine │  │  Engine │
└───┬────┘  └───┬────┘  └────┬────┘
    │           │            │
    ▼           ▼            ▼
┌────────┐  ┌────────┐  ┌────────┐
│ Metal  │  │ CUDA   │  │ SIMD   │
│Kernels│  │Kernels│  │  CPU   │
└────────┘  └────────┘  └────────┘
```

## Core Components

### 1. Engine Interface (`internal/engine/interface.go`)

```go
type Engine interface {
    Infer(tokens []int, count int, cfg SamplerConfig) ([]int, error)
    InferWithCallback(tokens []int, count int, cfg SamplerConfig, callback func(token int)) ([]int, error)
    Config() config.Config
    Close()
    SwapModel(modelPath string, cfg config.Config) error
}
```

Priority: Metal → CUDA → CPU → Mock

### 2. GPU Backends

| Backend | File | Platform |
|--------|------|----------|
| Metal | `internal/device/metal.go` | macOS/Apple Silicon |
| CUDA | `internal/device/cuda.go` | Linux/NVIDIA |
| CPU | `internal/device/cpu.go` | Fallback |

### 3. Engine Implementations

| Engine | Location | GPU |
|--------|----------|-----|
| metalEngine | `engine.go` | Metal |
| cudaEngine | `engine_cuda.go` | CUDA |
| CPUEngine | `engine_cpu.go` | CPU |

### 4. Model Loading

- **GGUF Parser**: `internal/gguf/` - Parses GGUF format
- **Tensor Loader**: `internal/engine/tensor_loader.go` - Loads weights to GPU

### 5. Quantization Support

| Type | Status |
|------|--------|
| Q4_K | Full support |
| Q6_K | Full support |
| Q8_0 | Full support |
| FP16 | Full support |
| FP32 | Reference |

### 6. Attention Mechanisms

| Model | Mechanism |
|-------|----------|
| Llama 3 | Standard + sliding |
| Mistral | Sliding window (4096) |
| Gemma 4 | Hybrid (5 sliding + 1 full) |
| Qwen2 | Standard |
| SmolLM2 | Standard |

## Data Flow

```
Prompt → Tokenizer → Encode → [Prefill Phase] → [Decode Phase] → Sampler → Token
                    │              │
                    ▼              ▼
              Layer Loop     KV Cache Update
              (Q/K/V/O)      (Sliding or Paged)
```

## Key Files

```
internal/engine/
  interface.go      # Engine interface
  engine.go         # Metal engine (main)
  engine_cuda.go    # CUDA engine
  engine_cpu.go    # CPU fallback
  sampler.go      # Token sampling
  kv_cache.go     # KV cache management

internal/device/
  metal.go        # Metal GPU kernels
  cuda.go         # CUDA GPU kernels
  cpu.go          # CPU/SIMD kernels

internal/gguf/
  gguf.go         # GGUF parser
  quantization.go # Quantization/dequantization
```

## Build Tags

| Tag | Description |
|-----|------------|
| `darwin` | macOS build |
| `metal` | Metal GPU support |
| `cuda` | CUDA GPU support |

## Memory Management

- Tensor pooling for reduced allocations
- Memory budget tracking
- KV cache with sliding window support
- Weight caching for dequantized tensors