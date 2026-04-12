<img width="2784" height="1536" alt="quarrel_logo" src="https://github.com/user-attachments/assets/e1ab45ae-f4de-4f68-91a5-fe931a720c21" />

# Longbow-Quarrel

High-performance LLM inference engine written in Go with Metal GPU acceleration for Apple Silicon, with optional CUDA support for NVIDIA GPUs on Linux.

## Features

### Model Support
- **Architectures**: Llama 3/3.1/3.2, Mistral, Gemma 4, Qwen2, SmolLM2
- **Quantizations**: Q4_K, Q6_K, Q8_0, FP16, FP32 (via GGUF format)

### GPU Acceleration
- **Metal**: Custom compute kernels for Apple Silicon (MatMul, RMSNorm, RoPE, SwiGLU, Attention)
- **CUDA**: Fused kernels with cuDNN flash attention support

### Advanced Attention
- Sliding window attention (Mistral 4096 tokens)
- Gemma4 hybrid attention (5 sliding + 1 full per 6 layers)
- Grouped Query Attention (GQA)

### API
- OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/completions`)
- WebSocket streaming
- Prometheus metrics at `/metrics`

## Quick Start

```bash
# Run with GGUF model (macOS/Metal)
go run -tags darwin,metal ./cmd/simple/main.go -model model.gguf -prompt "Hello"

# Run with Docker
docker run ghcr.io/23skdu/longbow-quarrel:latest --model model.gguf -prompt "Hello"
```

## Benchmark

```bash
# Kernel benchmark
./cmd/benchmark --mode kernel --size 4096

# Inference benchmark  
./cmd/benchmark --mode inference --model model.gguf --prompt "Your prompt"
```

## Testing

```bash
# Unit tests
go test ./internal/device/...

# Engine tests
go test ./internal/engine/...
```

## Docker Images

| Image | Description |
|-------|-------------|
| `ghcr.io/23skdu/longbow-quarrel:latest` | Linux CPU (amd64) |
| `ghcr.io/23skdu/longbow-quarrel:cuda-latest` | Linux CUDA (amd64) |

## Project Structure

```
cmd/
  simple/          # Simple CLI inference
  benchmark/       # Performance benchmarking
  webui/           # Web UI with API
  quarrel/        # CUDA CLI (Linux only)
internal/
  engine/         # Inference engine
  device/         # GPU backends (Metal, CUDA, CPU)
  gguf/           # GGUF parsing
  tokenizer/      # Tokenization
```

---

*See `docs/` for detailed documentation.*