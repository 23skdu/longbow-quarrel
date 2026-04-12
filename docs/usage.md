# Longbow-Quarrel Usage Guide

## Command-Line Usage

### Simple Inference (cmd/simple)

```bash
# Build for macOS/Metal
go build -tags darwin,metal -o quarrel ./cmd/simple/

# Run with GGUF model
./quarrel --model model.gguf --prompt "Your prompt here"

# Options
--model string      # Path to GGUF model file
--prompt string   # Prompt for text generation (default: "The capital of France is")
--n int          # Max tokens to generate (default: 50)
--temp float      # Temperature 0.0-1.0 (default: 0.7)
--topk int       # Top-K sampling (default: 40)
--topp float     # Top-P nucleus sampling (default: 0.95)
--verbose        # Verbose output
```

### Benchmark (cmd/benchmark)

```bash
# Benchmark mode
./benchmark --mode kernel --size 4096           # Kernel benchmark
./benchmark --mode inference --model model.gguf    # Inference benchmark

# Options
--model string    # GGUF model path
--prompt string  # Benchmark prompt
--warmup int    # Warmup runs (default: 2)
--runs int       # Benchmark runs (default: 5)
--len int       # Max tokens (default: 50)
--csv string    # Output CSV file
```

## Supported Models

| Model Family | Architecture | Quantizations |
|-------------|-------------|-------------|
| Llama 3/3.1 | Full + sliding window | Q4_K, Q6_K, Q8_0 |
| Mistral | Sliding window (4096) | Q4_K, Q6_K |
| Gemma 4 | Hybrid (5 sliding + 1 full) | Q4_K, Q6_K |
| Qwen2 | Standard | Q4_K, Q6_K |
| SmolLM2 | Standard | Q4_K, FP16 |

## Docker

```bash
# Linux CPU
docker run ghcr.io/23skdu/longbow-quarrel:latest \
  --model model.gguf \
  --prompt "Hello"

# Build CUDA version (requires cuda_kernels.a)
docker build -f Dockerfile.cuda -t longbow-quarrel:cuda .
```