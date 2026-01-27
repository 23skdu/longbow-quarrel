# Usage Guide

## Quick Start

```bash
# Build with Metal support
go build -tags metal -o quarrel ./cmd/quarrel

# Run inference
./quarrel -model "mistral:latest" -n 50 -prompt "Hello, world"
```

> [!TIP]
> Longbow-quarrel automatically resolves Ollama model names by searching `~/.ollama/models`. You don't need to provide the full blob path manually.

## Command Line Options

### General

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-model` | (required) | Path to GGUF model file, or Ollama model name (e.g., `smollm2:135m`) |
| `-prompt` | "Hello world" | Input prompt text |
| `-n` | 20 | Number of tokens to generate |
| `-stream` | `false` | Stream tokens to stdout as they are generated |
| `-metrics` | `:9090` | Address to serve Prometheus metrics |
| `-gpu` | `false` | Explicitly request Metal GPU acceleration (Auto-detected usually) |

### Sampling & quality

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-temp` | `0.7` | Sampling Temperature (0.0 = greedy) |
| `-topk` | `40` | Top-K Sampling |
| `-topp` | `0.95` | Top-P (Nucleus) Sampling |
| `-penalty` | `1.1` | Repetition Penalty (1.0 = none) |
| `-quality` | `false` | Enable advanced quality-guided sampling |
| `-chatml` | `false` | Wrap prompt in ChatML template (Instruction Tuning) |

### Advanced / Debugging

| Flag | Default | Description |
|------|---------|-------------|
| `-kv-cache-size` | 22 | KV cache size in MiB |
| `-benchmark` | `false` | Run in benchmark mode (minimal output) |
| `-debug-dequant` | `false` | Enable dequantization debug dump |
| `-debug-activations` | `false` | Enable layer-by-layer activation dumping |
| `-debug-perf` | `false` | Enable performance metric logging for kernels |

## Model Format

Longbow-quarrel supports GGUF format models with Llama 3 architecture:

```bash
# Example: Download smollm2 via Ollama
ollama pull smollm2:135m-instruct
```

- smollm2-135M/360M-instruct (tested)
- Mistral-7B-v0.1/v0.3 (tested)
- Llama 3.2-1B/3B (compatible)
- Any Llama 3/Mistral architecture GGUF model

### Supported Quantizations

- **K-Quantization**: Q3_K, Q4_K, Q6_K (fully accelerated)
- **Standard**: FP16, FP32
