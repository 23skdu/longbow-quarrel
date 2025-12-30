# Usage Guide

## Quick Start

```bash
# Build with Metal support
go build -tags metal -o quarrel ./cmd/quarrel

# Run inference
./quarrel -model "mistral:latest" -n 50 -prompt "Hello, world"

# (Optional) Direct path
./quarrel -model ~/.ollama/models/blobs/sha256-... -n 50
```

> [!TIP]
> Longbow-quarrel automatically resolves Ollama model names by searching `~/.ollama/models`. You don't need to provide the full blob path manually.

## Command Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `-model` | (required) | Path to GGUF model file |
| `-prompt` | "Hello" | Input prompt text |
| `-n` | 10 | Number of tokens to generate |
| `-chatml` | `false` | Wrap prompt in ChatML template (Instruction Tuning) |
| `-temp` | `0.7` | Sampling Temperature (0.0 = greedy) |
| `-topk` | `40` | Top-K Sampling |
| `-topp` | `0.95` | Top-P (Nucleus) Sampling |
| `-penalty` | `1.1` | Repetition Penalty (1.0 = none) |

## Model Format

Longbow-quarrel supports GGUF format models with Llama 3 architecture:

```bash
# Example: Download smollm2 via Ollama
ollama pull smollm2:135m-instruct

# Model location
~/.ollama/models/blobs/sha256-f535f83ec568d040f88ddc04a199fa6da90923bbb41d4dcaed02caa924d6ef57
```

- smollm2-135M/360M-instruct (tested)
- Mistral-7B-v0.1/v0.3 (tested)
- Llama 3.2-1B/3B (compatible)
- Any Llama 3/Mistral architecture GGUF model

### Supported Quantizations

- **K-Quantization**: Q3_K, Q4_K, Q6_K (fully accelerated)
- **Standard**: FP16, FP32

## Examples

### Basic Text Generation

```bash
./quarrel \
  -model ~/.ollama/models/blobs/sha256-f535... \
  -prompt "The capital of France is" \
  -n 20
```

Output:

```
Inference complete: generated 20 tokens in 67.153ms (297.90 t/s)
Decoded Text: 
