# Longbow-Quarrel Usage Guide

## Overview

Longbow-Quarrel provides high-performance LLM inference via Metal GPU acceleration (macOS) or CUDA (Linux/NVIDIA). Supports direct GGUF loading and Ollama model integration.

## Supported Models & Quantization

| Model Family | Architecture | Quantization |
|--------------|---------------|--------------|
| Llama 3/3.1 | Full attention + sliding | Q3_K, Q4_K, Q6_K, IQ4_NL |
| Llama 3.2 | 1B-8B variants | Q4_K_M (default) |
| Mistral | Sliding window | Q4_K, Q6_K |
| Gemma 4 | Hybrid attention | Q4_K, Q6_K |
| Qwen2 | Standard | Q4_K, Q6_K |
| SmolLM2 | 135M/360M | Q4_K, FP16 |

## Quick Start

```bash
# Build
go build -tags darwin,metal -o generate_text ./cmd/generate_text/

# Run with Ollama model
./generate_text -model gemma4:e4b -prompt "Hello world"

# Run with GGUF file
./generate_text -model /path/to/model.gguf -prompt "Your prompt"
```
