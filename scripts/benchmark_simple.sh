#!/bin/bash

# Simple benchmark script for Linux/CPU
# Usage: ./scripts/benchmark_simple.sh [model.gguf]

set -e

MODEL="${1:-$HOME/.cache/ollama/models/ggml-model.gguf}"
PROMPT="The capital of France is"
N_TOKENS=32

echo "=== Benchmark (Linux/CPU) ==="
echo "Model: $MODEL"
echo "Prompt: '$PROMPT'"
echo "Tokens: $N_TOKENS"
echo ""

if [ ! -f "$MODEL" ]; then
    echo "Error: Model not found: $MODEL"
    exit 1
fi

echo "Running longbow-quarrel CPU benchmark..."
if ./cmd/benchmark/benchmark -model "$MODEL" -prompt "$PROMPT" -tokens $N_TOKENS 2>&1 | tee /tmp/benchmark_out.txt; then
    TPS=$(grep -oP '\d+\.?\d*\s*tokens/s' /tmp/benchmark_out.txt | head -1 || echo "N/A")
    echo ""
    echo "Throughput: $TPS"
else
    echo "Benchmark failed"
    exit 1
fi

echo ""
echo "Benchmark complete."
