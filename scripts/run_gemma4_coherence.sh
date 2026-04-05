#!/bin/bash

# Configuration
MODEL_PATH="./models/gemma4-8.0B-Q4_K_M.gguf"
OLLAMA_MODEL="gemma4:e4b" # Exact model from ollama list
MAX_TOKENS=50
OUTPUT_FILE="gemma4_coherence_results.json"

echo "Building gemma4_coherence tool..."
go build -tags "darwin metal" -o gemma4_coherence ./cmd/gemma4_coherence

if [ $? -ne 0 ]; then
    echo "Build failed. Ensure you have 'metal' build tags available on Darwin."
    exit 1
fi

echo "Running coherence tests..."
./gemma4_coherence --model "$MODEL_PATH" --ollama-model "$OLLAMA_MODEL" --tokens "$MAX_TOKENS" --output "$OUTPUT_FILE"

if [ $? -eq 0 ]; then
    echo "Testing complete. Results saved to $OUTPUT_FILE"
else
    echo "Testing failed."
    exit 1
fi
