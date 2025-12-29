#!/bin/bash
set -e

# Mistral blob hash from manifest
BLOB_HASH="sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f"
MODEL_PATH="$HOME/.ollama/models/blobs/$BLOB_HASH"

echo "Verifying Mistral output..."
echo "Model Blob: $MODEL_PATH"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model blob not found!"
    exit 1
fi

echo "Running Quarrel with Instruct Prompt..."
# Use [INST] tag for Mistral 7B Instruct
./quarrel --model "$MODEL_PATH" --prompt "[INST] Write a short haiku about code. [/INST]" --n 50 --temp 0.1 > verification_out.txt 2>&1

echo "Quarrel Output:"
cat verification_out.txt

if grep -q "code" verification_out.txt; then
    echo "Keyword 'code' validation passed (weak check)."
fi

echo "Done."
