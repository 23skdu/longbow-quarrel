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

# Create a temporary symlink with .gguf extension for clarity (and if loader requires it)
# Quarrel loader might rely on extension or magic bytes. GGUF magic bytes are 'GGUF'.
# Let's just pass the path directly first.

echo "Running Quarrel..."
# Generating 50 tokens to check for coherence and repetition
./quarrel --model "$MODEL_PATH" --prompt "The quick brown fox jumps over the lazy dog." -n 50 --temp 0.1 > verification_out.txt 2>&1

echo "Quarrel Output:"
cat verification_out.txt

if grep -q "The quick brown fox" verification_out.txt; then
    echo "Prompt found in output."
else
    echo "Warning: Prompt not found in output."
fi

# Check for gibberish markers or weird encodings
if grep -q "▁" verification_out.txt; then
    echo "FAIL: Output still contains U+2581 block character!"
    exit 1
fi

echo "Done."
