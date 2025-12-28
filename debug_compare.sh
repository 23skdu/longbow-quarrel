#!/bin/bash
MODEL_PATH="/Users/rsd/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f"
PROMPT="What is the chemical symbol for gold?"

echo "=== Running Quarrel ==="
./quarrel -model "$MODEL_PATH" -prompt "$PROMPT" -n 30 > quarrel_out.txt 2>&1
cat quarrel_out.txt

echo -e "\n=== Running llama.cpp ==="
/opt/homebrew/bin/llama-cli -m "$MODEL_PATH" -p "$PROMPT" -n 30 --no-conversation --simple-io > llama_out.txt 2>&1
cat llama_out.txt
