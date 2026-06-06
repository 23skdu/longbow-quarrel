#!/usr/bin/env bash
# Test Quarrel vs llama.cpp for token coherence and speed.
# Usage: ./scripts/test_vs_llamacpp.sh [model.gguf] [prompt]
set -euo pipefail

MODEL="${1:-}"
PROMPT="${2:-"The capital of France is"}"
N_TOKENS=50
TEMP=0.0

if [ -z "$MODEL" ]; then
  echo "Usage: $0 <model.gguf> [prompt]"
  echo ""
  echo "Downloads a small test model if none provided..."
  MODEL="/tmp/test_model.gguf"
  if [ ! -f "$MODEL" ]; then
    echo "Downloading Qwen2-0.5B-Instruct Q4_K_M..."
    wget -q -O "$MODEL" \
      "https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF/resolve/main/qwen2-0_5b-instruct-q4_k_m.gguf" || {
      echo "Download failed. Please provide a model path manually."
      echo "  $0 /path/to/model.gguf"
      exit 1
    }
  fi
fi

echo "========================================"
echo " Quarrel vs llama.cpp Comparison"
echo "========================================"
echo "Model:  $MODEL"
echo "Prompt: $PROMPT"
echo "Tokens: $N_TOKENS"
echo "Temp:   $TEMP"
echo ""

# --- Run Quarrel ---
echo "--- Quarrel ---"
QUARREL_BIN="./quarrel_cpu"
if [ -x "$QUARREL_BIN" ]; then
  QUARREL_CMD="$QUARREL_BIN"
else
  QUARREL_CMD="go run ./cmd/simple/"
fi
QUARREL_OUT=$(timeout 120 $QUARREL_CMD \
  --model "$MODEL" \
  --prompt "$PROMPT" \
  --n "$N_TOKENS" \
  --temp "$TEMP" \
  --topk 1 \
  --max-memory 4096 \
  2>&1 || echo "QUARREL_EXIT:$?")
echo "$QUARREL_OUT" | tail -5
QUARREL_TS=$(echo "$QUARREL_OUT" | grep -oP '[\d.]+(?= tokens/s)' || echo "N/A")

echo ""
echo "--- llama.cpp ---"
LLAMA_OUT=$(timeout 120 llama-cli \
  --model "$MODEL" \
  --prompt "$PROMPT" \
  --temp "$TEMP" \
  --seed 42 \
  --ctx-size 4096 \
  --predict "$N_TOKENS" \
  --no-display-prompt \
  --log-disable 2>&1 || echo "LLAMA_EXIT:$?")
LLAMA_TOKENS=$(echo "$LLAMA_OUT" | grep -oP '(?<=generated tokens = )\d+' || true)
LLAMA_TS=$(echo "$LLAMA_OUT" | grep -oP '[\d.]+(?= tokens/sec)' || true)
echo "$LLAMA_OUT" | tail -5

echo ""
echo "========================================"
echo " Results Summary"
echo "========================================"
echo "Metric              Quarrel     llama.cpp"
echo "----------------------------------------"
echo "Tokens generated    ${QUARREL_TOKENS:-N/A}        ${LLAMA_TOKENS:-N/A}"
echo "Speed (t/s)         ${QUARREL_TS:-N/A}       ${LLAMA_TS:-N/A}"
echo "----------------------------------------"

# Simple coherence check: compare token-per-line output if available
if command -v diff &>/dev/null; then
  Q_LINES=$(echo "$QUARREL_OUT" | grep -oP '[a-zA-Z0-9[:punct:] ]+' | head -20)
  L_LINES=$(echo "$LLAMA_OUT"  | grep -oP '(?<=\[end of text\]).*' | head -20 || echo "")
  if [ -n "$Q_LINES" ] && [ -n "$L_LINES" ]; then
    echo ""
    echo "Token-level coherence:"
    if [ "$Q_LINES" = "$L_LINES" ]; then
      echo "  ✓ PERFECT MATCH (all tokens identical)"
    else
      echo "  ✗ Partial match (models may differ due to floating-point rounding)"
    fi
  fi
fi
echo ""
echo "Done."
