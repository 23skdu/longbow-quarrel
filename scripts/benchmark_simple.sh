#!/bin/bash

# Simple benchmark script to compare longbow-quarrel vs llama.cpp
MODEL="smollm2:135m"
PROMPT="The capital of France is"
N_TOKENS=32

echo "=== Benchmark Comparison ==="
echo "Model: $MODEL"
echo "Prompt: '$PROMPT'"
echo "Tokens: $N_TOKENS"
echo ""

echo "1. Running llama.cpp reference..."
LLAMA_OUT=$(timeout 30s llama-bench -m "$MODEL" -p "$PROMPT" -n "$N_TOKENS" -t 1 2>/dev/null || echo "llama.cpp failed" || echo "llama.cpp failed")

if [ -n "$LLAMA_OUT" ]; then
    echo "❌ llama.cpp benchmark failed"
    exit 1
fi

echo "2. Running longbow-quarrel..."
QUARREL_OUT=$(timeout 30s ./cmd/quarrel/quarrel -model "$MODEL" -p "$PROMPT" -n "$N_TOKENS" -t 1 2>/dev/null || echo "longbow-quarrel failed" || echo "longbow-quarrel failed")

if [ -n "$QUARREL_OUT" ]; then
    echo "❌ longbow-quarrel benchmark failed"
    exit 1
fi

echo ""
echo "3. Results:"

# Extract t/s from llama.cpp output
LLAMA_TPS=$(echo "$LLAMA_OUT" | grep "tokens/s" | tail -1 | awk '{print $2}')
echo "llama.cpp: $LLAMA_TPS tokens/s"

# Extract t/s from longbow-quarrel output  
QUARREL_TPS=$(echo "$QUARREL_OUT" | grep "user" | tail -1 | awk '{print $6}')
echo "longbow-quarrel: $QUARREL_TPS tokens/s"

echo ""
echo "Performance Comparison:"
if [ -n "$LLAMA_TPS" ] && [ -n "$QUARREL_TPS" ]; then
    RATIO=$(echo "scale=4; $QUARREL_TPS / $LLAMA_TPS" | bc -l)
    if [ $(echo "$RATIO < 1" | bc -l) -eq 1 ]; then
        echo "🟢 longbow-quarrel: $QUARREL_TPS t/s ($RATIO% of llama.cpp)"
    elif [ $(echo "$RATIO < 0.5" | bc -l) -eq 1 ]; then
        echo "🟡 longbow-quarrel: $QUARREL_TPS t/s ($RATIO% of llama.cpp)"
    else
        echo "🔴 longbow-quarrel: $QUARREL_TPS t/s ($RATIO% of llama.cpp)"
    fi
else
    echo "📊 Performance comparison failed"
fi
else
    echo "📊 Performance comparison failed"
fi

echo ""
echo "4. Performance Ratio: $(echo "$RATIO" | bc -l 2>/dev/null || echo "calculation failed")" || echo "calculation failed"
echo ""