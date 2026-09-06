#!/bin/bash
# Coherence test for longbow-quarrel Qwen 3.5 model
# Tests deterministic output and intelligibility

set -uo pipefail

MODEL="/home/rsd/.cache/llmfit/models/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated.Q8_0.gguf"
QUARREL="./bin/quarrel-test"
OUTPUT_DIR="coherence_results_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTPUT_DIR"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=== longbow-quarrel Coherence Test ==="
echo "Model: $MODEL"
echo ""

# Test prompts that work with the Qwen 3.5 model
PROMPTS=(
    "Hello"
    "The" 
    "I"
    "A"
    "The capital of France is"
)

# Test with temperature 0 (deterministic)
TEMP=0
TOKENS=5

echo "Testing $TEMP temperature (deterministic)..."
echo ""

# Test each prompt across 3 seeds for consistency
for prompt in "${PROMPTS[@]}"; do
    echo "--- Prompt: '$prompt' ---"
    RESULTS=()
    
    for seed in 0 42 123; do
        # Run quarantine test and extract the generated text
        OUTPUT=$($QUARREL --model "$MODEL" --prompt "$prompt" --n $TOKENS --temp $TEMP --seed $seed 2>&1 | grep "Full output:" | head -1)
        if [ -n "$OUTPUT" ]; then
            # Extract just the generated text part
            GENERATED=$(echo "$OUTPUT" | sed 's/Full output: //')
            RESULTS+=("$seed: $GENERATED")
            echo "  seed=$seed: $GENERATED"
        else
            RESULTS+=("$seed: (no output)")
            echo "  seed=$seed: (no output or error)"
        fi
    done
    
    # Check consistency
    FIRST=$(echo "${RESULTS[0]}" | sed 's/^[^:]*://')
    CONSISTENT=true
    for r in "${RESULTS[@]:1}"; do
        CURRENT=$(echo "$r" | sed 's/^[^:]*://')
        if [ "$CURRENT" != "$FIRST" ]; then
            CONSISTENT=false
        fi
    done
    
    if [ "$CONSISTENT" = true ]; then
        echo -e "  ${GREEN}Consistent across seeds${NC}"
    else
        echo -e "  ${YELLOW}Variable output (expected with sampling)${NC}"
    fi
    
    echo ""
done

echo "=== Results saved to $OUTPUT_DIR ==="
echo "Run: cat $OUTPUT_DIR/* to see detailed output"
