#!/bin/bash

# Simple coherence check script for Linux/CPU
# Usage: ./scripts/simple_coherence_check.sh [model.gguf]

set -e

MODEL_PATH="${1:-$HOME/.cache/ollama/models/ggml-model.gguf}"
PROMPT="The quick brown fox jumps over the lazy dog"
TOKENS=30
TEMPERATURE=0.7
OUTPUT_DIR="coherence_results"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}=== Output Coherence Validation (Linux/CPU) ===${NC}"
echo

mkdir -p "$OUTPUT_DIR"

echo -e "${YELLOW}Testing longbow-quarrel CPU engine...${NC}"

if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}Model not found: $MODEL_PATH${NC}"
    echo "Please provide a valid GGUF model path"
    exit 1
fi

echo "Generation 1:"
OUTPUT_FILE1="$OUTPUT_DIR/output_1.txt"
./cmd/benchmark/benchmark -model "$MODEL_PATH" -prompt "$PROMPT" -tokens $TOKENS -temperature $TEMPERATURE 2>&1 | tail -10 | tee "$OUTPUT_FILE1"
echo -e "${GREEN}✓ Generation 1 completed${NC}"

echo "Generation 2:"
OUTPUT_FILE2="$OUTPUT_DIR/output_2.txt"
./cmd/benchmark/benchmark -model "$MODEL_PATH" -prompt "$PROMPT" -tokens $TOKENS -temperature $TEMPERATURE 2>&1 | tail -10 | tee "$OUTPUT_FILE2"
echo -e "${GREEN}✓ Generation 2 completed${NC}"

echo "Generation 3:"
OUTPUT_FILE3="$OUTPUT_DIR/output_3.txt"
./cmd/benchmark/benchmark -model "$MODEL_PATH" -prompt "$PROMPT" -tokens $TOKENS -temperature $TEMPERATURE 2>&1 | tail -10 | tee "$OUTPUT_FILE3"
echo -e "${GREEN}✓ Generation 3 completed${NC}"

echo -e "${BLUE}Analysis:${NC}"

OUTPUT1=$(cat "$OUTPUT_FILE1" | grep -v "^===" | grep -v "^---" | tail -5 | tr '\n' ' ')
OUTPUT2=$(cat "$OUTPUT_FILE2" | grep -v "^===" | grep -v "^---" | tail -5 | tr '\n' ' ')
OUTPUT3=$(cat "$OUTPUT_FILE3" | grep -v "^===" | grep -v "^---" | tail -5 | tr '\n' ' ')

CLEANED1=$(echo "$OUTPUT1" | sed 's/<[^>]*>//g' | tr -s ' ')
CLEANED2=$(echo "$OUTPUT2" | sed 's/<[^>]*>//g' | tr -s ' ')
CLEANED3=$(echo "$OUTPUT3" | sed 's/<[^>]*>//g' | tr -s ' ')

echo "Cleaned outputs:"
echo "1: ${CLEANED1:0:100}..."
echo "2: ${CLEANED2:0:100}..."
echo "3: ${CLEANED3:0:100}..."

if [[ "$CLEANED1" == "$CLEANED2" && "$CLEANED2" == "$CLEANED3" ]]; then
    echo -e "${GREEN}✓ Perfect coherence - all outputs identical${NC}"
elif [[ "$CLEANED1" == "$CLEANED2" || "$CLEANED1" == "$CLEANED3" || "$CLEANED2" == "$CLEANED3" ]]; then
    echo -e "${YELLOW}⚠ Good coherence - 2/3 outputs identical${NC}"
else
    echo -e "${YELLOW}⚠ Variable outputs - expected with non-deterministic sampling${NC}"
fi

echo -e "${BLUE}Recommendation:${NC}"
echo "  - Test with more diverse prompts"
echo "  - Try different temperature settings"
echo "  - Validate with larger token counts"

echo "$CLEANED1" > "$OUTPUT_DIR/outputs.txt"
echo "$CLEANED2" >> "$OUTPUT_DIR/outputs.txt"
echo "$CLEANED3" >> "$OUTPUT_DIR/outputs.txt"

echo ""
echo -e "${BLUE}Results saved to: $OUTPUT_DIR/outputs.txt ===${NC}"
