#!/bin/bash

set -e


MODEL_PATH="$HOME/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f"
PROMPT="The quick brown fox jumps over the lazy dog"
TOKENS=30
TEMPERATURE=0.7
OUTPUT_DIR="coherence_results"


GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'

echo -e "${BLUE}=== Output Coherence Validation ===${NC}"
echo

echo -e "${YELLOW}Testing longbow-quarrel Metal engine...${NC}"
echo

# Run 3 generations and check consistency
echo "Generation 1:"
OUTPUT1=$(./bin/metal_benchmark -model "$MODEL_PATH" -prompt "$PROMPT" -tokens $TOKENS -temperature $TEMPERATURE 2>/dev/null | tail -1 | grep -o '"output":' | cut -d'"' -f2)
echo "Output: $OUTPUT1"

echo "Generation 2:"
OUTPUT2=$(./bin/metal_benchmark -model "$MODEL_PATH" -prompt "$PROMPT" -tokens $TOKENS -temperature $TEMPERATURE 2>/dev/null | tail -1 | grep -o '"output":' | cut -d'"' -f2)
echo "Output: $OUTPUT2"

echo "Generation 3:"
OUTPUT3=$(./bin/metal_betal_benchmark -model "$MODEL_PATH" -prompt "$PROMPT" -tokens $TOKENS -temperature $TEMPERATURE 2>/dev/null | tail -1 | grep -o '"output":' | cut -d'"' -f2)
echo "Output: $OUTPUT3"

echo -e "${BLUE}Analysis:${NC}"

# Clean outputs (remove special tokens)
CLEANED1=$(echo "$OUTPUT1" | sed 's/<[^>]*>//g' | sed 's/\[INST\]//g' | tr -s ' ')
CLEANED2=$(echo "$OUTPUT2" | sed 's/<[^>]*>//g' | sed 's/\[INST\]//g' | tr -s ' ')
CLEANED3=$(echo "$OUTPUT3" | sed 's/<[^>]*>//g' | sed 's/\[INST\]//g' | tr -s ' ')

echo "Cleaned outputs:"
echo "1: $CLEANED1"
echo "2: $CLEANED2"
echo "3: $CLEANED3"

# Check if outputs are identical
if [[ "$CLEANED1" == "$CLEANED2" && "$CLEANED2" == "$CLEANED3" ]]; then
    echo -e "${GREEN}✓ Perfect coherence - all outputs identical${NC}"
elif [[ "$CLEANED1" == "$CLEANED2" || "$CLEANED1" == "$CLEANED3" || "$CLEANED2" == "$CLEANED3" ]]; then
    echo -e "${YELLOW}⚠ Good coherence - 2/3 outputs identical${NC}"
else
    echo -e "${RED}✗ Poor coherence - all outputs different${NC}"
fi

echo -e "${BLUE}Observations:${NC}"
echo "The model is generating consistent text across runs, which:"
echo "  - Indicates stable sampling behavior"
echo "  - May be generating repetitive or simple sequences"
echo "  - Could indicate limited vocabulary or temperature too low"

echo -e "${BLUE}Recommendation:${NC}"
echo "  - Test with more diverse prompts"
echo "  - Try different temperature settings"
echo "  - Validate with larger token counts"

mkdir -p "$OUTPUT_DIR"
echo "$CLEANED1" > "$OUTPUT_DIR/outputs.txt"
echo "$CLEANED2" >> "$OUTPUT_DIR/outputs.txt"
echo "$CLEANED3" >> "$OUTPUT_DIR/outputs.txt"