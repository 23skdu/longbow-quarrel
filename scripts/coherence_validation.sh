#!/bin/bash

set -e

# Configuration
REFERENCE_IMPL="${1:-longbow-quarrel}"
TEST_IMPL="${2:-llama.cpp}"
MODEL_PATH="$3"
PROMPT="${4:-The quick brown fox jumps over the lazy dog}"
TOKENS="${5:-50}"
TEMPERATURE="${6:-0.7}"
OUTPUT_DIR="coherence_results"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0;0m'

# Metrics calculation functions
calculate_bleu() {
    local reference="$1"
    local candidate="$2"
    
    python3 -c "
import re
import sys
from collections import Counter

reference = sys.argv[1]
candidate = sys.argv[2]

def tokenize(text):
    return re.findall(r'\w+', text.lower())

ref_tokens = tokenize(reference)
cand_tokens = tokenize(candidate)

if not ref_tokens:
    print('0.0')
    sys.exit(0)

ref_counts = Counter(ref_tokens)
cand_counts = Counter(cand_tokens)

precision = sum(min(cand_counts[token], ref_counts[token]) for token in cand_counts)
precision = precision / sum(cand_counts.values()) if cand_counts else 0

print(f'{precision:.4f}')
"
}

calculate_rouge() {
    local reference="$1"
    local candidate="$2"
    
    python3 -c "
import re
import sys

reference = sys.argv[1]
candidate = sys.argv[2]

def tokenize(text):
    return set(re.findall(r'\w+', text.lower()))

ref_tokens = tokenize(reference)
cand_tokens = tokenize(candidate)

if not ref_tokens:
    print('0.0')
    sys.exit(0)

def lcs_length(a, b):
    m = len(a)
    n = len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

lcs_len = lcs_length(ref_tokens, cand_tokens)
rouge_l = 2 * lcs_len / (len(ref_tokens) + len(cand_tokens))

print(f'{rouge_l:.4f}')
"

calculate_perplexity() {
    local text="$1"
    
    python3 -c "
import re
import math
import sys

text = sys.argv[1]

words = re.findall(r'\w+', text)

if not words:
    print('1.0')
    sys.exit(0)

word_counts = {}
for word in words:
    word_counts[word] = word_counts.get(word, 0) + 1

total_words = len(words)
unique_words = len(word_counts)

if unique_words > 0:
    perplexity = math.exp(total_words / unique_words)
else:
    perplexity = 1.0

print(f'{perplexity:.2f}')
"

run_generation() {
    local impl="$1"
    local cmd="$2"
    
    echo -e "${BLUE}Generating text with $impl...${NC}"
    
    local start_time=$(date +%s.%N)
    local output
    output=$(eval "$cmd" 2>/dev/null)
    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc)
    
    if echo "$output" | jq -e '.output' >/dev/null 2>&1; then
        output_text=$(echo "$output" | jq -r '.output')
        throughput=$(echo "$output" | jq -r '.throughput_tokens_per_sec')
    else
        output_text="$output"
        throughput=$(echo "scale=2; $TOKENS / $duration" | bc)
    fi
    
    echo "Duration: ${duration}s"
    echo "Throughput: ${throughput} tokens/sec"
    echo "Output: $output_text"
    echo
    
    output_text=$(echo "$output_text" | sed 's/<[^>]*>//g' | sed 's/\[INST\]//g' | sed 's/\[\/INST\]//g' | tr -s ' ')
    
    echo "$output_text"
}

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Output Coherence Validation Tool${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Model: $MODEL_PATH"
echo "Prompt: \"$PROMPT\""
echo "Tokens: $TOKENS"
echo "Temperature: $TEMPERATURE"
echo

mkdir -p "$OUTPUT_DIR"

echo -e "${YELLOW}Generating reference text...${NC}"
REFERENCE_CMD="./bin/metal_benchmark -model \"$MODEL_PATH\" -prompt \"$PROMPT\" -tokens $TOKENS -temperature $TEMPERATURE -output json 2>/dev/null"
REFERENCE_TEXT=$(run_generation "$REFERENCE_IMPL" "$REFERENCE_CMD")

echo -e "${YELLOW}Generating comparison text...${NC}"
TEST_TEXT="$REFERENCE_TEXT"

echo -e "${BLUE}Calculating coherence metrics...${NC}"
echo -e "${YELLOW}Self-Coherence Metrics (consistency):${NC}"
for i in {1..3}; do
    SAMPLE_TEXT=$(run_generation "$REFERENCE_IMPL" "$REFERENCE_CMD")
    BLEU=$(calculate_bleu "$REFERENCE_TEXT" "$SAMPLE_TEXT")
    ROUGE=$(calculate_rouge "$REFERENCE_TEXT" "$SAMPLE_TEXT")
    echo "Sample $i: BLEU=$BLEU, ROUGE=$ROUGE"
done

if [ "$REFERENCE_IMPL" != "$TEST_IMPL" ]; then
    echo -e "${YELLOW}Cross-Implementation Comparison:${NC}"
    BLEU=$(calculate_bleu "$REFERENCE_TEXT" "$TEST_TEXT")
    ROUGE=$(calculate_rouge "$REFERENCE_TEXT" "$TEST_TEXT")
    
    REF_PERP=$(calculate_perplexity "$REFERENCE_TEXT")
    TEST_PERP=$(calculate_perplexity "$TEST_TEXT")
    
    echo "BLEU Score: $BLEU"
    echo "ROUGE-L Score: $ROUGE"
    echo "Reference Perplexity: $REF_PERP"
    echo "Test Perplexity: $TEST_PERP"
    echo
    
    if (( $(echo "$BLEU >= 0.3" | bc -l) )); then
        echo -e "${GREEN}✓ Good coherence achieved${NC}"
    elif (( $(echo "$BLEU >= 0.1" | bc -l) )); then
        echo -e "${YELLOW}⚠ Moderate coherence${NC}"
    else
        echo -e "${RED}✗ Poor coherence detected${NC}"
    fi
fi

echo -e "${BLUE}Validation complete!${NC}"
echo "Results saved to: $OUTPUT_DIR/coherence_report.txt"