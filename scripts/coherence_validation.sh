#!/bin/bash

# Coherence validation script for Linux/CPU
# Usage: ./scripts/coherence_validation.sh [model.gguf]

# Configuration
MODEL_PATH="${1:-$HOME/.cache/ollama/models/ggml-model.gguf}"
PROMPT="${2:-The quick brown fox jumps over the lazy dog}"
TOKENS="${3:-16}"
TEMPERATURE="${4:-0.7}"
OUTPUT_DIR="${5:-coherence_results}"

QUARREL_CMD="./cmd/benchmark/benchmark"
PYTHON="${PYTHON:-python3}"

mkdir -p "$OUTPUT_DIR"

echo "Coherence Validation (Linux/CPU)"
echo "Model: $MODEL_PATH"
echo "Prompt: $PROMPT"
echo ""

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model not found: $MODEL_PATH"
    exit 1
fi

run_gen() {
    local cmd="$1"
    local path="$2"
    local prompt="$3"
    local tokens="$4"
    local temp="$5"
    
    $cmd -model "$path" -prompt "$prompt" -tokens "$tokens" -temperature "$temp" 2>&1 | tail -10 | tr '\n' ' '
}

# Metrics calculation functions
calculate_bleu() {
    local ref="$1"
    local cand="$2"
    $PYTHON -c "
import re
import sys
from collections import Counter

def tokenize(text):
    return re.findall(r'\w+', text.lower())

ref_tokens = tokenize(sys.argv[1])
cand_tokens = tokenize(sys.argv[2])

if not ref_tokens:
    print('0.0')
    sys.exit(0)

ref_counts = Counter(ref_tokens)
cand_counts = Counter(cand_tokens)

precision = sum(min(cand_counts[token], ref_counts[token]) for token in cand_counts)
precision = precision / sum(cand_counts.values()) if cand_counts else 0
print(f'{precision:.4f}')
" "$ref" "$cand"
}

calculate_rouge() {
    local ref="$1"
    local cand="$2"
    $PYTHON -c "
import re
import sys

def tokenize(text):
    return re.findall(r'\w+', text.lower())

ref_tokens = tokenize(sys.argv[1])
cand_tokens = tokenize(sys.argv[2])

def lcs(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]: 
                dp[i][j] = dp[i-1][j-1] + 1
            else: 
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

l_len = lcs(ref_tokens, cand_tokens)
score = 2 * l_len / (len(ref_tokens) + len(cand_tokens)) if (len(ref_tokens) + len(cand_tokens)) > 0 else 0
print(f'{score:.4f}')
" "$ref" "$cand"
}

# Run 3 generations
echo "Generation 1:"
TEXT1=$(run_gen "$QUARREL_CMD" "$MODEL_PATH" "$PROMPT" "$TOKENS" "$TEMPERATURE")
echo "${TEXT1:0:100}..."
echo ""

echo "Generation 2:"
TEXT2=$(run_gen "$QUARREL_CMD" "$MODEL_PATH" "$PROMPT" "$TOKENS" "$TEMPERATURE")
echo "${TEXT2:0:100}..."
echo ""

echo "Generation 3:"
TEXT3=$(run_gen "$QUARREL_CMD" "$MODEL_PATH" "$PROMPT" "$TOKENS" "$TEMPERATURE")
echo "${TEXT3:0:100}..."
echo ""

# Calculate coherence metrics
BLEU=$(calculate_bleu "$TEXT1" "$TEXT2")
ROUGE=$(calculate_rouge "$TEXT1" "$TEXT2")

echo "Metrics:"
echo "  BLEU-1 (1 vs 2): $BLEU"
echo "  ROUGE-L (1 vs 2): $ROUGE"
echo ""

# Check consistency
CLEAN1=$(echo "$TEXT1" | sed 's/<[^>]*>//g' | tr -s ' ')
CLEAN2=$(echo "$TEXT2" | sed 's/<[^>]*>//g' | tr -s ' ')

if [[ "$CLEAN1" == "$CLEAN2" ]]; then
    echo "Consistency: PERFECT (identical outputs)"
else
    echo "Consistency: VARIABLE (expected with non-deterministic sampling)"
fi

echo ""
echo "Report: $OUTPUT_DIR/coherence_report.md"
cat > "$OUTPUT_DIR/coherence_report.md" << EOF
# Coherence Validation Report

**Date:** $(date)
**Model:** $MODEL_PATH
**Prompt:** $PROMPT

## Results

| Metric | Value |
|--------|-------|
| BLEU-1 | $BLEU |
| ROUGE-L | $ROUGE |
| Generations | 3 |

## Sample Outputs

1. ${TEXT1:0:200}...
2. ${TEXT2:0:200}...
3. ${TEXT3:0:200}...

## Conclusion

Coherence validation completed successfully.
EOF

echo "Done!"
