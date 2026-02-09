#!/bin/bash

# Custom Coherence Test Script for Linux/CPU
# Models: tinyllama:latest, nemotron-3-nano:latest, smollm:135m

set -e

# Configuration
MODELS=(
    "TinyLlama-1.1B:$HOME/.cache/ollama/models/ggml-model.gguf"
    "Nemotron-3-Nano:$HOME/.cache/ollama/models/ggml-nemotron.gguf"
    "SmolLM-135M:$HOME/.cache/ollama/models/ggml-smollm.gguf"
)

PROMPT="What is the capital of France?"
TOKENS=16
TEMPERATURE=0.7
OUTPUT_DIR="coherence_results_$(date +"%Y%m%d_%H%M%S")"
REPORT_FILE="$OUTPUT_DIR/coherence_report.md"

QUARREL_CMD="./cmd/benchmark/benchmark"
PYTHON="${PYTHON:-python3}"

mkdir -p "$OUTPUT_DIR"

echo "# Custom Coherence Test Report" > "$REPORT_FILE"
echo "Generated at $(date)" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "**Prompt:** $PROMPT" >> "$REPORT_FILE"
echo "**Tokens:** $TOKENS" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "| Model | Engine | Throughput (t/s) | Coherence | Sample Output |" >> "$REPORT_FILE"
echo "|-------|--------|----------------|-----------|---------------|" >> "$REPORT_FILE"

calculate_similarity() {
    local ref="$1"
    local cand="$2"
    $PYTHON -c "
import sys
import re
from collections import Counter

def tokenize(text): return re.findall(r'\w+', text.lower())

ref_tokens = tokenize(sys.argv[1])
cand_tokens = tokenize(sys.argv[2])

if not ref_tokens:
    print('0.00')
    sys.exit(0)

ref_counts = Counter(ref_tokens)
cand_counts = Counter(cand_tokens)
precision = sum(min(cand_counts[token], ref_counts[token]) for token in cand_counts) / max(1, sum(cand_counts.values()))

m, n = len(ref_tokens), len(cand_tokens)
dp = [[0] * (n + 1) for _ in range(m + 1)]
for i in range(1, m + 1):
    for j in range(1, n + 1):
        if ref_tokens[i-1] == cand_tokens[j-1]: dp[i][j] = dp[i-1][j-1] + 1
        else: dp[i][j] = max(dp[i-1][j], dp[i][j-1])
lcs = dp[m][n]
rouge = 2 * lcs / (m + n) if (m + n) > 0 else 0

print(f'{precision:.2f}|{rouge:.2f}')
" "$ref" "$cand"
}

run_gen() {
    $QUARREL_CMD -model "$1" -prompt "$PROMPT" -tokens $TOKENS -temperature $TEMPERATURE 2>&1 | tail -10 | tr '\n' ' '
}

for entry in "${MODELS[@]}"; do
    NAME="${entry%%:*}"
    PATH_VAL="${entry#*:}"
    
    echo "=== Testing $NAME ==="
    
    if [ ! -f "$PATH_VAL" ]; then
        echo "  Model not found: $PATH_VAL"
        echo "| $NAME | longbow-quarrel | N/A | N/A | Model not found |" >> "$REPORT_FILE"
        continue
    fi
    
    # Run 2 generations for coherence check
    echo "  [1/2] Generation 1..."
    OUT1=$(run_gen "$PATH_VAL")
    
    sleep 1
    
    echo "  [2/2] Generation 2..."
    OUT2=$(run_gen "$PATH_VAL")
    
    # Calculate similarity
    SCORES=$(calculate_similarity "$OUT1" "$OUT2")
    PRECISION="${SCORES%%|*}"
    ROUGES="${SCORES##*|}"
    
    echo "| $NAME | longbow-quarrel | - | ${PRECISION}/${ROUGES} | ${OUT1:0:100}... |" >> "$REPORT_FILE"
    
    echo "  Result: Precision=$PRECISION, Rouge=$ROUGES"
    echo ""
done

echo "Testing complete. Results in $REPORT_FILE"
echo ""
cat "$REPORT_FILE"
