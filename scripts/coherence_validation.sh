#!/bin/bash

# Configuration
REFERENCE_IMPL="${1:-longbow-quarrel}"
TEST_IMPL="${2:-llama.cpp}"
MODEL_PATH="$3"
PROMPT="${4:-The quick brown fox jumps over the lazy dog}"
TOKENS="${5:-16}"
TEMPERATURE="${6:-0.7}"
OUTPUT_DIR="coherence_results"

# Tools
LLAMACPP_CLI="/opt/homebrew/bin/llama-cli"
QUARREL_CMD="./bin/metal_benchmark"
PYTHON="/opt/homebrew/bin/python3"
JQ="/usr/bin/jq"

mkdir -p "$OUTPUT_DIR"

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

if not ref_tokens:
    print('0.0')
    sys.exit(0)

def lcs(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]: dp[i][j] = dp[i-1][j-1] + 1
            else: dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

l_len = lcs(ref_tokens, cand_tokens)
score = 2 * l_len / (len(ref_tokens) + len(cand_tokens)) if (len(ref_tokens) + len(cand_tokens)) > 0 else 0
print(f'{score:.4f}')
" "$ref" "$cand"
}

run_gen() {
    local impl="$1"
    local path="$2"
    if [[ "$impl" == "longbow-quarrel" ]]; then
        # Capture last line (JSON)
        $QUARREL_CMD -model "$path" -prompt "$PROMPT" -tokens $TOKENS -output json 2>/dev/null | tail -n 1 | $JQ -r '.output'
    else
        # llama-cli, simple one-shot
        $LLAMACPP_CLI -m "$path" -p "$PROMPT" -n $TOKENS --temp $TEMPERATURE --log-disable -c 1024 2>/dev/null | grep -v ">" | tr -d '\r' | tr '\n' ' '
    fi
}

echo "Coherence Validation"
echo "Model: $MODEL_PATH"
echo "Prompt: $PROMPT"

REF_PATH="/Users/rsd/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f"

REF_TEXT=$(run_gen "llama.cpp" "$REF_PATH")
echo "Reference (llama.cpp): $REF_TEXT"

TEST_TEXT=$(run_gen "longbow-quarrel" "$REF_PATH")
echo "Test (longbow-quarrel): $TEST_TEXT"

BLEU=$(calculate_bleu "$REF_TEXT" "$TEST_TEXT")
ROUGE=$(calculate_rouge "$REF_TEXT" "$TEST_TEXT")

echo "----------------------------------------"
echo "BLEU Score: $BLEU"
echo "ROUGE Score: $ROUGE"

REPORT="$OUTPUT_DIR/coherence_report.md"
echo "# Coherence Report" > "$REPORT"
echo "BLEU: $BLEU" >> "$REPORT"
echo "ROUGE: $ROUGE" >> "$REPORT"
echo "Reference: $REF_TEXT" >> "$REPORT"
echo "Test: $TEST_TEXT" >> "$REPORT"

echo "Validation complete. Report: $REPORT"