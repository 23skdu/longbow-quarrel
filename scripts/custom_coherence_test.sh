#!/bin/bash

# Custom Coherence Test Script
# Models: tinyllama:latest, nemotron-3-nano:latest, smollm:135m

# Configuration
MODELS=(
    "TinyLlama-1.1B:/Users/rsd/.ollama/models/blobs/sha256-2af3b81862c6be03c769683af18efdadb2c33f60ff32ab6f83e42c043d6c7816"
    "Nemotron-3-Nano:/Users/rsd/.ollama/models/blobs/sha256-a70437c41b3b0b768c48737e15f8160c90f13dc963f5226aabb3a160f708d1ce"
    "SmolLM-135M:/Users/rsd/.ollama/models/blobs/sha256-eb2c714d40d4b35ba4b8ee98475a06d51d8080a17d2d2a75a23665985c739b94"
)

PROMPT="What is the capital of France?"
TOKENS=16
OUTPUT_DIR="coherence_results_$(date +"%Y%m%d_%H%M%S")"
REPORT_FILE="$OUTPUT_DIR/coherence_report.md"

# Tools
LLAMABENCH="/opt/homebrew/bin/llama-bench"
LLAMACLI="/opt/homebrew/bin/llama-completion"
QUARREL_CMD="./bin/metal_benchmark"
JQ="/usr/bin/jq"
PYTHON="/opt/homebrew/bin/python3"

mkdir -p "$OUTPUT_DIR"

# Score functions
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
    print('0.00|0.00')
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

echo "# Custom Coherence Test Report" > "$REPORT_FILE"
echo "Generated at $(date)" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "**Prompt:** $PROMPT" >> "$REPORT_FILE"
echo "**Tokens:** $TOKENS" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "| Model | Engine | Throughput (t/s) | Coherence (Prec/LCS) | Sample Output |" >> "$REPORT_FILE"
echo "|---|---|---|---|---|" >> "$REPORT_FILE"

for entry in "${MODELS[@]}"; do
    NAME="${entry%%:*}"
    PATH_VAL="${entry#*:}"
    
    echo "=== Testing $NAME ==="
    
    # 1. llama.cpp Generation (Reference)
    echo "  [1/2] Running llama-completion (Reference)..."
    L_FULL_OUT=$($LLAMACLI -m "$PATH_VAL" -p "$PROMPT" -n $TOKENS --temp 0.0 --log-disable --simple-io --no-display-prompt < /dev/null 2>/dev/null)
    # Extract only the completion
    L_OUT=$($PYTHON -c "
import sys
full = sys.stdin.read().replace('\n', ' ')
prompt = sys.argv[1].replace('\n', ' ')
if full.startswith(prompt):
    full = full[len(prompt):]
if '> EOF' in full:
    full = full.split('> EOF')[0]
print(full.strip())
" "$PROMPT" <<< "$L_FULL_OUT")
    
    # 2. Quarrel Generation
    echo "  [2/2] Running longbow-quarrel..."
    Q_RAW_OUT=$($QUARREL_CMD -model "$PATH_VAL" -prompt "$PROMPT" -tokens $TOKENS -output json 2>&1)
    
    # Extract last JSON object
    Q_JSON=$(echo "$Q_RAW_OUT" | grep "^{.*}$" | tail -n 1)
    if [[ -z "$Q_JSON" ]]; then
        echo "    ERROR: Quarrel output is not JSON. Raw: $(echo "$Q_RAW_OUT" | tail -n 2)"
        Q_PERF="ERROR"
        Q_OUT="ERROR"
    else
        Q_PERF=$(echo "$Q_JSON" | $JQ -r '.throughput_tokens_per_sec' 2>/dev/null)
        Q_OUT=$(echo "$Q_JSON" | $JQ -r '.output' 2>/dev/null | tr '\n' ' ' | sed 's/  */ /g')
        [ -z "$Q_PERF" ] || [ "$Q_PERF" == "null" ] && Q_PERF="0.00"
    fi
    
    # 3. Compare
    SCORES=$(calculate_similarity "$L_OUT" "$Q_OUT")
    
    echo "| $NAME | llama.cpp | - | - | \"${L_OUT:0:100}...\" |" >> "$REPORT_FILE"
    echo "| | longbow-quarrel | $Q_PERF | $SCORES | \"${Q_OUT:0:100}...\" |" >> "$REPORT_FILE"
    echo "| | | | | |" >> "$REPORT_FILE"
    
    echo "  Result: $Q_PERF t/s"
    echo "  Similarity: $SCORES"
    
    # Cleanup between runs
    sleep 2
    sync
done

echo "Testing complete. Results in $REPORT_FILE"
cat "$REPORT_FILE"
