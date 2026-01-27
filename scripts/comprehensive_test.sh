#!/bin/bash

# Configuration
MODELS=(
    "SmolLM2-135M:/Users/rsd/.ollama/models/blobs/sha256-f535f83ec568d040f88ddc04a199fa6da90923bbb41d4dcaed02caa924d6ef57"
    "TinyLlama-1.1B:/Users/rsd/.ollama/models/blobs/sha256-2af3b81862c6be03c769683af18efdadb2c33f60ff32ab6f83e42c043d6c7816"
    "Granite-3B:/Users/rsd/.ollama/models/blobs/sha256-5c7ac4aead1bcf4c8da9534ed72cc632d005aeed6547f1e8662ccdfae688364e"
    "Mistral-7B:/Users/rsd/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f"
)

PROMPT="What is the capital of France?"
TOKENS=16
OUTPUT_DIR="comprehensive_results_$(date +"%Y%m%d_%H%M%S")"
REPORT_FILE="$OUTPUT_DIR/bench_report.md"

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

echo "# Comprehensive Benchmarking Report" > "$REPORT_FILE"
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
    
    # 1. llama.cpp Performance
    echo "  [1/3] Running llama-bench..."
    L_JSON_RAW=$($LLAMABENCH -m "$PATH_VAL" -p 0 -n $TOKENS -b 1 --output json 2>/dev/null)
    # Extract JSON between [ and ]
    # Using avg_ts for throughput as observed in manual run
    L_PERF=$(echo "$L_JSON_RAW" | sed -n '/\[/,/\]/p' | $JQ -r '.[0].avg_ts' 2>/dev/null)
    [[ -z "$L_PERF" || "$L_PERF" == "null" ]] && L_PERF="0.00"
    
    # 2. llama.cpp Generation
    echo "  [2/3] Running llama-completion..."
    L_FULL_OUT=$($LLAMACLI -m "$PATH_VAL" -p "$PROMPT" -n $TOKENS --temp 0.0 --log-disable --simple-io --no-display-prompt < /dev/null 2>/dev/null)
    # Extract only the completion by removing the prompt and any subsequent noise
    # We use python for robust prefix stripping and whitespace normalization
    L_OUT=$($PYTHON -c "
import sys
full = sys.stdin.read().replace('\n', ' ')
prompt = sys.argv[1].replace('\n', ' ')
if full.startswith(prompt):
    full = full[len(prompt):]
# Strip trailing noise like '> EOF'
if '> EOF' in full:
    full = full.split('> EOF')[0]
print(full.strip())
" "$PROMPT" <<< "$L_FULL_OUT")
    
    # 3. Quarrel Performance & Generation
    echo "  [3/3] Running longbow-quarrel..."
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
    
    # 4. Compare
    SCORES=$(calculate_similarity "$L_OUT" "$Q_OUT")
    
    echo "| $NAME | llama.cpp | $L_PERF | - | \"${L_OUT:0:100}...\" |" >> "$REPORT_FILE"
    echo "| | longbow-quarrel | $Q_PERF | $SCORES | \"${Q_OUT:0:100}...\" |" >> "$REPORT_FILE"
    echo "| | | | | |" >> "$REPORT_FILE"
    
    echo "  Result: $Q_PERF t/s (Quarrel) vs $L_PERF t/s (llama.cpp)"
    echo "  Similarity: $SCORES"
    
    # Cleanup between runs
    sleep 2
    sync
done

echo "" >> "$REPORT_FILE"
echo "## Performance Discrepancy Analysis" >> "$REPORT_FILE"
echo "### SMOL-LM2 (135M)" >> "$REPORT_FILE"
echo "- Quarrel uses optimized Metal kernels for small models, typically F32 accumulation even in F16 paths to maintain precision." >> "$REPORT_FILE"
echo "### Mistral / Granite" >> "$REPORT_FILE"
echo "- llama.cpp often uses highly tuned SIMD and Metal Shaders (GGML_METAL_NODE_...) which might outperform Quarrel's generic dequantization kernels for GQA/Mistral architectures." >> "$REPORT_FILE"

echo "Testing complete. Results in $REPORT_FILE"
