#!/bin/bash

# Configuration
MODELS=(
    "Mistral:/Users/rsd/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f"
    "Granite4:/Users/rsd/.ollama/models/blobs/sha256-5c7ac4aead1bcf4c8da9534ed72cc632d005aeed6547f1e8662ccdfae688364e"
)
PROMPT="The quick brown fox jumps over the lazy dog"
TOKENS=16
OUTPUT_DIR="benchmark_results_final"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_FILE="$OUTPUT_DIR/benchmark_report.md"

# Tools - Use absolute paths
LLAMABENCH="/opt/homebrew/bin/llama-bench"
QUARREL_BENCH="./bin/metal_benchmark"
JQ="/usr/bin/jq"
GREP="/usr/bin/grep"
AWK="/usr/bin/awk"
CAT="/bin/cat"

mkdir -p "$OUTPUT_DIR"

echo "# Performance Benchmark Results ($TIMESTAMP)" > "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "Prompt: $PROMPT" >> "$REPORT_FILE"
echo "Tokens: $TOKENS" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "| Model | Engine | Throughput (t/s) |" >> "$REPORT_FILE"
echo "|---|---|---|" >> "$REPORT_FILE"

run_quarrel() {
    local name=$1
    local path=$2
    echo "Running longbow-quarrel ($name)..."
    # Run and capture ONLY the JSON part (last line)
    local output=$($QUARREL_BENCH -model "$path" -prompt "$PROMPT" -tokens $TOKENS -output json 2>&1 | tail -n 1)
    
    local throughput=$(echo "$output" | $JQ -r '.throughput_tokens_per_sec' 2>/dev/null)
    
    if [[ -z "$throughput" || "$throughput" == "null" ]]; then
        throughput="0.0"
    fi
    printf "  Throughput: %.2f t/s\n" "$throughput"
    echo "| $name | longbow-quarrel | $throughput |" >> "$REPORT_FILE"
}

run_llama_bench() {
    local name=$1
    local path=$2
    echo "Running llama-bench ($name)..."
    
    # Use JSON output for llama-bench if possible
    local output=$($LLAMABENCH -m "$path" -p 0 -n $TOKENS -b 1 --output json 2>/dev/null)
    local throughput=$(echo "$output" | $JQ -r '.[0].results[0].tps' 2>/dev/null)
    
    if [[ -z "$throughput" || "$throughput" == "null" ]]; then
        # Fallback to text parsing
        output=$($LLAMABENCH -m "$path" -p 0 -n $TOKENS -b 1 2>&1)
        # Search line with the result and extract throughput
        throughput=$(echo "$output" | $GREP -E "tg$TOKENS|test gen" | $AWK -F'|' '{print $9}' | $AWK '{print $1}')
    fi

    if [[ -z "$throughput" || "$throughput" == "null" ]]; then throughput="0.0"; fi
    printf "  Throughput: %.2f t/s\n" "$throughput"
    echo "| $name | llama.cpp (bench) | $throughput |" >> "$REPORT_FILE"
}

for entry in "${MODELS[@]}"; do
    NAME="${entry%%:*}"
    PATH_VAL="${entry#*:}"
    
    run_quarrel "$NAME" "$PATH_VAL"
    
    if [[ -f "$LLAMABENCH" ]]; then
        run_llama_bench "$NAME" "$PATH_VAL"
    fi
    echo "" >> "$REPORT_FILE"
done

echo "" >> "$REPORT_FILE"
echo "Benchmark complete."
$CAT "$REPORT_FILE"
