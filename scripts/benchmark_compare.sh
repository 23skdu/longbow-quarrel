#!/bin/bash

# Configuration
LLAMA_BENCH="/opt/homebrew/bin/llama-bench"
QUARREL="./quarrel"
MODEL="/Users/rsd/.ollama/models/blobs/sha256-f535f83ec568d040f88ddc04a199fa6da90923bbb41d4dcaed02caa924d6ef57"
PROMPT="The capital of France is"
N_TOKENS=32
PROFILE="false"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -m|--model) MODEL="$2"; shift ;;
        -p|--prompt) PROMPT="$2"; shift ;;
        -n|--tokens) N_TOKENS="$2"; shift ;;
        --profile) PROFILE="true" ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "===================================================="
echo "Benchmarking: longbow-quarrel vs llama.cpp"
echo "Model: $MODEL"
echo "Prompt: '$PROMPT'"
echo "Tokens: $N_TOKENS"
echo "===================================================="

# 1. Run llama-bench (reference)
echo "[1/2] Running llama-bench (reference)..."
LLAMA_OUT=$( "$LLAMA_BENCH" -m "$MODEL" -p "$N_TOKENS" -n "$N_TOKENS" -r 1 2>&1 )
# Parse tg (text generation) t/s from llama-bench output (format: "tgN ... X.XX ± Y.YY")
LLAMA_TPS=$( echo "$LLAMA_OUT" | grep "tg" | tail -n 1 | awk '{for(i=1;i<=NF;i++) if($i ~ /^[0-9]+\.[0-9]+$/) {print $i; exit}}' )
if [ -z "$LLAMA_TPS" ]; then
    LLAMA_TPS="N/A"
fi

# 2. Run longbow-quarrel
echo "[2/2] Running longbow-quarrel..."
if [[ "$PROFILE" == "true" ]]; then
    # Run in background to capture pprof
    "$QUARREL" -model "$MODEL" -n 100 -prompt "$PROMPT" > quarrel.log 2>&1 &
    Q_PID=$!
    echo "Gathering pprof profile from quarrel (PID $Q_PID)..."
    sleep 2
    curl -s "http://localhost:9090/debug/pprof/profile?seconds=5" -o cpu.pprof &
    CURL_PID=$!
    wait $CURL_PID
    wait $Q_PID
    QUARREL_OUT=$(cat quarrel.log)
    rm -f quarrel.log
    echo "Profile saved to cpu.pprof"
else
    QUARREL_OUT=$( "$QUARREL" -model "$MODEL" -n "$N_TOKENS" -prompt "$PROMPT" 2>&1 )
fi

QUARREL_TPS=$( echo "$QUARREL_OUT" | grep "Inference complete" | sed -E 's/.*\(([0-9.]+) t\/s\).*/\1/' )
QUARREL_TEXT=$( echo "$QUARREL_OUT" | grep "Decoded Text:" | sed 's/.*Decoded Text: //' )

echo ""
echo "RESULTS:"
echo "----------------------------------------------------"
printf "%-20s | %-15s | %s\n" "Implementation" "Perf (t/s)" "Output Snippet"
printf "%-20s | %-15s | %s\n" "----------------" "----------" "--------------"
printf "%-20s | %-15s | %s\n" "llama.cpp (bench)" "$LLAMA_TPS" "(benchmark only)"
printf "%-20s | %-15s | %s\n" "longbow-quarrel" "$QUARREL_TPS" "${QUARREL_TEXT:0:50}..."
echo "----------------------------------------------------"

# Performance Comparison
if [ -n "$LLAMA_TPS" ] && [ -n "$QUARREL_TPS" ]; then
    RATIO=$(echo "scale=2; $QUARREL_TPS / $LLAMA_TPS * 100" | bc)
    echo ""
    echo "PERFORMANCE:"
    echo "longbow-quarrel is at ${RATIO}% of llama.cpp throughput."
fi
