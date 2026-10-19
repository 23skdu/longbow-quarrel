#!/bin/bash

# Configuration
# Override any of these from the environment or with CLI flags below, e.g.
#   MODEL=/path/to/model.gguf ./scripts/benchmark_compare.sh
LLAMA_BENCH="${LLAMA_BENCH:-/opt/homebrew/bin/llama-bench}"
QUARREL="${QUARREL:-./quarrel}"
MODEL="${MODEL:-}"
PROMPT="The capital of France is"
N_TOKENS=32
PROFILE="false"

# Parse arguments FIRST.
#
# The MODEL check used to run before argv was parsed, so the documented
# invocation
#   ./scripts/benchmark_compare.sh -m /path/to/model.gguf -p "..." -n 32
# always exited 2 -- the script could never be run with flags. CI called it
# exactly that way, so the benchmarks job failed.
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -m|--model) MODEL="$2"; shift ;;
        -p|--prompt) PROMPT="$2"; shift ;;
        -n|--tokens) N_TOKENS="$2"; shift ;;
        --profile) PROFILE="true" ;;
        *) echo "Unknown parameter passed: $1" >&2; exit 1 ;;
    esac
    shift
done

if [[ -z "$MODEL" ]]; then
    echo "error: set MODEL to the GGUF file to benchmark, e.g." >&2
    echo "  MODEL=/path/to/model.gguf $0" >&2
    echo "  $0 -m /path/to/model.gguf -p 'prompt' -n 32" >&2
    exit 2
fi

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
    # Build the package, not a single file: cmd/quarrel/main.go is gated
    # `linux && amd64 && cuda && cgo`, so `go run ./cmd/quarrel/main.go`
    # fails to build on macOS. Building the package lets Go pick the right
    # main for the host.
    QUARREL_OUT=$( go run -tags=darwin,metal ./cmd/quarrel -model "$MODEL" -n "$N_TOKENS" -prompt "$PROMPT" 2>&1 )
fi

# The binaries print:
#   === Complete ===
#   32 tokens in 1.23s (26.0 t/s)
#   <decoded text>
QUARREL_TPS=$( echo "$QUARREL_OUT" | grep -oE '\([0-9.]+ t/s\)' | tail -n 1 | tr -d '()' | awk '{print $1}' )
if [ -z "$QUARREL_TPS" ]; then
    QUARREL_TPS="N/A"
    echo "warning: could not parse throughput from the quarrel output:" >&2
    echo "$QUARREL_OUT" | tail -n 5 >&2
fi
QUARREL_TEXT=$( echo "$QUARREL_OUT" | awk '
    /^=== Complete ===$/ { found=1; next }
    found && /tokens in .*t\/s/ { next }   # the throughput line
    found && NF { print; exit }
' )

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
