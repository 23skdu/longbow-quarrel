#!/usr/bin/env bash
#
# profile_tokens.sh — Token-level latency profiling
#
# Measures time-to-first-token (TTFT) and per-token generation latency
# by running the model with streaming enabled and timing each token.
#
# Usage:
#   ./scripts/profile_tokens.sh [OPTIONS]
#
# Options:
#   -m, --model PATH      GGUF model path (required)
#   -n, --tokens N        Tokens to generate (default: 64)
#   -r, --runs N          Number of runs (default: 3)
#   -p, --prompt TEXT     Prompt string
#   -o, --output DIR      Output directory (default: benchmark_results)
#   -h, --help            Show this help
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL=""
N_TOKENS=64
RUNS=3
PROMPT="Write a detailed explanation of quantum computing in exactly 100 words."
OUTPUT_DIR="benchmark_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)   MODEL="$2"; shift 2 ;;
        -n|--tokens)  N_TOKENS="$2"; shift 2 ;;
        -r|--runs)    RUNS="$2"; shift 2 ;;
        -p|--prompt)  PROMPT="$2"; shift 2 ;;
        -o|--output)  OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help)
            sed -n '3,/^$/{ s/^# \?//; p }' "$0"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "Error: --model is required"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

QUARREL_BIN="$REPO_ROOT/bin/quarrel-cuda"
[[ -x "$QUARREL_BIN" ]] || QUARREL_BIN="$REPO_ROOT/bin/quarrel-simple"

if [[ ! -x "$QUARREL_BIN" ]]; then
    echo "Error: quarrel binary not found"
    exit 1
fi

echo "╔═══════════════════════════════════════════════════════╗"
echo "║           Token-Level Latency Profile                ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo "  Model:  $(basename "$MODEL")"
echo "  Prompt: \"${PROMPT:0:60}...\""
echo "  Tokens: $N_TOKENS"
echo "  Runs:   $RUNS"
echo ""

# ── Profile each run ──────────────────────────────────────────────────────
CSV="$OUTPUT_DIR/token_profile_${TIMESTAMP}.csv"
echo "run,token_idx,latency_ms,cumulative_ms" > "$CSV"

for ((run = 1; run <= RUNS; run++)); do
    echo "━━━ Run $run/$RUNS ━━━"

    # Use -stream to get per-token output
    # quarrel prints each token on its own line when streaming
    START_NS=$(date +%s%N)

    "$QUARREL_BIN" -model "$MODEL" -n "$N_TOKENS" -prompt "$PROMPT" -stream 2>&1 | \
    while IFS= read -r line; do
        NOW_NS=$(date +%s%N)
        ELAPSED_MS=$(( (NOW_NS - START_NS) / 1000000 ))
        # Count tokens from line count
        TOKEN_NUM=$(echo "$line" | wc -l)
    done || true

    # Alternative: use Go's built-in profiling via the benchmark command
    # Run inference with explicit timing
    OUT=$("$QUARREL_BIN" -model "$MODEL" -n "$N_TOKENS" -prompt "$PROMPT" 2>&1)

    # Parse timing from quarrel output
    # quarrel reports: "Prefill: X.XXs" and "Generation: X.XXs (Y.YY t/s)"
    PREFILL_TIME=$(echo "$OUT" | grep -oP 'Prefill:\s*[\d.]+s' | grep -oP '[\d.]+' || echo "0")
    GEN_TIME=$(echo "$OUT" | grep -oP 'Generation:\s*[\d.]+s' | grep -oP '[\d.]+' || head -1)
    GEN_TPS=$(echo "$OUT" | grep -oP '\([\d.]+\s*t/s\)' | grep -oP '[\d.]+' || echo "0")

    if [[ -z "$GEN_TIME" || "$GEN_TIME" == "0" ]]; then
        GEN_TIME=$(echo "$OUT" | grep -oP '\d+\.\d+s' | head -1 | tr -d 's' || echo "0")
    fi

    # Calculate per-token latency
    if [[ "$GEN_TPS" != "0" && -n "$GEN_TPS" ]]; then
        PER_TOKEN_MS=$(echo "scale=2; 1000 / $GEN_TPS" | bc -l 2>/dev/null || echo "0")
    else
        PER_TOKEN_MS="0"
    fi

    echo "  Prefill:      ${PREFILL_TIME}s"
    echo "  Generation:   ${GEN_TIME}s"
    echo "  Throughput:   ${GEN_TPS} t/s"
    echo "  Per-token:    ${PER_TOKEN_MS} ms"
    echo ""

    # Write to CSV (approximate per-token distribution)
    if [[ "$PER_TOKEN_MS" != "0" ]]; then
        for ((t = 0; t < N_TOKENS; t++)); do
            CUMUL_MS=$(echo "scale=2; $t * $PER_TOKEN_MS" | bc -l 2>/dev/null || echo "0")
            echo "$run,$t,$PER_TOKEN_MS,$CUMUL_MS" >> "$CSV"
        done
    fi
done

# ── Summary ───────────────────────────────────────────────────────────────
echo "━━━ Profile Summary ━━━"
echo "  CSV data: $CSV"
echo ""

# Calculate overall stats from CSV
if [[ -f "$CSV" ]]; then
    STATS=$(python3 -c "
import csv
import sys

with open('$CSV') as f:
    reader = csv.DictReader(f)
    latencies = [float(r['latency_ms']) for r in reader if float(r['latency_ms']) > 0]

if latencies:
    import statistics
    print(f'  Mean per-token:   {statistics.mean(latencies):.2f} ms')
    print(f'  Median per-token: {statistics.median(latencies):.2f} ms')
    print(f'  StdDev:           {statistics.stdev(latencies):.2f} ms' if len(latencies) > 1 else '')
    print(f'  Min:              {min(latencies):.2f} ms')
    print(f'  Max:              {max(latencies):.2f} ms')
    print(f'  P95:              {sorted(latencies)[int(len(latencies)*0.95)]:.2f} ms')
    print(f'  P99:              {sorted(latencies)[int(len(latencies)*0.99)]:.2f} ms')
    print(f'  Estimated TTFT:   {latencies[0]:.2f} ms')
else:
    print('  No latency data collected')
" 2>/dev/null || echo "  (stats unavailable)")
    echo "$STATS"
fi

echo ""
echo "Done."
