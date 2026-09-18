#!/usr/bin/env bash
#
# benchmark_models.sh — Multi-model benchmark across engines
#
# Runs the same prompt against multiple GGUF models, comparing quarrel
# and llama.cpp throughput for each.
#
# Usage:
#   ./scripts/benchmark_models.sh [OPTIONS]
#
# Options:
#   -l, --list FILE       File with model entries (one per line: "name:/path/to/model.gguf")
#   -n, --tokens N        Tokens to generate (default: 128)
#   -r, --runs N          Runs per model (default: 3)
#   -p, --prompt TEXT     Prompt string (default: "The capital of France is")
#   -o, --output DIR      Output directory (default: benchmark_results)
#   -e, --engines LIST    Comma-separated engines (default: quarrel,llama)
#   -ngl, --gpu-layers N  GPU layers to offload (default: -1)
#   -h, --help            Show this help
#
# Model list file format (one per line):
#   ModelName:/absolute/path/to/model.gguf
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults
MODEL_LIST=""
N_TOKENS=128
RUNS=3
PROMPT="The capital of France is"
OUTPUT_DIR="benchmark_results"
ENGINES="quarrel,llama"
GPU_LAYERS=-1
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        -l|--list)      MODEL_LIST="$2"; shift 2 ;;
        -n|--tokens)    N_TOKENS="$2"; shift 2 ;;
        -r|--runs)      RUNS="$2"; shift 2 ;;
        -p|--prompt)    PROMPT="$2"; shift 2 ;;
        -o|--output)    OUTPUT_DIR="$2"; shift 2 ;;
        -e|--engines)   ENGINES="$2"; shift 2 ;;
        -ngl|--gpu-layers) GPU_LAYERS="$2"; shift 2 ;;
        -h|--help)
            sed -n '3,/^$/{ s/^# \?//; p }' "$0"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Auto-discover models from ollama if no list provided
if [[ -z "$MODEL_LIST" ]]; then
    echo -e "${YELLOW}No --list provided. Discovering models from ollama...${NC}"
    # Create temp model list from ollama blobs
    MODEL_LIST=$(mktemp /tmp/models_XXXXXX.txt)
    ollama list 2>/dev/null | tail -n +2 | while read -r name id size _; do
        blob_path="$HOME/.ollama/models/blobs/sha256-${id}"
        if [[ -f "$blob_path" ]]; then
            echo "${name}:${blob_path}"
        fi
    done > "$MODEL_LIST" || true

    if [[ ! -s "$MODEL_LIST" ]]; then
        echo -e "${RED}No models found. Provide a model list with --list${NC}"
        exit 1
    fi
    echo -e "Found $(wc -l < "$MODEL_LIST") models"
fi

if [[ ! -f "$MODEL_LIST" ]]; then
    echo -e "${RED}Model list not found: $MODEL_LIST${NC}"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Output files
RESULTS_FILE="$OUTPUT_DIR/multi_model_${TIMESTAMP}.json"
CSV_FILE="$OUTPUT_DIR/multi_model_summary.csv"
REPORT_FILE="$OUTPUT_DIR/multi_model_${TIMESTAMP}.md"

# CSV header
echo "timestamp,model,engine,tokens,tps_mean,tps_stddev,tps_min,tps_max,runs" > "$CSV_FILE"

# Markdown report header
cat > "$REPORT_FILE" <<EOF
# Multi-Model Benchmark Report

**Date:** $(date -Iseconds)
**Prompt:** ${PROMPT}
**Tokens:** ${N_TOKENS}
**Runs per model:** ${RUNS}
**GPU layers:** ${GPU_LAYERS}
**Engines:** ${ENGINES}

EOF

echo "" > "$RESULTS_FILE"
echo "[" >> "$RESULTS_FILE"

# ── Parse quarrel tps ────────────────────────────────────────────────────
parse_quarrel_tps() {
    echo "$1" | grep -oP '\([\d.]+\s*t/s\)' | head -1 | tr -d '() t/s' || echo "0"
}

# ── Stats helper ──────────────────────────────────────────────────────────
calc_stats() {
    local -n arr=$1
    local n=${#arr[@]}
    if [[ $n -eq 0 ]]; then echo "0 0 0 0"; return; fi
    local sum=0
    for v in "${arr[@]}"; do sum=$(echo "$sum + $v" | bc -l); done
    local mean=$(echo "scale=2; $sum / $n" | bc -l)
    local variance=0
    for v in "${arr[@]}"; do variance=$(echo "$variance + ($v - $mean)^2" | bc -l); done
    if [[ $n -gt 1 ]]; then variance=$(echo "scale=2; $variance / ($n - 1)" | bc -l); fi
    local stddev=$(echo "scale=2; sqrt($variance)" | bc -l)
    local min=$(printf '%s\n' "${arr[@]}" | sort -n | head -1)
    local max=$(printf '%s\n' "${arr[@]}" | sort -n | tail -1)
    echo "$mean $stddev $min $max"
}

# ── Bench quarrel ─────────────────────────────────────────────────────────
bench_quarrel_model() {
    local model_path="$1"
    local bin="$REPO_ROOT/bin/quarrel-cuda"
    [[ -x "$bin" ]] || bin="$REPO_ROOT/bin/quarrel-simple"
    [[ -x "$bin" ]] || { echo "0 0 0 0"; return; }

    local tps_values=()
    # Warmup
    for ((i = 0; i < 2; i++)); do
        "$bin" -model "$model_path" -n "$N_TOKENS" -prompt "$PROMPT" -ngl "$GPU_LAYERS" > /dev/null 2>&1 || true
    done
    for ((i = 0; i < RUNS; i++)); do
        local out
        out=$("$bin" -model "$model_path" -n "$N_TOKENS" -prompt "$PROMPT" -ngl "$GPU_LAYERS" 2>&1) || true
        local tps
        tps=$(parse_quarrel_tps "$out")
        [[ -z "$tps" || "$tps" == "0" ]] && continue
        tps_values+=("$tps")
    done
    [[ ${#tps_values[@]} -eq 0 ]] && { echo "0 0 0 0"; return; }
    calc_stats tps_values
}

# ── Bench llama.cpp ───────────────────────────────────────────────────────
bench_llama_model() {
    local model_path="$1"
    local lb="$(which llama-bench 2>/dev/null || true)"
    [[ -x "$lb" ]] || { echo "0 0 0 0"; return; }

    local out
    out=$("$lb" -m "$model_path" -n "$N_TOKENS" -p "$N_TOKENS" -r "$RUNS" \
        -ngl "$GPU_LAYERS" -o json 2>/dev/null || true)

    if [[ -n "$out" ]]; then
        local tg_tps
        tg_tps=$(echo "$out" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for b in data.get('benchmarks', []):
        params = b.get('parameters', {})
        if params.get('pp', 0) == 0 and params.get('tg', 0) > 0:
            for r in b.get('results', []):
                print(f\"{r.get('avg_ts', 0):.2f}\")
                break
            break
except: pass
" 2>/dev/null || true)
        if [[ -n "$tg_tps" && "$tg_tps" != "0" ]]; then
            echo "$tg_tps 0 $tg_tps $tg_tps"
            return
        fi
    fi

    # Fallback: text parsing
    out=$("$lb" -m "$model_path" -n "$N_TOKENS" -p "$N_TOKENS" -r "$RUNS" \
        -ngl "$GPU_LAYERS" 2>/dev/null || true)
    local tg_tps
    tg_tps=$(echo "$out" | grep -E "tg[0-9]" | awk '{for(i=1;i<=NF;i++) if($i ~ /^[0-9]+\.[0-9]+$/) {print $i; exit}}' | tail -1)
    [[ -z "$tg_tps" ]] && tg_tps="0"
    echo "$tg_tps 0 $tg_tps $tg_tps"
}

# ── Main loop ─────────────────────────────────────────────────────────────
model_idx=0
total_models=$(wc -l < "$MODEL_LIST")

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║          Multi-Model Benchmark Suite                    ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

while IFS=: read -r model_name model_path || [[ -n "$model_name" ]]; do
    model_name=$(echo "$model_name" | tr -d '[:space:]')
    model_path=$(echo "$model_path" | tr -d '[:space:]')

    [[ -z "$model_name" || -z "$model_path" ]] && continue
    [[ ! -f "$model_path" ]] && { echo -e "${RED}  Skipping $model_name — file not found${NC}"; continue; }

    model_idx=$((model_idx + 1))
    echo -e "${BOLD}[$model_idx/$total_models] $model_name${NC}"
    echo -e "  Path: $model_path"
    echo ""

    # Table header
    printf "  %-12s │ %10s │ %10s │ %10s │ %10s\n" "Engine" "Mean t/s" "StdDev" "Min" "Max"
    printf "  %-12s─┼─%10s─┼─%10s─┼─%10s─┼─%10s\n" "────────────" "──────────" "──────────" "──────────" "──────────"

    first_engine_tps=""
    IFS=',' read -ra ENGINES_LIST <<< "$ENGINES"

    for eng in "${ENGINES_LIST[@]}"; do
        eng=$(echo "$eng" | tr -d ' ')
        local stats=""

        case "$eng" in
            quarrel) stats=$(bench_quarrel_model "$model_path") ;;
            llama)   stats=$(bench_llama_model "$model_path") ;;
            *) continue ;;
        esac

        read -r mean stddev min max <<< "$stats"
        [[ "$mean" == "0" ]] && continue

        local color="$NC"
        [[ "$eng" == "quarrel" ]] && color="$GREEN"
        [[ "$eng" == "llama" ]] && color="$YELLOW"

        printf "  ${color}%-12s${NC} │ %10s │ %10s │ %10s │ %10s\n" "$eng" "$mean" "$stddev" "$min" "$max"

        # CSV
        echo "$TIMESTAMP,$model_name,$eng,$N_TOKENS,$mean,$stddev,$min,$max,$RUNS" >> "$CSV_FILE"

        # JSON
        echo "    {\"engine\":\"$eng\",\"model\":\"$model_name\",\"tps_mean\":$mean,\"tps_stddev\":$stddev,\"tps_min\":$min,\"tps_max\":$max}," >> "$RESULTS_FILE"

        if [[ "$eng" == "quarrel" && "$first_engine_tps" == "" ]]; then
            first_engine_tps="$mean"
        fi
    done

    # Markdown table row
    echo "| $model_name | " >> "$REPORT_FILE"
    for eng in "${ENGINES_LIST[@]}"; do
        eng=$(echo "$eng" | tr -d ' ')
        printf "| $eng t/s |" >> "$REPORT_FILE"
    done
    echo "" >> "$REPORT_FILE"

    echo ""
done < "$MODEL_LIST"

# Close JSON
echo "]" >> "$RESULTS_FILE"

# Trailing comma cleanup
sed -i '$ s/,$//' "$RESULTS_FILE" 2>/dev/null || true

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║                    Summary                              ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  JSON results:  ${CYAN}$RESULTS_FILE${NC}"
echo -e "  CSV summary:   ${CYAN}$CSV_FILE${NC}"
echo -e "  Report:        ${CYAN}$REPORT_FILE${NC}"
echo ""
echo "Done."
