#!/usr/bin/env bash
#
# benchmark_engines.sh — Cross-engine inference benchmark suite
#
# Compares longbow-quarrel against llama.cpp (llama-bench) and ollama
# for prompt processing (pp) and text generation (tg) throughput.
#
# Usage:
#   ./scripts/benchmark_engines.sh [OPTIONS]
#
# Options:
#   -m, --model PATH      GGUF model path (required)
#   -n, --tokens N        Tokens to generate (default: 128)
#   -r, --runs N          Number of benchmark runs (default: 5)
#   -w, --warmup N        Warmup runs (default: 2)
#   -p, --prompt TEXT     Prompt string (default: "The capital of France is")
#   -o, --output DIR      Output directory (default: benchmark_results)
#   -e, --engines LIST    Comma-separated engines: quarrel,llama,ollama (default: all available)
#   -ngl, --gpu-layers N  GPU layers to offload (-1=all, default: -1)
#   --json                Output raw JSON results
#   --csv                 Append to CSV summary file
#   -h, --help            Show this help
#
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────
MODEL=""
N_TOKENS=128
RUNS=5
WARMUP=2
PROMPT="The capital of France is"
OUTPUT_DIR="benchmark_results"
ENGINES=""
GPU_LAYERS=-1
JSON_OUTPUT=false
CSV_APPEND=false
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# ── Paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
QUARREL_BIN="$REPO_ROOT/bin/quarrel-cuda"
QUARREL_SIMPLE="$REPO_ROOT/bin/quarrel-simple"
LLAMA_BENCH="$(which llama-bench 2>/dev/null || true)"
OLLAMA_BIN="$(which ollama 2>/dev/null || true)"

# ── Colors ───────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ── Parse args ───────────────────────────────────────────────────────────
usage() {
    sed -n '3,/^$/{ s/^# \?//; p }' "$0"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)      MODEL="$2"; shift 2 ;;
        -n|--tokens)     N_TOKENS="$2"; shift 2 ;;
        -r|--runs)       RUNS="$2"; shift 2 ;;
        -w|--warmup)     WARMUP="$2"; shift 2 ;;
        -p|--prompt)     PROMPT="$2"; shift 2 ;;
        -o|--output)     OUTPUT_DIR="$2"; shift 2 ;;
        -e|--engines)    ENGINES="$2"; shift 2 ;;
        -ngl|--gpu-layers) GPU_LAYERS="$2"; shift 2 ;;
        --json)          JSON_OUTPUT=true; shift ;;
        --csv)           CSV_APPEND=true; shift ;;
        -h|--help)       usage ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo -e "${RED}Error: --model is required${NC}"
    usage
fi

if [[ ! -f "$MODEL" ]]; then
    echo -e "${RED}Error: Model file not found: $MODEL${NC}"
    exit 1
fi

# ── Auto-detect available engines ────────────────────────────────────────
detect_engines() {
    local available=""
    if [[ -x "$QUARREL_BIN" ]]; then
        available="${available}quarrel,"
    fi
    if [[ -n "$LLAMA_BENCH" && -x "$LLAMA_BENCH" ]]; then
        available="${available}llama,"
    fi
    if [[ -n "$OLLAMA_BIN" ]]; then
        available="${available}ollama,"
    fi
    echo "${available%,}"
}

if [[ -z "$ENGINES" ]]; then
    ENGINES=$(detect_engines)
fi

if [[ -z "$ENGINES" ]]; then
    echo -e "${RED}Error: No benchmark engines available${NC}"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ── Helpers ──────────────────────────────────────────────────────────────
RESULTS_JSON="$OUTPUT_DIR/results_${TIMESTAMP}.json"
CSV_FILE="$OUTPUT_DIR/summary.csv"

log() { echo -e "${CYAN}[$(date +%H:%M:%S)]${NC} $*"; }
ok()  { echo -e "${GREEN}✓${NC} $*"; }
err() { echo -e "${RED}✗${NC} $*"; }

# Parse quarrel output for tokens/sec
# quarrel outputs lines like: "Inference complete: 128 tokens in 3.45s (37.10 t/s)"
parse_quarrel_tps() {
    local output="$1"
    echo "$output" | grep -oP '\([\d.]+\s*t/s\)' | head -1 | tr -d '() t/s' || echo "0"
}

# Parse quarrel output for text generation time
parse_quarrel_gen_time() {
    local output="$1"
    echo "$output" | grep -oP '[\d.]+s\)' | head -1 | tr -d 's)' || echo "0"
}

# Calculate mean and stddev from array of numbers
calc_stats() {
    local -n arr=$1
    local n=${#arr[@]}
    if [[ $n -eq 0 ]]; then
        echo "0 0 0"
        return
    fi
    local sum=0
    for v in "${arr[@]}"; do
        sum=$(echo "$sum + $v" | bc -l)
    done
    local mean=$(echo "scale=2; $sum / $n" | bc -l)
    local variance=0
    for v in "${arr[@]}"; do
        variance=$(echo "$variance + ($v - $mean)^2" | bc -l)
    done
    if [[ $n -gt 1 ]]; then
        variance=$(echo "scale=2; $variance / ($n - 1)" | bc -l)
    fi
    local stddev=$(echo "scale=2; sqrt($variance)" | bc -l)
    local min=$(printf '%s\n' "${arr[@]}" | sort -n | head -1)
    local max=$(printf '%s\n' "${arr[@]}" | sort -n | tail -1)
    echo "$mean $stddev $min $max"
}

# ── Benchmark: quarrel ──────────────────────────────────────────────────
bench_quarrel() {
    log "Benchmarking longbow-quarrel..."
    local tps_values=()
    local pp_tps_values=()
    local output_text=""

    local bin="$QUARREL_BIN"
    if [[ ! -x "$bin" ]]; then
        bin="$QUARREL_SIMPLE"
    fi
    if [[ ! -x "$bin" ]]; then
        err "quarrel binary not found"
        return
    fi

    # Warmup
    for ((i = 0; i < WARMUP; i++)); do
        "$bin" -model "$MODEL" -n "$N_TOKENS" -prompt "$PROMPT" -ngl "$GPU_LAYERS" > /dev/null 2>&1 || true
    done

    # Benchmark
    for ((i = 0; i < RUNS; i++)); do
        local start_ns=$(date +%s%N)
        local out
        out=$("$bin" -model "$MODEL" -n "$N_TOKENS" -prompt "$PROMPT" -ngl "$GPU_LAYERS" 2>&1) || true
        local end_ns=$(date +%s%N)

        local elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))
        local tps
        tps=$(parse_quarrel_tps "$out")

        if [[ -z "$tps" || "$tps" == "0" ]]; then
            # Fallback: compute from elapsed time
            tps=$(echo "scale=2; $N_TOKENS / ($elapsed_ms / 1000.0)" | bc -l 2>/dev/null || echo "0")
        fi

        tps_values+=("$tps")
        output_text="$out"
    done

    local stats
    stats=$(calc_stats tps_values)
    local mean stddev min max
    read -r mean stddev min max <<< "$stats"

    echo "quarrel|$mean|$stddev|$min|$max|$RUNS|$N_TOKENS"

    if [[ "$JSON_OUTPUT" == "true" ]]; then
        cat <<EOF
{"engine":"quarrel","model":"$(basename "$MODEL")","tps_mean":$mean,"tps_stddev":$stddev,"tps_min":$min,"tps_max":$max,"runs":$RUNS,"tokens":$N_TOKENS}
EOF
    fi
}

# ── Benchmark: llama.cpp ────────────────────────────────────────────────
bench_llama() {
    if [[ -z "$LLAMA_BENCH" || ! -x "$LLAMA_BENCH" ]]; then
        err "llama-bench not found, skipping"
        return
    fi

    log "Benchmarking llama.cpp (llama-bench)..."
    local tps_values=()

    # llama-bench outputs: pp N ... pp_tps ... tg N ... tg_tps
    local out
    out=$("$LLAMA_BENCH" -m "$MODEL" -n "$N_TOKENS" -p "$N_TOKENS" -r "$RUNS" \
        -ngl "$GPU_LAYERS" -o json 2>/dev/null || true)

    if [[ -n "$out" ]]; then
        # Extract tg (text generation) tokens/sec from JSON output
        # llama-bench JSON: {"model":"...","benchmarks":[{"parameters":{"n":N},"results":[{"avg_ts":X}]}]}
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
            echo "llama|$tg_tps|0|$tg_tps|$tg_tps|$RUNS|$N_TOKENS"
            return
        fi
    fi

    # Fallback: parse text output
    out=$("$LLAMA_BENCH" -m "$MODEL" -n "$N_TOKENS" -p "$N_TOKENS" -r "$RUNS" \
        -ngl "$GPU_LAYERS" 2>/dev/null || true)
    local tg_tps
    tg_tps=$(echo "$out" | grep -E "tg[0-9]" | awk '{for(i=1;i<=NF;i++) if($i ~ /^[0-9]+\.[0-9]+$/) {print $i; exit}}' | tail -1)

    if [[ -z "$tg_tps" ]]; then
        tg_tps="0"
    fi

    echo "llama|$tg_tps|0|$tg_tps|$tg_tps|$RUNS|$N_TOKENS"
}

# ── Benchmark: ollama ───────────────────────────────────────────────────
bench_ollama() {
    if [[ -z "$OLLAMA_BIN" || ! -x "$OLLAMA_BIN" ]]; then
        err "ollama not found, skipping"
        return
    fi

    # Need to resolve ollama model name from GGUF path
    local model_name=""
    # Check if model is already an ollama model reference
    if [[ "$MODEL" == *"/"* && "$MODEL" != *"/"*"/"* ]]; then
        model_name="$MODEL"
    fi

    log "Benchmarking ollama..."
    local tps_values=()

    # Warmup
    for ((i = 0; i < WARMUP; i++)); do
        echo "$PROMPT" | timeout 30 "$OLLAMA_BIN" run "$model_name" --verbose > /dev/null 2>&1 || true
    done

    # Benchmark
    for ((i = 0; i < RUNS; i++)); do
        local out
        out=$(echo "$PROMPT" | timeout 60 "$OLLAMA_BIN" run "$model_name" --verbose 2>&1) || true

        # ollama --verbose outputs: "eval rate: XX.XX tokens/s"
        local tps
        tps=$(echo "$out" | grep -oP 'eval rate:\s*[\d.]+ tokens/s' | grep -oP '[\d.]+' | tail -1 || true)

        if [[ -n "$tps" ]]; then
            tps_values+=("$tps")
        fi
    done

    if [[ ${#tps_values[@]} -eq 0 ]]; then
        err "No ollama results captured"
        return
    fi

    local stats
    stats=$(calc_stats tps_values)
    local mean stddev min max
    read -r mean stddev min max <<< "$stats"

    echo "ollama|$mean|$stddev|$min|$max|$RUNS|$N_TOKENS"
}

# ── Run all benchmarks ──────────────────────────────────────────────────
main() {
    echo ""
    echo -e "${BOLD}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║     Longbow-Quarrel Cross-Engine Benchmark Suite       ║${NC}"
    echo -e "${BOLD}╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "  Model:      ${CYAN}$(basename "$MODEL")${NC}"
    echo -e "  Prompt:     ${CYAN}\"${PROMPT:0:50}...\"${NC}"
    echo -e "  Tokens:     ${CYAN}$N_TOKENS${NC}"
    echo -e "  Runs:       ${CYAN}$RUNS${NC} (warmup: $WARMUP)"
    echo -e "  GPU layers: ${CYAN}$GPU_LAYERS${NC}"
    echo -e "  Engines:    ${CYAN}$ENGINES${NC}"
    echo -e "  Output:     ${CYAN}$OUTPUT_DIR${NC}"
    echo ""

    # CSV header
    if [[ "$CSV_APPEND" == "true" ]]; then
        if [[ ! -f "$CSV_FILE" ]]; then
            echo "timestamp,model,engine,tokens,tps_mean,tps_stddev,tps_min,tps_max,runs" > "$CSV_FILE"
        fi
    fi

    # Collect results
    declare -A ENGINE_RESULTS

    IFS=',' read -ra ENGINE_LIST <<< "$ENGINES"
    for eng in "${ENGINE_LIST[@]}"; do
        eng=$(echo "$eng" | tr -d ' ')
        local result=""
        case "$eng" in
            quarrel) result=$(bench_quarrel) ;;
            llama)   result=$(bench_llama) ;;
            ollama)  result=$(bench_ollama) ;;
            *)       err "Unknown engine: $eng"; continue ;;
        esac

        if [[ -n "$result" ]]; then
            ENGINE_RESULTS["$eng"]="$result"
        fi
    done

    # ── Print comparison table ───────────────────────────────────────────
    echo ""
    echo -e "${BOLD}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║                    Results                              ║${NC}"
    echo -e "${BOLD}╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    printf "%-12s │ %10s │ %10s │ %10s │ %10s │ %6s\n" "Engine" "Mean t/s" "StdDev" "Min" "Max" "Runs"
    printf "%-12s─┼─%10s─┼─%10s─┼─%10s─┼─%10s─┼─%6s\n" "────────────" "──────────" "──────────" "──────────" "──────────" "──────"

    local quarrel_tps=""
    for eng in "${ENGINE_LIST[@]}"; do
        eng=$(echo "$eng" | tr -d ' ')
        local data="${ENGINE_RESULTS[$eng]:-}"
        if [[ -z "$data" ]]; then
            continue
        fi
        IFS='|' read -r name mean stddev minmax_min maxmin_max runs tokens <<< "$data"

        local color="$NC"
        case "$eng" in
            quarrel) color="$GREEN" ;;
            llama)   color="$YELLOW" ;;
            ollama)  color="$CYAN" ;;
        esac

        printf "${color}%-12s${NC} │ %10s │ %10s │ %10s │ %10s │ %6s\n" \
            "$eng" "$mean" "$stddev" "$minmax_min" "$maxmin_max" "$runs"

        if [[ "$eng" == "quarrel" ]]; then
            quarrel_tps="$mean"
        fi

        # CSV append
        if [[ "$CSV_APPEND" == "true" ]]; then
            echo "$TIMESTAMP,$(basename "$MODEL"),$eng,$tokens,$mean,$stddev,$minmax_max,$maxmin_max,$runs" >> "$CSV_FILE"
        fi
    done

    echo ""

    # ── Performance comparison ──────────────────────────────────────────
    if [[ -n "$quarrel_tps" && "$quarrel_tps" != "0" ]]; then
        for eng in "${ENGINE_LIST[@]}"; do
            eng=$(echo "$eng" | tr -d ' ')
            if [[ "$eng" == "quarrel" ]]; then continue; fi
            local data="${ENGINE_RESULTS[$eng]:-}"
            if [[ -z "$data" ]]; then continue; fi
            IFS='|' read -r name ref_tps _ _ _ _ _ <<< "$data"
            if [[ -n "$ref_tps" && "$ref_tps" != "0" ]]; then
                local ratio
                ratio=$(echo "scale=1; $quarrel_tps / $ref_tps * 100" | bc -l 2>/dev/null || echo "?")
                echo -e "  quarrel vs $eng: ${BOLD}${ratio}%${NC} of $eng throughput"
            fi
        done
    fi

    echo ""

    # ── JSON output ─────────────────────────────────────────────────────
    if [[ "$JSON_OUTPUT" == "true" ]]; then
        echo "{"
        echo "  \"timestamp\": \"$TIMESTAMP\","
        echo "  \"model\": \"$(basename "$MODEL")\","
        echo "  \"prompt\": \"${PROMPT}\","
        echo "  \"tokens\": $N_TOKENS,"
        echo "  \"runs\": $RUNS,"
        echo "  \"engines\": {"
        local first=true
        for eng in "${ENGINE_LIST[@]}"; do
            eng=$(echo "$eng" | tr -d ' ')
            local data="${ENGINE_RESULTS[$eng]:-}"
            if [[ -z "$data" ]]; then continue; fi
            IFS='|' read -r name mean stddev minmax_min maxmin_max runs tokens <<< "$data"
            [[ "$first" == "true" ]] && first=false || echo ","
            echo -n "    \"$eng\": {\"tps_mean\": $mean, \"tps_stddev\": $stddev, \"tps_min\": $minmax_min, \"tps_max\": $maxmin_max}"
        done
        echo ""
        echo "  }"
        echo "}"
    fi

    log "Results saved to: $OUTPUT_DIR/"
    [[ "$CSV_APPEND" == "true" ]] && log "CSV appended to: $CSV_FILE"
}

main "$@"
