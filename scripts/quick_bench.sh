#!/usr/bin/env bash
#
# quick_bench.sh — Quick single-model benchmark comparison
#
# One-liner to compare quarrel vs llama.cpp on a single model.
#
# Usage:
#   ./scripts/quick_bench.sh <model.gguf> [prompt] [tokens]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL="${1:?Usage: $0 <model.gguf> [prompt] [tokens]}"
PROMPT="${2:-The capital of France is}"
N_TOKENS="${3:-64}"
RUNS=3

QUARREL_BIN="$REPO_ROOT/bin/quarrel-cuda"
[[ -x "$QUARREL_BIN" ]] || QUARREL_BIN="$REPO_ROOT/bin/quarrel-simple"

LLAMA_BENCH="$(which llama-bench 2>/dev/null || true)"

echo "╔═══════════════════════════════════════════════════════╗"
echo "║           Quick Benchmark Comparison                 ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo "  Model:  $(basename "$MODEL")"
echo "  Prompt: \"$PROMPT\""
echo "  Tokens: $N_TOKENS"
echo "  Runs:   $RUNS"
echo ""

# ── Quarrel ───────────────────────────────────────────────────────────────
echo "━━━ longbow-quarrel ━━━"
Q_TPS=""
if [[ -x "$QUARREL_BIN" ]]; then
    SUM=0
    COUNT=0
    for ((i = 0; i < RUNS; i++)); do
        OUT=$("$QUARREL_BIN" -model "$MODEL" -n "$N_TOKENS" -prompt "$PROMPT" 2>&1 || true)
        TPS=$(echo "$OUT" | grep -oP '\([\d.]+\s*t/s\)' | head -1 | tr -d '() t/s' || true)
        if [[ -n "$TPS" && "$TPS" != "0" ]]; then
            echo "  Run $((i+1)): $TPS t/s"
            SUM=$(echo "$SUM + $TPS" | bc -l)
            COUNT=$((COUNT + 1))
        fi
    done
    if [[ $COUNT -gt 0 ]]; then
        Q_TPS=$(echo "scale=2; $SUM / $COUNT" | bc -l)
        echo "  Average: $Q_TPS t/s"
    fi
else
    echo "  (binary not found)"
fi
echo ""

# ── llama.cpp ─────────────────────────────────────────────────────────────
echo "━━━ llama.cpp ━━━"
L_TPS=""
if [[ -n "$LLAMA_BENCH" && -x "$LLAMA_BENCH" ]]; then
    OUT=$("$LLAMA_BENCH" -m "$MODEL" -n "$N_TOKENS" -p "$N_TOKENS" -r "$RUNS" 2>/dev/null || true)
    L_TPS=$(echo "$OUT" | grep -E "tg[0-9]" | awk '{for(i=1;i<=NF;i++) if($i ~ /^[0-9]+\.[0-9]+$/) {print $i; exit}}' | tail -1 || true)
    if [[ -n "$L_TPS" ]]; then
        echo "  Average: $L_TPS t/s"
    else
        echo "  (no result)"
    fi
else
    echo "  (llama-bench not found)"
fi
echo ""

# ── Summary ───────────────────────────────────────────────────────────────
echo "━━━ Comparison ━━━"
if [[ -n "$Q_TPS" && -n "$L_TPS" && "$L_TPS" != "0" ]]; then
    RATIO=$(echo "scale=1; $Q_TPS / $L_TPS * 100" | bc -l 2>/dev/null || echo "?")
    echo "  quarrel:   $Q_TPS t/s"
    echo "  llama.cpp: $L_TPS t/s"
    echo "  Ratio:     ${RATIO}% of llama.cpp"
elif [[ -n "$Q_TPS" ]]; then
    echo "  quarrel: $Q_TPS t/s"
elif [[ -n "$L_TPS" ]]; then
    echo "  llama.cpp: $L_TPS t/s"
fi
echo ""
