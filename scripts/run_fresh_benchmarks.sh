#!/bin/bash
# run_fresh_benchmarks.sh

# Use absolute paths for tools to avoid PATH issues
JQ="/usr/bin/jq"
GREP="/usr/bin/grep"
TAIL="/usr/bin/tail"
AWK="/usr/bin/awk"
BC="/usr/bin/bc"

MODELS=(
    "gpt-oss:latest|/Users/rsd/.ollama/models/blobs/sha256-e7b273f9636059a689e3ddcab3716e4f65abe0143ac978e46673ad0e52d09efb"
    "nemotron-3-nano:latest|/Users/rsd/.ollama/models/blobs/sha256-a70437c41b3b0b768c48737e15f8160c90f13dc963f5226aabb3a160f708d1ce"
    "nemotron-mini:4b|/Users/rsd/.ollama/models/blobs/sha256-1dcbd925825b41744ddc2fc3047db6d3ad0aecf8d336f4fadc044eaaf79779d5"
    "mistral:latest|/Users/rsd/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f"
)

BIN="./bin/metal_benchmark"
LLAMA_BENCH="/opt/homebrew/bin/llama-bench"

echo "Implementation | Model | TPS (Metal) | TPS (llama.bench) | Ratio | Coherence"
echo "---|---|---|---|---|---"

for entry in "${MODELS[@]}"; do
    NAME="${entry%%|*}"
    PATH_MODEL="${entry##*|}"
    
    # 1. Run Quarrel Benchmark
    Q_OUT=$( "$BIN" -model "$PATH_MODEL" -tokens 32 -output json 2>/dev/null )
    Q_TPS=$( echo "$Q_OUT" | "$JQ" -r '.throughput_tokens_per_sec' )
    
    # 2. Run Llama Bench
    L_OUT=$( "$LLAMA_BENCH" -m "$PATH_MODEL" -p 32 -n 32 -r 1 2>&1 )
    L_TPS=$( echo "$L_OUT" | "$GREP" "tg" | "$TAIL" -n 1 | "$AWK" '{for(i=1;i<=NF;i++) if($i ~ /^[0-9]+\.[0-9]+$/) {print $i; exit}}' )
    
    # 3. Calculate Ratio
    if [ -n "$Q_TPS" ] && [ -n "$L_TPS" ] && [ "$L_TPS" != "0.00" ]; then
        RATIO=$( echo "scale=2; $Q_TPS / $L_TPS" | "$BC" )
    else
        RATIO="N/A"
    fi
    
    # 4. Simple Coherence Check
    TEXT=$( echo "$Q_OUT" | "$JQ" -r '.output' )
    if [ -n "$TEXT" ] && [ "$TEXT" != "null" ] && [ ${#TEXT} -gt 10 ]; then
        COHERENCE="PASSED"
    else
        COHERENCE="FAILED"
    fi
    
    echo "Quarrel | $NAME | $Q_TPS | $L_TPS | ${RATIO}x | $COHERENCE"
done
