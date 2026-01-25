#!/bin/bash

# Simple benchmark comparison script
set -e

MODEL_PATH="/Users/rsd/.ollama/models/blobs/sha256-f535f83ec568d040f88ddc04a199fa6da90923bbb41d4dcaed02caa924d6ef57"
PROMPT="The quick brown fox jumps over the lazy dog"
TOKENS=50
ITERATIONS=3

echo "========================================"
echo "   Simple LLM Performance Benchmark   "
echo "========================================"
echo "Model: SmolLM2 135M"
echo "Tokens: $TOKENS"
echo "Iterations: $ITERATIONS"
echo

run_metal_benchmark() {
    echo "Running longbow-quarrel (Metal)..."
    local total_time=0
    
    for i in $(seq 1 $ITERATIONS); do
        echo -n "  Run $i/$ITERATIONS: "
        local output=$(./bin/metal_benchmark --model "$MODEL_PATH" --prompt "$PROMPT" --tokens $TOKENS --output json 2>/dev/null | tail -1)
        local throughput=$(echo "$output" | jq -r '.throughput_tokens_per_sec')
        echo "$throughput tokens/sec"
        total_time=$(echo "$total_time + 1/$throughput" | bc -l)
    done
    
    local avg_throughput=$(echo "scale=2; $ITERATIONS / $total_time" | bc -l)
    echo "  Average: $avg_throughput tokens/sec"
    echo
    echo "$avg_throughput"
}

run_llamacpp_benchmark() {
    echo "Running llama.cpp..."
    local total_time=0
    
    for i in $(seq 1 $ITERATIONS); do
        echo -n "  Run $i/$ITERATIONS: "
        local start=$(date +%s.%N)
        /opt/homebrew/bin/llama-cli -m "$MODEL_PATH" -p "$PROMPT" -n $TOKENS --temp 0.0 -c 2048 >/dev/null 2>&1
        local end=$(date +%s.%N)
        local duration=$(echo "$end - $start" | bc -l)
        local throughput=$(echo "scale=2; $TOKENS / $duration" | bc -l)
        echo "$throughput tokens/sec"
        total_time=$(echo "$total_time + $duration" | bc -l)
    done
    
    local avg_throughput=$(echo "scale=2; $TOKENS * $ITERATIONS / $total_time" | bc -l)
    echo "  Average: $avg_throughput tokens/sec"
    echo
    echo "$avg_throughput"
}

# Run benchmarks if tools exist
if [[ -f "./bin/metal_benchmark" ]]; then
    METAL_RESULT=$(run_metal_benchmark)
else
    echo "Metal benchmark not found. Build with: go build -tags \"darwin,metal\" -o bin/metal_benchmark ./cmd/metal_benchmark"
    METAL_RESULT="0"
fi

if command -v /opt/homebrew/bin/llama-cli >/dev/null 2>&1; then
    LLAMACPP_RESULT=$(run_llamacpp_benchmark)
else
    echo "llama.cpp not found. Install with: brew install llama.cpp"
    LLAMACPP_RESULT="0"
fi

# Generate report
echo "========================================"
echo "           RESULTS                    "
echo "========================================"
echo "longbow-quarrel (Metal): $METAL_RESULT tokens/sec"
echo "llama.cpp:              $LLAMACPP_RESULT tokens/sec"
echo

if [[ "$LLAMACPP_RESULT" != "0" && "$METAL_RESULT" != "0" ]]; then
    local speedup=$(echo "scale=2; $METAL_RESULT / $LLAMACPP_RESULT" | bc -l)
    echo "Metal GPU speedup: ${speedup}x"
else
    echo "Could not calculate speedup (missing results)"
fi

echo "========================================"