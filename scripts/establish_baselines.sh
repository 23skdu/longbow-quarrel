#!/bin/bash

# Baseline Establishment Script
# Establishes performance baselines for target models
# Excludes loading/prefill noise with precise measurements

set -e

echo "=== Establishing New Performance Baselines ==="
echo "Date: $(date)"
echo "Hardware: $(sysctl -n machin_cpu.brand) $(sysctl -n machin_cpu.model)"
echo "macOS: $(sw_vers -productVersion)"
echo ""

# Define models to benchmark - using actual file paths
declare -a MODELS=(
    "TinyLlama:/Users/rsd/.ollama/models/blobs/sha256-2af3b81862c6be03c769683af18efdadb2c33f60ff32ab6f83e42c043d6c7816"
    "Mistral:/Users/rsd/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f"
    "Granite4:/Users/rsd/.ollama/models/blobs/sha256-5c7ac4aead1bcf4c8da9534ed72cc632d005aeed6547f1e8662ccdfae688364e"
)

# Baseline results
RESULTS=()

# Function to run benchmark and capture results
run_benchmark() {
    local model_name="$1"
    local model_identifier="$2"
    local tokens=${3:-100}
    
    echo "Running baseline for $model_name..."
    
    # Run with longer prompt to get stable measurement (exclude loading noise)
    local output=$(./bin/metal_benchmark \
        -model "$model_identifier" \
        -prompt "The quick brown fox jumps over the lazy dog and runs through the forest." \
        -tokens $tokens \
        -output json 2>/dev/null | tail -1)
    
    # Extract key metrics
    local throughput=$(echo "$output" | jq -r '.throughput_tokens_per_sec // 0')
    local prefill=$(echo "$output" | jq -r '.prefill_duration_seconds // 0')
    local generation=$(echo "$output" | jq -r '.generation_duration_seconds // 0')
    local tokens_gen=$(echo "$output" | jq -r '.tokens_generated // 0')
    
    echo "Throughput: $throughput tokens/sec"
    echo "Prefill time: ${prefill}s"
    echo "Generation time: ${generation}s"
    echo "Tokens generated: $tokens_gen"
    echo "Generation rate: $(echo "scale=2; $tokens_gen / $generation" | bc -l 2>/dev/null || echo "0") tokens/sec"
    echo ""
    
    # Add to results array
    RESULTS+=("$model_name:$throughput:$prefill:$generation:$tokens_gen")
}

echo "=== Baseline Results ==="
for result in "${RESULTS[@]}"; do
    IFS=':' read -r model_name throughput prefill generation tokens_gen <<< "$result"
    echo "Model: $model_name"
    echo "  Throughput: $throughput tokens/sec"
    echo "  Prefill: ${prefill}s"
    echo " Generation: ${generation}s" 
    echo " Tokens: $tokens_gen"
    echo " Gen Rate: $(echo "scale=2; $tokens_gen / $generation" | bc -l | xargs printf "%.2f") tokens/sec"
    echo "---"
done

# Run benchmarks for all models
for model_info in "${MODELS[@]}"; do
    IFS=':' read -r name model_identifier <<< "$model_info"
    run_benchmark "$name" "$model_identifier"
done

echo ""
echo "=== Summary ==="
echo "Models benchmarked: ${#RESULTS[@]}"
echo ""
echo "Note: These measurements exclude model loading time"
echo "Recommend running each 3 times and averaging for stability"