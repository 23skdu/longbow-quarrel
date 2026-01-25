#!/bin/bash

set -e

MODEL_PATH="$HOME/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f"
PROMPT="The capital of France is"
OUTPUT_DIR="performance_profiles"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0;0m'

echo -e "${BLUE}=== Performance Profiling Tool ===${NC}"
echo "Model: $MODEL_PATH"
echo "Prompt: \"$PROMPT\""
echo

mkdir -p "$OUTPUT_DIR"

CONFIGS=(
    "small:5:10:0.7"
    "medium:25:10:0.8" 
    "large:100:20:1.0"
    "fast:50:5:0.3"
)

echo -e "${YELLOW}Running performance tests...${NC}"

for config in "${CONFIGS[@]}"; do
    IFS=':' read -r -a config
    tokens=$(echo $config | cut -d: -f1)
    temp=$(echo $config | cut -d: -f2)
    iter=3
    
    echo -e "${BLUE}Config: $config ($tokens tokens, temp=$temp)${NC}"
    echo "Running $iter iterations..."
    
    total_duration=0
    total_tokens=0
    
    for i in $(seq 1 $iter); do
        echo -n "  Run $i/$iter: "
        
        start_time=$(date +%s.%N)
        output=$(./bin/metal_benchmark -model "$MODEL_PATH" -prompt "$PROMPT" -tokens $tokens -temperature $temp -output json 2>/dev/null)
        end_time=$(date +%s.%N)
        
        duration=$(echo "$end_time - $start_time" | bc)
        throughput=$(echo "scale=2; $tokens / $duration" | bc)
        
        printf "%6.2f tokens/sec (%.2fs) - %s\n" "$throughput" "$duration" "$output"
        
        total_duration=$(echo "$total_duration + $duration" | bc)
        total_tokens=$((total_tokens + tokens))
    done
    
    avg_duration=$(echo "$total_duration / $iter" | bc)
    avg_throughput=$(echo "scale=2; $total_tokens / $total_duration" | bc)
    
    echo "  Average: ${avg_duration}s"
    echo "  Average: ${avg_throughput} tokens/sec"
    echo
    
    result_file="$OUTPUT_DIR/config_${config}.txt"
    {
        echo "Configuration: $config ($tokens tokens, temp=$temp)"
        echo "Iterations: $iter"
        echo "Average Duration: ${avg_duration}s"
        echo "Average Throughput: ${avg_throughput} tokens/sec"
        echo "Average Tokens/sec: $avg_throughput"
        echo "Memory Usage: ~$(echo "scale=0; 5000 / 1024 / 1024 * $tokens" | bc)MB"
    } > "$result_file"
    
    echo "Results saved to: $result_file"
    echo
    
    echo -e "${BLUE}Recommendations:${NC}"
    echo "  - Use larger batch sizes for better throughput"
    echo "  - Experiment with different temperatures"
    echo "  - Profile memory usage patterns"
    echo "  - Test different model sizes for scaling analysis"
done