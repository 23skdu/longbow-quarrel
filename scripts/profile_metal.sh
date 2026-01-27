#!/bin/bash
set -e

# Configuration
OUTPUT_DIR="profiles"
mkdir -p "$OUTPUT_DIR"

# Models (Name:Path)
MODELS=(
    "Mistral:/Users/rsd/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f"
    "Granite4:/Users/rsd/.ollama/models/blobs/sha256-5c7ac4aead1bcf4c8da9534ed72cc632d005aeed6547f1e8662ccdfae688364e"
    "Nemotron:/Users/rsd/.ollama/models/blobs/sha256-a70437c41b3b0b768c48737e15f8160c90f13dc963f5226aabb3a160f708d1ce"
    "TinyLlama:/Users/rsd/.ollama/models/blobs/sha256-2af3b81862c6be03c769683af18efdadb2c33f60ff32ab6f83e42c043d6c7816"
)

PROMPT="The quick brown fox jumps over the lazy dog"
TOKENS=10

echo "Starting Profiling Run..."
echo "Output Directory: $OUTPUT_DIR"

for entry in "${MODELS[@]}"; do
    NAME="${entry%%:*}"
    PATH_VAL="${entry#*:}"
    
    echo "========================================"
    echo "Profiling $NAME"
    echo "Path: $PATH_VAL"
    
    # 1. Capture JSON Baseline
    echo "Running Baseline Benchmark..."
    # Capture only the last line which contains the JSON
    ./bin/metal_benchmark -model "$PATH_VAL" -prompt "$PROMPT" -tokens $TOKENS -output json > "$OUTPUT_DIR/${NAME}_baseline.json" 2>&1 || true
    
    # 2. Capture Metal Trace (if xctrace is available)
    if command -v xcrun &> /dev/null; then
        echo "Capturing Metal Trace..."
        TRACE_FILE="$OUTPUT_DIR/${NAME}.trace"
        rm -rf "$TRACE_FILE"
        
        # Note: xctrace record attaches to the process
        xcrun xctrace record --template 'Metal System Trace' --output "$TRACE_FILE" --launch -- ./bin/metal_benchmark -model "$PATH_VAL" -prompt "$PROMPT" -tokens $TOKENS > /dev/null 2>&1 || echo "Trace failed (likely permissions or locked device)"
        
        echo "Trace saved to $TRACE_FILE"
    else
        echo "xcrun not found, skipping trace."
    fi
    
    echo "Done with $NAME"
done

echo "Profiling Complete. Results in $OUTPUT_DIR"
