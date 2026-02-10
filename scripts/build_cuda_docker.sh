#!/bin/bash
# =============================================================================
# Build and Test CUDA Docker Image
# =============================================================================

set -e

IMAGE_NAME="longbow-quarrel:cuda"
CONTAINER_NAME="quarrel-test"
MODEL_PATH="${MODEL_PATH:-./models}"
MODEL_NAME="${MODEL:-smollm2.gguf}"

echo "=== Longbow-Quarrel CUDA Docker Build & Test ==="

# Check NVIDIA Container Toolkit
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. GPU may not be available in container."
fi

# Build the image
echo ""
echo "[1/3] Building CUDA image..."
docker build -f Dockerfile.cuda -t "$IMAGE_NAME" .

# Verify nvidia runtime
echo ""
echo "[2/3] Testing NVIDIA runtime..."
docker run --rm --gpus all nvidia/cuda:12.4-runtime-ubuntu22.04 nvidia-smi

# Create model directory if needed
mkdir -p "$MODEL_PATH"

if [ ! -f "$MODEL_PATH/$MODEL_NAME" ]; then
    echo ""
    echo "[3/3] Note: Model not found at $MODEL_PATH/$MODEL_NAME"
    echo "Place your GGUF model in that location or mount it at /data"
else
    echo ""
    echo "[3/3] Running inference test..."
    docker run --rm --gpus all \
        -v "$MODEL_PATH:/data:ro" \
        --name "$CONTAINER_NAME" \
        "$IMAGE_NAME" \
        --model "/data/$MODEL_NAME" \
        --prompt "The capital of France is" \
        --n 10
fi

echo ""
echo "=== Build Complete ==="
echo ""
echo "Usage:"
echo "  docker run --gpus all -v \$(pwd)/models:/data $IMAGE_NAME \\"
echo "      --model /data/$MODEL_NAME --prompt 'Hello' --n 50"
echo ""
echo "Or with docker-compose:"
echo "  MODEL=smollm2.gguf MODEL_PATH=./models docker-compose -f docker-compose.cuda.yml up quarrel-cuda"
