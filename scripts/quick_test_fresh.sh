#!/bin/bash

MODEL="/Users/rsd/.ollama/models/blobs/sha256-063ec4f237009d975b410461c091922e374c1c7c1dc1b8304226a5d927b"
LONGTOOL="./cmd/generate_text/main.go"

echo "Testing fresh TinyLlama model..."
echo "Model: $MODEL"
echo ""

# Test single token generation
echo "=== Test 1: Single token ==="
$LONGTOOL -model="$MODEL" -prompt="The" -len=1 -temp=0 2>&1 | tail -10
RESULT_1=$?
echo ""

echo "=== Test 2: Multi-token ==="
$LONGTOOL -model="$MODEL" -prompt="The capital of France is" -len=3 -temp=0 2>&1 | tail -10
RESULT_2=$?
echo ""

if [ $RESULT_1 -eq 0 ] && [ $RESULT_2 -eq 0 ]; then
    echo "✅ Both tests passed"
else
    echo "⚠️  Some tests failed: Test1=$RESULT_1 Test2=$RESULT_2"
fi
