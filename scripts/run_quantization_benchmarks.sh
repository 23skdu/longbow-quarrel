#!/bin/bash
set -e

RESULTS_DIR="$HOME/REPOS/longbow-quarrel/benchmark_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p "$RESULTS_DIR"

echo "=== Quantization Benchmark Runner ==="
echo "Timestamp: $TIMESTAMP"

cd ~/REPOS/longbow-quarrel/internal/gguf

echo "Running GGUF Q4_K quant benchmarks..."
go test -v -run=NONE -bench=BenchmarkQuantizeQ4K -benchtime=1x -count=5 . 2>&1 | grep "^Benchmark" > "$RESULTS_DIR/gguf_q4k_quant.txt"

echo "Running GGUF Q4_K dequant benchmarks..."
go test -v -run=NONE -bench=BenchmarkDequantizeQ4K -benchtime=1x -count=5 . 2>&1 | grep "^Benchmark" > "$RESULTS_DIR/gguf_q4k_dequant.txt"

echo "Running GGUF TurboQuant benchmarks..."
go test -v -run=NONE -bench=BenchmarkTurboQuant -benchtime=1x -count=5 . 2>&1 | grep "^Benchmark" > "$RESULTS_DIR/gguf_turboquant.txt"

cd ~/REPOS/longbow-quarrel/internal/simd

echo "Running SIMD TurboQuant benchmarks..."
go test -v -run=NONE -bench=BenchmarkTurboQuant -benchtime=1x -count=5 . 2>&1 | grep "^Benchmark" > "$RESULTS_DIR/simd_turboquant.txt"

echo "Running SIMD MatMul benchmarks..."
go test -v -run=NONE -bench=BenchmarkSIMD_MatMul -benchtime=1x -count=5 . 2>&1 | grep "^Benchmark" > "$RESULTS_DIR/simd_matmul.txt"

echo "Done! Results in $RESULTS_DIR/"
