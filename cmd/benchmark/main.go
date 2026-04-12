package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"runtime"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/cpu"
	"github.com/23skdu/longbow-quarrel/internal/engine"
	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
)

var (
	size       = flag.Int("size", 4096, "Tensor size for CPU kernel benchmark")
	iterations = flag.Int("iterations", 100, "Number of iterations for CPU benchmark")
	modelPath  = flag.String("model", "", "Path to GGUF model for inference benchmark")
	prompt     = flag.String("prompt", "The quick brown fox jumps over the lazy dog", "Benchmark prompt")
	maxLen     = flag.Int("len", 50, "Max tokens to generate per run")
	warmup     = flag.Int("warmup", 2, "Number of warmup runs")
	runs       = flag.Int("runs", 5, "Number of benchmark runs")
	temp       = flag.Float64("temp", 0.0, "Temperature (0 for greedy)")
	mode       = flag.String("mode", "kernel", "Benchmark mode: kernel, inference")
)

func benchmarkSoftmax(ctx *cpu.Context, size int, iterations int) time.Duration {
	input := ctx.NewTensor([2]int{1, size}, 4)
	output := ctx.NewTensor([2]int{1, size}, 4)
	data := ctx.GetTensorData(input).([]float32)
	for i := range data {
		data[i] = float32(i%100) / 10.0
	}

	start := time.Now()
	for i := 0; i < iterations; i++ {
		ctx.SoftmaxF32(input, output)
	}
	elapsed := time.Since(start)
	ctx.PutTensor(input)
	ctx.PutTensor(output)
	return elapsed
}

func benchmarkSwiGLU(ctx *cpu.Context, size int, iterations int) time.Duration {
	gate := ctx.NewTensor([2]int{1, size}, 4)
	up := ctx.NewTensor([2]int{1, size}, 4)
	out := ctx.NewTensor([2]int{1, size}, 4)
	gData := ctx.GetTensorData(gate).([]float32)
	uData := ctx.GetTensorData(up).([]float32)
	for i := range gData {
		gData[i] = float32(i % 100)
		uData[i] = float32(i % 100)
	}

	start := time.Now()
	for i := 0; i < iterations; i++ {
		ctx.SwiGLU(gate, up, out)
	}
	elapsed := time.Since(start)
	ctx.PutTensor(gate)
	ctx.PutTensor(up)
	ctx.PutTensor(out)
	return elapsed
}

func benchmarkLinear(ctx *cpu.Context, m, k, n int, iterations int) time.Duration {
	weight := ctx.NewTensor([2]int{k, n}, 4)
	input := ctx.NewTensor([2]int{1, k}, 4)
	output := ctx.NewTensor([2]int{1, n}, 4)

	wData := ctx.GetTensorData(weight).([]float32)
	inData := ctx.GetTensorData(input).([]float32)
	for i := range wData {
		wData[i] = float32(i % 100)
	}
	for i := range inData {
		inData[i] = float32(i % 100)
	}

	start := time.Now()
	for i := 0; i < iterations; i++ {
		ctx.LinearF32(weight, input, output)
	}
	elapsed := time.Since(start)
	ctx.PutTensor(weight)
	ctx.PutTensor(input)
	ctx.PutTensor(output)
	return elapsed
}

func benchmarkRMSNorm(ctx *cpu.Context, size int, iterations int) time.Duration {
	input := ctx.NewTensor([2]int{1, size}, 4)
	weight := ctx.NewTensor([2]int{1, size}, 4)
	output := ctx.NewTensor([2]int{1, size}, 4)
	inData := ctx.GetTensorData(input).([]float32)
	wData := ctx.GetTensorData(weight).([]float32)
	for i := range inData {
		inData[i] = float32(i % 100)
	}
	for i := range wData {
		wData[i] = 1.0
	}

	start := time.Now()
	for i := 0; i < iterations; i++ {
		ctx.RMSNorm(input, weight, output, 1e-5)
	}
	elapsed := time.Since(start)
	ctx.PutTensor(input)
	ctx.PutTensor(weight)
	ctx.PutTensor(output)
	return elapsed
}

func printResult(name string, elapsed time.Duration, iterations int, size int) {
	avgMs := float64(elapsed.Microseconds()) / float64(iterations) / 1000.0
	opsPerSec := float64(iterations) / elapsed.Seconds()
	throughput := float64(iterations*size) / elapsed.Seconds() / 1e6
	fmt.Printf("%-20s: %7.3f ms/op, %10.0f ops/s, %8.2f M elements/s\n",
		name, avgMs, opsPerSec, throughput)
}

func main() {
	flag.Parse()

	if *mode == "inference" {
		runInferenceBenchmark()
		return
	}

	fmt.Println("=== Longbow-Quarrel CPU Kernel Benchmark ===")
	fmt.Printf("Go Version: %s\n", runtime.Version())
	fmt.Printf("NumCPU: %d\n", runtime.NumCPU())
	fmt.Printf("Iterations: %d\n", *iterations)
	fmt.Println()

	ctx := cpu.NewContext()
	defer ctx.Free()

	fmt.Println("--- Softmax (float32) ---")
	elapsed := benchmarkSoftmax(ctx, *size, *iterations)
	printResult(fmt.Sprintf("Softmax(%d)", *size), elapsed, *iterations, *size)

	fmt.Println()
	fmt.Println("--- SwiGLU ---")
	elapsed = benchmarkSwiGLU(ctx, *size, *iterations)
	printResult(fmt.Sprintf("SwiGLU(%d)", *size), elapsed, *iterations, *size)

	fmt.Println()
	fmt.Println("--- RMSNorm ---")
	elapsed = benchmarkRMSNorm(ctx, *size, *iterations)
	printResult(fmt.Sprintf("RMSNorm(%d)", *size), elapsed, *iterations, *size)

	fmt.Println()
	fmt.Println("--- Linear Layer (FP32) ---")
	m, k, n := 1, 4096, 4096
	elapsed = benchmarkLinear(ctx, m, k, n, *iterations)
	printResult(fmt.Sprintf("Linear(%d×%d×%d)", m, k, n), elapsed, *iterations, m*n)

	os.Exit(0)
}

func runInferenceBenchmark() {
	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "Error: --model flag required for inference benchmark")
		os.Exit(1)
	}

	fmt.Printf("=== Longbow-Quarrel Inference Benchmark ===\n")
	fmt.Printf("Model: %s\n", *modelPath)
	fmt.Printf("Prompt: %s\n", *prompt)
	fmt.Printf("Max tokens: %d\n", *maxLen)
	fmt.Printf("Warmup runs: %d\n", *warmup)
	fmt.Printf("Benchmark runs: %d\n", *runs)
	fmt.Println()

	conf := config.Default()
	conf.KVCacheSize = 2048

	e, err := engine.NewEngine(*modelPath, conf)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	defer e.Close()

	tok, err := tokenizer.New(*modelPath)
	if err != nil {
		log.Fatalf("Failed to load tokenizer: %v", err)
	}

	tokens := tok.Encode(*prompt)
	tokens = append([]int{1}, tokens...)
	promptLen := len(tokens)

	fmt.Printf("Prompt tokens: %d\n", promptLen)
	fmt.Println()

	// Warmup runs
	fmt.Println("Running warmup...")
	for i := 0; i < *warmup; i++ {
		_, err := e.Infer(tokens, *maxLen, engine.SamplerConfig{Temperature: *temp})
		if err != nil {
			log.Printf("Warmup %d failed: %v", i, err)
		}
	}

	// Benchmark runs
	var totalTime time.Duration
	var totalTokens int

	fmt.Println("Running benchmark...")
	for i := 0; i < *runs; i++ {
		start := time.Now()
		resultTokens, err := e.Infer(tokens, *maxLen, engine.SamplerConfig{Temperature: *temp})
		elapsed := time.Since(start)

		if err != nil {
			log.Printf("Run %d failed: %v", i, err)
			continue
		}

		generated := len(resultTokens) - promptLen
		tokensPerSec := float64(generated) / elapsed.Seconds()

		fmt.Printf("Run %d: %d tokens in %.2fs (%.1f tokens/s)\n",
			i+1, generated, elapsed.Seconds(), tokensPerSec)

		totalTime += elapsed
		totalTokens += generated
	}

	avgTime := totalTime / time.Duration(*runs)
	avgTokens := totalTokens / *runs
	avgTokensPerSec := float64(avgTokens) / avgTime.Seconds()

	fmt.Println()
	fmt.Printf("=== Results ===\n")
	fmt.Printf("Average: %d tokens in %.2fs (%.1f tokens/s)\n",
		avgTokens, avgTime.Seconds(), avgTokensPerSec)
}
