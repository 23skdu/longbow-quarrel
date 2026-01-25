//go:build darwin && metal

package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/engine"
	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
)

type BenchmarkResult struct {
	Model      string        `json:"model"`
	Prompt     string        `json:"prompt"`
	Tokens     int           `json:"tokens"`
	Duration   time.Duration `json:"duration"`
	Throughput float64       `json:"throughput_tokens_per_sec"`
	Output     string        `json:"output"`
	Timestamp  time.Time     `json:"timestamp"`
}

func main() {
	modelPath := flag.String("model", "", "Path to GGUF model file")
	prompt := flag.String("prompt", "The quick brown fox jumps over the lazy dog.", "Prompt for generation")
	tokens := flag.Int("tokens", 32, "Number of tokens to generate")
	outputFormat := flag.String("output", "text", "Output format: text, json")
	iterations := flag.Int("iterations", 1, "Number of benchmark iterations")

	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintf(os.Stderr, "Usage: %s -model <gguf-path> [-prompt <text>] [-tokens <num>] [-output <text|json>] [-iterations <num>]\n", os.Args[0])
		os.Exit(1)
	}

	var results []BenchmarkResult

	for i := 0; i < *iterations; i++ {
		result := runBenchmark(*modelPath, *prompt, *tokens)
		results = append(results, result)

		if *outputFormat == "text" {
			fmt.Printf("=== Iteration %d ===\n", i+1)
			fmt.Printf("Duration: %v\n", result.Duration)
			fmt.Printf("Throughput: %.2f tokens/sec\n", result.Throughput)
			fmt.Printf("Output: %s\n\n", result.Output)
		}
	}

	if *outputFormat == "json" {
		if len(results) == 1 {
			json.NewEncoder(os.Stdout).Encode(results[0])
		} else {
			// Calculate average for multiple iterations
			avg := calculateAverage(results)
			json.NewEncoder(os.Stdout).Encode(avg)
		}
	}
}

func runBenchmark(modelPath, prompt string, tokens int) BenchmarkResult {
	config := engine.EngineConfig{
		DebugDequant: false,
		KVCacheSize:  32,
	}

	e, err := engine.NewEngine(modelPath, config)
	if err != nil {
		log.Fatalf("Failed to create engine: %v", err)
	}
	defer e.Close()

	// Load tokenizer
	tok, err := tokenizer.New(modelPath)
	if err != nil {
		log.Fatalf("Failed to load tokenizer: %v", err)
	}

	// Tokenize prompt
	promptTokens := tok.Encode(prompt)
	// Add BOS token
	promptTokens = append([]int{1}, promptTokens...)

	samplerConfig := engine.SamplerConfig{
		Temperature: 0.7,
		TopK:        40,
		TopP:        0.95,
		RepPenalty:  1.0,
	}

	start := time.Now()
	resultTokens, err := e.Infer(promptTokens, tokens, samplerConfig)
	if err != nil {
		log.Fatalf("Inference failed: %v", err)
	}
	duration := time.Since(start)

	// Decode output
	output := tok.Decode(resultTokens)
	throughput := float64(tokens) / duration.Seconds()

	return BenchmarkResult{
		Model:      modelPath,
		Prompt:     prompt,
		Tokens:     tokens,
		Duration:   duration,
		Throughput: throughput,
		Output:     output,
		Timestamp:  time.Now(),
	}
}

func calculateAverage(results []BenchmarkResult) BenchmarkResult {
	if len(results) == 0 {
		return BenchmarkResult{}
	}

	var totalDuration time.Duration
	var totalThroughput float64

	for _, result := range results {
		totalDuration += result.Duration
		totalThroughput += result.Throughput
	}

	avgResult := results[0]
	avgResult.Duration = totalDuration / time.Duration(len(results))
	avgResult.Throughput = totalThroughput / float64(len(results))
	avgResult.Output = "[average result - output omitted]"
	avgResult.Timestamp = time.Now()

	return avgResult
}
