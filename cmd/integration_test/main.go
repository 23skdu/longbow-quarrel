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

func main() {
	modelPath := flag.String("model", "", "Path to GGUF model file")
	tokens := flag.Int("tokens", 10, "Number of tokens to generate")
	iterations := flag.Int("iterations", 100, "Number of inference iterations for performance measurement")
	warmup := flag.Int("warmup", 5, "Number of warmup iterations")
	outputJSON := flag.Bool("json", false, "Output results in JSON format")

	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintf(os.Stderr, "Usage: %s -model <gguf-path> [-tokens N] [-iterations N] [-warmup N] [-json]\n", os.Args[0])
		os.Exit(1)
	}

	fmt.Printf("=== Longbow-Quarrel Integration Test ===\n")
	fmt.Printf("Model: %s\n", *modelPath)
	fmt.Printf("Tokens per generation: %d\n", *tokens)
	fmt.Printf("Total iterations: %d (including %d warmup)\n", *iterations, *warmup)
	fmt.Printf("Start time: %s\n", time.Now().Format("2006-01-02 15:04:05"))

	// Create engine
	config := engine.EngineConfig{
		KVCacheSize: 64, // 64 MiB cache
	}
	e, err := engine.NewEngine(*modelPath, config)
	if err != nil {
		log.Fatalf("Failed to create engine: %v", err)
	}
	defer e.Close()

	// Performance measurement struct
	type InferenceResult struct {
		Iteration    int           `json:"iteration"`
		Duration     time.Duration `json:"duration_ms"`
		TokensPerSec float64       `json:"tokens_per_sec"`
		Latency      float64       `json:"latency_ms"`
		Success      bool          `json:"success"`
		Error        string        `json:"error,omitempty"`
	}

	results := make([]InferenceResult, 0, *iterations*2)

	// Warmup phase
	fmt.Printf("\n=== Warmup Phase (%d iterations) ===\n", *warmup)
	for i := 0; i < *warmup; i++ {
		start := time.Now()
		prompt := fmt.Sprintf("Warmup iteration %d", i+1)
		output, err := e.Infer(prompt, *tokens)
		if err != nil {
			results = append(results, InferenceResult{
				Iteration: i + 1,
				Error:     err.Error(),
			})
			continue
		}

		duration := time.Since(start)
		tokensPerSec := float64(*tokens) / duration.Seconds()
		latency := float64(duration.Nanoseconds()) / float64(*tokens) * 1000000

		results = append(results, InferenceResult{
			Iteration:    i + 1,
			Duration:     duration,
			TokensPerSec: tokensPerSec,
			Latency:      latency,
			Success:      true,
		})

		fmt.Printf("Warmup %d: %s (%.2f t/s, %.2f ms latency)\n",
			i+1, output, tokensPerSec, latency)
	}

	// Measurement phase
	fmt.Printf("\n=== Measurement Phase (%d iterations) ===\n", *iterations)
	measurementStart := time.Now()

	for i := 0; i < *iterations; i++ {
		start := time.Now()
		prompt := fmt.Sprintf("Test iteration %d", i+1)
		output, err := e.Infer(prompt, *tokens)
		if err != nil {
			results = append(results, InferenceResult{
				Iteration: *warmup + i + 1,
				Error:     err.Error(),
			})
			continue
		}

		duration := time.Since(start)
		tokensPerSec := float64(*tokens) / duration.Seconds()
		latency := float64(duration.Nanoseconds()) / float64(*tokens) * 1000000

		results = append(results, InferenceResult{
			Iteration:    *warmup + i + 1,
			Duration:     duration,
			TokensPerSec: tokensPerSec,
			Latency:      latency,
			Success:      true,
		})

		fmt.Printf("Measurement %d: %s (%.2f t/s, %.2f ms latency)\n",
			*warmup+i+1, output, tokensPerSec, latency)
	}

	measurementDuration := time.Since(measurementStart)
	totalDuration := time.Since(time.Time{})

	// Calculate statistics
	var totalTokensPerSec, totalLatency float64
	var successfulIterations int

	for _, result := range results {
		if result.Success {
			totalTokensPerSec += result.TokensPerSec
			totalLatency += result.Latency
			successfulIterations++
		}
	}

	avgTokensPerSec := totalTokensPerSec / float64(successfulIterations)
	avgLatency := totalLatency / float64(successfulIterations)

	fmt.Printf("\n=== Results Summary ===\n")
	fmt.Printf("Total duration: %v\n", totalDuration)
	fmt.Printf("Measurement duration: %v\n", measurementDuration)
	fmt.Printf("Successful iterations: %d/%d\n", successfulIterations, *iterations)
	fmt.Printf("Average throughput: %.2f tokens/sec\n", avgTokensPerSec)
	fmt.Printf("Average latency: %.2f ms/token\n", avgLatency)

	if *outputJSON {
		jsonResults, err := json.MarshalIndent(results, "", "  ")
		if err != nil {
			log.Printf("Failed to marshal JSON results: %v", err)
			return
		}
		fmt.Printf("\n=== JSON Results ===\n%s\n", string(jsonResults))
	} else {
		fmt.Printf("\n=== Text Results ===\n")
		for _, result := range results {
			if result.Success {
				fmt.Printf("Iteration %d: %s\n", result.Iteration, result.Duration)
			} else {
				fmt.Printf("Iteration %d failed: %s\n", result.Iteration, result.Error)
			}
		}
	}
}
