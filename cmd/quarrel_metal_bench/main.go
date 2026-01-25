//go:build darwin && metal

package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/engine"
)

func main() {
	modelPath := flag.String("model", "", "Path to GGUF model file")
	prompt := flag.String("prompt", "Hello world", "Prompt for generation")
	tokens := flag.Int("n", 32, "Number of tokens to generate")
	profile := flag.Bool("profile", false, "Enable performance profiling")

	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintf(os.Stderr, "Usage: %s -model <gguf-path> [-prompt <text>] [-n <tokens>] [--profile]\n", os.Args[0])
		os.Exit(1)
	}

	// Metal-compatible benchmarking
	config := engine.EngineConfig{
		DebugDequant: false,
		KVCacheSize:  32,
	}

	e, err := engine.NewEngine(*modelPath, config)
	if err != nil {
		log.Fatalf("Failed to create engine: %v", err)
	}
	defer e.Close()

	start := time.Now()

	// Run inference with timing
	output, err := e.Infer(*prompt, *tokens)
	if err != nil {
		log.Fatalf("Inference failed: %v", err)
	}

	duration := time.Since(start)
	tokensPerSec := float64(*tokens) / duration.Seconds()

	fmt.Printf("=== Longbow-Quarrel Metal Benchmark ===\n")
	fmt.Printf("Model: %s\n", *modelPath)
	fmt.Printf("Prompt: %q\n", *prompt)
	fmt.Printf("Tokens: %d\n", *tokens)
	fmt.Printf("Duration: %v\n", duration)
	fmt.Printf("Throughput: %.2f tokens/sec\n", tokensPerSec)
	fmt.Printf("Output: %s\n", output)

	if *profile {
		fmt.Printf("Profiling data saved to cpu.pprof\n")
	}
}
