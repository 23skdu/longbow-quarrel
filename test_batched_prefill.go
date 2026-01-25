package main

import (
	"fmt"
	"os"
	"path/filepath"
	"time"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/engine"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run main.go <model-path>")
		os.Exit(1)
	}

	modelPath := os.Args[1]

	// Create engine (note: no batched strategy option available yet)
	e, err := engine.NewEngine(modelPath, engine.EngineConfig{
		KVCacheSize: 4096,
	})
	if err != nil {
		panic(err)
	}

	// Simple test
	prompt := "The quick brown fox jumps"
	tokens := []int{1, 504, 38478, 22216} // First 4 tokens

	start := time.Now()
	output, err := e.Generate(tokens, prompt, nil)
	if err != nil {
		panic(err)
	}
	duration := time.Since(start)

	fmt.Printf("Sequential Prefill Test Results:\n")
	fmt.Printf("Model: %s\n", modelPath)
	fmt.Printf("Tokens: %d\n", len(tokens))
	fmt.Printf("Duration: %v\n", duration)
	fmt.Printf("Tokens/sec: %.2f\n", float64(len(tokens))/duration.Seconds())
	fmt.Printf("Output: %s\n", output)
}

	modelPath := os.Args[1]

	// Load model
	g := gguf.NewGGUFReader(modelPath)
	if err := g.Load(); err != nil {
		panic(err)
	}

	// Create engine with batched prefill
	e, err := engine.NewEngine(g, engine.EngineConfig{
		PrefillStrategy: engine.PrefillStrategyBatched, // Use our batched optimization
	})
	if err != nil {
		panic(err)
	}

	// Simple test
	prompt := "The quick brown fox jumps"
	tokens := []int{1, 504, 38478} // First 3 tokens

	start := time.Now()
	output, err := e.Generate(tokens, prompt)
	if err != nil {
		panic(err)
	}
	duration := time.Since(start)

	fmt.Printf("Batched Prefill Test Results:\n")
	fmt.Printf("Model: %s\n", modelPath)
	fmt.Printf("Tokens: %d\n", len(tokens))
	fmt.Printf("Duration: %v\n", duration)
	fmt.Printf("Output: %s\n", output)
	fmt.Printf("Tokens/sec: %.2f\n", float64(len(tokens))/duration.Seconds())
}
