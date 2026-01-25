//go:build darwin && metal

package main

import (
	"fmt"
	"os"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/engine"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run main.go <model-path>")
		os.Exit(1)
	}

	modelPath := os.Args[1]

	e, err := engine.NewEngine(modelPath, engine.EngineConfig{
		KVCacheSize: 4096,
	})
	if err != nil {
		panic(err)
	}

	prompt := "The quick brown fox jumps"
	tokens := []int{1, 504, 38478, 22216, 29343, 90}

	start := time.Now()
	output, err := e.Infer(tokens, prompt, nil)
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
