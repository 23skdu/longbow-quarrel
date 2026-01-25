package main

import (
	"fmt"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/engine"
)

func main() {
	modelPath := "llama3:latest"
	fmt.Printf("Testing with model: %s\n", modelPath)

	e, err := engine.NewEngine(modelPath, engine.EngineConfig{
		KVCacheSize: 4096,
	})
	if err != nil {
		fmt.Printf("Engine error: %v\n", err)
		return
	}

	prompt := "The quick brown fox jumps"
	tokens := []int{1, 504, 38478, 22216}

	start := time.Now()
	output, err := e.Infer(tokens, prompt, nil)
	if err != nil {
		fmt.Printf("Generation error: %v\n", err)
		return
	}
	duration := time.Since(start)

	fmt.Printf("Engine Import Test Results:\n")
	fmt.Printf("Model: %s\n", modelPath)
	fmt.Printf("Tokens: %d\n", len(tokens))
	fmt.Printf("Duration: %v\n", duration)
	fmt.Printf("Tokens/sec: %.2f\n", float64(len(tokens))/duration.Seconds())
	fmt.Printf("Output: %s\n", output)
}
