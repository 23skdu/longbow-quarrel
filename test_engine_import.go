package main

import (
	"fmt"
	"os"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/engine"
	"github.com/23skdu/longbow-quarrel/internal/ollama"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Testing engine imports...")
		os.Exit(1)
	}

	// Use Ollama model resolver like benchmark does
	modelName := os.Args[1]
	modelPath, err := ollama.ResolveModelPath(modelName)
	if err != nil {
		fmt.Printf("Resolution error: %v\n", err)
		return
	}

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
	output, err := e.Generate(tokens, prompt, nil)
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
