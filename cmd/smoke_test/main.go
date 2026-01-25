package main

import (
	"context"
	"fmt"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/engine"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

func main() {
	modelPath := "/Users/rsd/.ollama/models/blobs/sha256-f535f83ec568d040f88ddc04a199fa6da90923bbb41d4dcaed02caa924d6ef57"

	fmt.Printf("=== Smoke Test ===\n")

	// Create engine with Metal acceleration
	config := engine.EngineConfig{
		KVCacheSize: 32,
	}
	e, err := engine.NewEngine(modelPath, config)
	if err != nil {
		fmt.Printf("Engine creation failed: %v\n", err)
		return
	}
	defer e.Close()

	// Perform single token inference test
	ctx := context.Background()
	start := time.Now()

	prompt := "Hello"
	output, err := e.Infer(ctx, prompt, 1)
	if err != nil {
		fmt.Printf("Inference failed: %v\n", err)
		return
	}

	duration := time.Since(start)

	fmt.Printf("Smoke test completed successfully\n")
	fmt.Printf("Prompt: %q\n", prompt)
	fmt.Printf("Output: %q\n", output)
	fmt.Printf("Duration: %v\n", duration)
	fmt.Printf("Tokens/sec: %.2f\n", float64(1)/duration.Seconds())
}
