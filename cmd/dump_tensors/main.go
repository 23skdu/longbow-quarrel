//go:build darwin && metal

package main

import (
	"fmt"
	"os"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/engine"
	"github.com/23skdu/longbow-quarrel/internal/ollama"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: dump_tensors <model-name>")
		os.Exit(1)
	}

	modelName := os.Args[1]
	modelPath, err := ollama.ResolveModelPath(modelName)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Loading model: %s\n", modelPath)

	cfg := config.Default()
	e, err := engine.NewEngine(modelPath, cfg)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	defer e.Close()
	fmt.Println("Model loaded successfully.")
}
