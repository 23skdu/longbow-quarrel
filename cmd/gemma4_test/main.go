//go:build darwin && metal

package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/engine"
	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
)

func main() {
	modelPath := flag.String("model", "", "Path to model file")
	prompt := flag.String("prompt", "The quick brown fox", "Input prompt")
	tokens := flag.Int("tokens", 10, "Tokens to generate")
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintf(os.Stderr, "--model required\n")
		os.Exit(1)
	}

	if _, err := os.Stat(*modelPath); os.IsNotExist(err) {
		fmt.Fprintf(os.Stderr, "Model not found: %s\n", *modelPath)
		os.Exit(1)
	}

	fmt.Printf("Loading model: %s\n", *modelPath)
	conf := config.Default()
	conf.KVCacheSize = 1024

	e, err := engine.NewEngine(*modelPath, conf)
	if err != nil {
		log.Fatalf("Failed to load engine: %v", err)
	}
	defer e.Close()

	tok, err := tokenizer.New(*modelPath)
	if err != nil {
		log.Fatalf("Failed to load tokenizer: %v", err)
	}

	inputTokens := tok.Encode(*prompt)
	fmt.Printf("Prompt: %s\n", *prompt)
	fmt.Printf("Input tokens: %d\n", len(inputTokens))

	sampler := engine.SamplerConfig{
		Temperature: 0.7,
		TopK:        40,
		TopP:        0.95,
	}

	result, err := e.Infer(inputTokens, *tokens, sampler)
	if err != nil {
		log.Fatalf("Inference failed: %v", err)
	}

	generated := tok.Decode(result)
	fmt.Printf("Generated (%d tokens): %s\n", len(result), generated)
}
