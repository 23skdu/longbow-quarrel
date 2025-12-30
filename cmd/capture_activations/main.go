package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	
	"github.com/23skdu/longbow-quarrel/internal/engine"
)

func main() {
	modelPath := flag.String("model", "", "Path to model file or model name (mistral, smollm2:135m)")
	prompt := flag.String("prompt", "The capital of France is", "Input prompt")
	outputFile := flag.String("output", "quarrel_activations.json", "Output JSON file for activations")
	flag.Parse()
	
	if *modelPath == "" {
		log.Fatal("--model required")
	}
	
	// Resolve model name to path
	resolvedPath := *modelPath
	if *modelPath == "mistral" {
		resolvedPath = "/Users/rsd/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f"
	} else if *modelPath == "smollm2:135m" {
		resolvedPath = "/Users/rsd/.ollama/models/blobs/sha256-5beabd937cfa6262c97dbf2df04034c00f625160c3746e5e696e6d2a3cf05959"
	}
	
	fmt.Printf("Loading model: %s\n", resolvedPath)
	e, err := engine.NewEngine(resolvedPath, false)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	
	fmt.Printf("Tokenizing prompt: %q\n", *prompt)
	// TODO: Load tokenizer and encode prompt
	// For now, use hardcoded tokens for "The capital of France is"
	tokens := []int{1782, 6333, 1070, 5611, 1117}
	
	fmt.Printf("Tokens: %v\n", tokens)
	fmt.Printf("Enabling activation logging...\n")
	
	// Enable activation logging
	e.ActLogger.Enable(*prompt, tokens)
	
	// Run inference for one token
	fmt.Printf("Running inference...\n")
	config := engine.SamplerConfig{
		Temperature:      0.0,
		RepetitionPenalty: 1.1,
	}
	
	result := e.Infer(tokens, 1, config)
	
	fmt.Printf("Generated tokens: %v\n", result)
	
	// Save activations
	fmt.Printf("Saving activations to: %s\n", *outputFile)
	err = e.ActLogger.SaveToFile(*outputFile)
	if err != nil {
		log.Fatalf("Failed to save activations: %v", err)
	}
	
	fmt.Printf("Done!\n")
}
