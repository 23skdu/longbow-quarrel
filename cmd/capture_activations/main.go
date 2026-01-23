//go:build darwin && metal

package main

import (
	"flag"
	"fmt"
	"log"

	"github.com/23skdu/longbow-quarrel/internal/engine"
	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
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
	}

	fmt.Printf("Loading model: %s\n", resolvedPath)
	e, err := engine.NewEngine(resolvedPath, false)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	defer e.Close()

	fmt.Printf("Model Config: RoPE Theta = %f, HeadDim = %d, Heads = %d, KVHeads = %d\n",
		e.Config.RopeTheta, e.Config.HeadDim, e.Config.Heads, e.Config.KVHeads)

	fmt.Printf("Tokenizing prompt: %q\n", *prompt)
	tok, err := tokenizer.New(resolvedPath)
	if err != nil {
		log.Fatalf("Failed to load tokenizer: %v", err)
	}

	tokens := tok.Encode(*prompt)
	// Prepend BOS token (1)
	tokens = append([]int{1}, tokens...)
	fmt.Printf("Tokens: %v\n", tokens)

	fmt.Printf("Enabling activation logging...\n")
	e.ActLogger.Enable(*prompt, tokens)

	// Run inference
	fmt.Printf("Running inference...\n")
	config := engine.SamplerConfig{
		Temperature: 0.0,
		RepPenalty:  1.1,
	}

	// Run for 1 token to capture activations
	result, err := e.Infer(tokens, 1, config)
	if err != nil {
		log.Fatalf("Inference failed: %v", err)
	}

	fmt.Printf("Generated tokens: %v\n", result)

	// Access logits from ActLogger.log.FinalLogits if stored

	// Save activations
	fmt.Printf("Saving activations to: %s\n", *outputFile)
	err = e.ActLogger.SaveToFile(*outputFile)
	if err != nil {
		log.Fatalf("Failed to save activations: %v", err)
	}

	fmt.Printf("Done!\n")
}
