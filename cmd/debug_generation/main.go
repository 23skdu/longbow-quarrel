//go:build darwin && metal

package main

import (
	"fmt"
	"os"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/engine"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: debug_generation <model_path> [prompt]")
		os.Exit(1)
	}

	modelPath := os.Args[1]
	prompt := "The capital of France is"
	if len(os.Args) > 2 {
		prompt = os.Args[2]
	}

	conf := config.Default()
	e, err := engine.NewEngine(modelPath, conf)
	if err != nil {
		fmt.Printf("Error creating engine: %v\n", err)
		os.Exit(1)
	}
	defer e.Close()

	ggufFile, err := gguf.LoadFile(modelPath)
	if err != nil {
		fmt.Printf("Error loading GGUF: %v\n", err)
		os.Exit(1)
	}
	defer ggufFile.Close()

	tok, err := tokenizer.NewFromGGUF(ggufFile)
	if err != nil {
		fmt.Printf("Error creating tokenizer: %v\n", err)
		os.Exit(1)
	}

	inputTokens := tok.Encode(prompt)
	fmt.Printf("=== Input ===\n")
	fmt.Printf("Prompt: %q\n", prompt)
	fmt.Printf("Tokens: %v\n", inputTokens)
	fmt.Printf("\n")

	sampler := engine.SamplerConfig{Temperature: 0, TopK: 1}
	tokensToGenerate := 3

	fmt.Printf("=== Generation (%d tokens) ===\n", tokensToGenerate)

	allGenerated := make([]int, 0, tokensToGenerate)

	for i := 0; i < tokensToGenerate; i++ {
		var generated []int
		var err error

		if i == 0 {
			fmt.Printf("\n--- Token %d/%d (from prompt) ---\n", i+1, tokensToGenerate)
			generated, err = e.Infer(inputTokens, 1, sampler)
		} else {
			fmt.Printf("\n--- Token %d/%d (from previous) ---\n", i+1, tokensToGenerate)
			prevTokens := make([]int, i)
			for j := 0; j < i; j++ {
				prevTokens[j] = allGenerated[j]
			}
			generated, err = e.Infer(prevTokens, 1, sampler)
		}

		if err != nil {
			fmt.Printf("ERROR: Inference failed: %v\n", err)
			os.Exit(1)
		}

		if len(generated) != 1 {
			fmt.Printf("ERROR: Expected 1 token, got %d\n", len(generated))
			os.Exit(1)
		}

		tokenID := generated[0]
		allGenerated[i] = tokenID

		tokenText := tok.Decode([]int{tokenID})
		fmt.Printf("Token ID: %d\n", tokenID)
		fmt.Printf("Text: %q\n", tokenText)
	}

	fmt.Printf("\n=== Full Output ===\n")
	fullTokens := append(inputTokens, allGenerated...)
	fmt.Printf("All tokens: %v\n", fullTokens)
	fmt.Printf("Decoded: %q\n", tok.Decode(fullTokens))
}
