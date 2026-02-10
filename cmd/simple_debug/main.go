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
		fmt.Println("Usage: simple_debug <model_path>")
		os.Exit(1)
	}

	modelPath := os.Args[1]

	conf := config.Default()
	e, err := engine.NewEngine(modelPath, conf)
	if err != nil {
		fmt.Printf("ERROR: Failed to create engine: %v\n", err)
		os.Exit(1)
	}
	defer e.Close()

	ggufFile, err := gguf.LoadFile(modelPath)
	if err != nil {
		fmt.Printf("ERROR: Failed to load GGUF: %v\n", err)
		os.Exit(1)
	}
	defer ggufFile.Close()

	tokenizer, err := tokenizer.NewFromGGUF(ggufFile)
	if err != nil {
		fmt.Printf("ERROR: Failed to create tokenizer: %v\n", err)
		os.Exit(1)
	}

	prompt := "The capital of France is"

	inputTokens := tokenizer.Encode(prompt)
	if len(inputTokens) == 0 {
		fmt.Println("ERROR: No input tokens encoded")
		os.Exit(1)
	}

	fmt.Printf("=== SIMPLE DEBUG TOOL ===\n")
	fmt.Printf("Model: %s\n", modelPath)
	fmt.Printf("Prompt: %q\n", prompt)
	fmt.Printf("Input tokens (%d): %v\n", len(inputTokens), inputTokens)
	fmt.Printf("\n")

	sampler := engine.SamplerConfig{Temperature: 0, TopK: 1}
	tokensToGenerate := 3

	allGenerated := make([]int, 0, tokensToGenerate)

	for i := 0; i < tokensToGenerate; i++ {
		fmt.Printf("\n--- Generation %d/%d ---\n", i+1, tokensToGenerate)

		var generated []int
		var err error

		if i == 0 {
			// First token: from input
			generated, err = e.Infer(inputTokens, 1, sampler)
			fmt.Printf("Input: %v tokens\n", inputTokens)
		} else {
			// Subsequent tokens: from previous output
			prevTokens := make([]int, i)
			for j := 0; j < i; j++ {
				prevTokens[j] = allGenerated[j]
			}
			generated, err = e.Infer(prevTokens, 1, sampler)
			fmt.Printf("Previous %d tokens: %v\n", prevTokens)
		}

		if err != nil {
			fmt.Printf("ERROR: Inference failed at step %d: %v\n", i+1, err)
			os.Exit(1)
		}

		if len(generated) != 1 {
			fmt.Printf("ERROR: Expected 1 token, got %d\n", len(generated))
			os.Exit(1)
		}

		tokenID := generated[0]
		allGenerated[i] = tokenID

		tokenText := tokenizer.Decode([]int{tokenID})
		fmt.Printf("Generated token ID: %d\n", tokenID)
		fmt.Printf("Text: %q\n", tokenText)

		// Show raw token chars
		tokenStr := tokenizer.Decode([]int{tokenID})
		tokenChars := []rune(tokenStr)
		fmt.Printf("Raw characters: %v\n", tokenChars)
	}

	fmt.Printf("\n=== SUMMARY ===\n")
	fullTokens := append(inputTokens, allGenerated...)
	fmt.Printf("All tokens: %v\n", fullTokens)
	fmt.Printf("Decoded: %q\n", tokenizer.Decode(fullTokens))
	fmt.Printf("Expected (manual): The capital of France is\n")
}
