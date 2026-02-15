package main

import (
	"fmt"
	"log"
	"os"

	"github.com/23skdu/longbow-quarrel/internal/gguf"
	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: demo <gguf-file-path>")
		fmt.Println("This demo shows GGUF file information and tokenizer capabilities")
		os.Exit(1)
	}

	modelPath := os.Args[1]

	// Demonstrate GGUF reading (CPU-only)
	fmt.Println("=== Longbow Quarrel Linux Demo ===")
	fmt.Printf("Reading GGUF file: %s\n", modelPath)

	// Try to read GGUF metadata
	file, err := os.Open(modelPath)
	if err != nil {
		log.Fatalf("Failed to open GGUF file: %v", err)
	}
	defer func() { _ = file.Close() }()

	// Read GGUF file
	ggufFile, err := gguf.LoadFile(modelPath)
	if err != nil {
		log.Fatalf("Failed to load GGUF file: %v", err)
	}
	defer func() { _ = ggufFile.Close() }()

	tensorInfo := ggufFile.Tensors
	metadata := ggufFile.KV

	fmt.Printf("Found %d tensors\n", len(tensorInfo))
	fmt.Printf("Metadata keys: %d\n", len(metadata))

	// Show some key metadata
	for key, value := range metadata {
		switch key {
		case "general.architecture":
			fmt.Printf("Architecture: %s\n", value)
		case "llama.vocab_size":
			fmt.Printf("Vocab size: %s\n", value)
		case "llama.context_length":
			fmt.Printf("Context length: %s\n", value)
		case "general.file_type":
			fmt.Printf("File type: %s\n", value)
		}
	}

	// Demonstrate tokenizer (if available)
	fmt.Println("\n=== Tokenizer Demo ===")
	if tokenizerPath, ok := metadata["tokenizer.ggml.model"]; ok {
		if pathStr, ok := tokenizerPath.(string); ok {
			fmt.Printf("Tokenizer model: %s\n", pathStr)

			// Try to create a simple tokenizer
			t, err := tokenizer.New(pathStr)
			if err != nil {
				fmt.Printf("Warning: Could not initialize tokenizer: %v\n", err)
			} else {
				testText := "Hello, world!"
				tokens := t.Encode(testText)
				fmt.Printf("Test text: '%s'\n", testText)
				fmt.Printf("Tokens: %v\n", tokens)
				fmt.Printf("Token count: %d\n", len(tokens))
			}
		} else {
			fmt.Printf("Tokenizer path found but not a string: %T\n", tokenizerPath)
		}
	}

	fmt.Println("\n=== Note ===")
	fmt.Println("This is a CPU-only demo for Linux environments.")
	fmt.Println("For full Metal acceleration, use macOS with Apple Silicon.")
}
