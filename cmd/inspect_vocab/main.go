package main

import (
	"flag"
	"fmt"
	"log"

	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
)

func main() {
	modelPath := flag.String("model", "", "Path to GGUF model")
	prompt := flag.String("prompt", "Hello World", "Prompt to tokenize")
	flag.Parse()

	if *modelPath == "" {
		log.Fatal("Please provide -model")
	}

	// Load Tokenizer
	tok, err := tokenizer.New(*modelPath)
	if err != nil {
		log.Fatalf("Failed to load tokenizer: %v", err)
	}

	fmt.Printf("Loaded Tokenizer. Vocab Size: %d, Merges: %d\n", len(tok.Tokens), len(tok.Merges))

	// Dump first 50 tokens
	fmt.Println("--- First 50 Tokens ---")
	for i := 0; i < 50 && i < len(tok.Tokens); i++ {
		fmt.Printf("[%d]: %q\n", i, tok.Tokens[i])
	}

	// Check Special Tokens
	special := []string{"<|im_start|>", "<|im_end|>", "<s>", "</s>", "<unk>"}
	fmt.Println("\n--- Special Tokens ---")
	for _, s := range special {
		id, ok := tok.Vocab[s]
		if ok {
			fmt.Printf("'%s': %d\n", s, id)
		} else {
			fmt.Printf("'%s': NOT FOUND\n", s)
		}
	}

	// Test Tokenization
	fmt.Printf("\n--- Tokenization Test ---\n")
	fmt.Printf("Input: %q\n", *prompt)
	ids := tok.Encode(*prompt)
	fmt.Printf("IDs: %v\n", ids)
	
	decoded := tok.Decode(ids)
	fmt.Printf("Decoded: %q\n", decoded)

	// Dump components to see how it was split
	fmt.Printf("Split: ")
	for _, id := range ids {
		fmt.Printf("'%s' ", tok.Tokens[id])
	}
	fmt.Println()
}
