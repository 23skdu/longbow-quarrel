//go:build darwin && metal

package main

import (
	"fmt"
	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
)

func main() {
	modelPath := "/Users/rsd/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f"
	tok, err := tokenizer.New(modelPath)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	prompt := "Hello"
	tokens := tok.Encode(prompt)
	fmt.Printf("Prompt: %q\n", prompt)
	fmt.Printf("Tokens: %v\n", tokens)
	for _, id := range tokens {
		fmt.Printf("  %d: %q\n", id, tok.Decode([]int{id}))
	}
}
