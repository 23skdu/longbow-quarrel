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
	
	words := []string{" Paris", " France", " is", " capital"}
	for _, w := range words {
		ids := tok.Encode(w)
		fmt.Printf("%q -> %v\n", w, ids)
	}
}
