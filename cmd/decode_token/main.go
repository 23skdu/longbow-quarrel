package main

import (
	"fmt"
	"log"
	
	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
)

func main() {
	modelPath := "/Users/rsd/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f"
	tok, err := tokenizer.New(modelPath)
	if err != nil {
		log.Fatalf("Failed to load tokenizer: %v", err)
	}
	
	id := 31980
	text := tok.Decode([]int{id})
	fmt.Printf("Token %d: %q\n", id, text)
}
