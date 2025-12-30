package main

import (
	"fmt"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

func main() {
	f, err := gguf.LoadFile("/Users/rsd/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f")
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	
	merges, ok := f.KV["tokenizer.ggml.merges"].([]interface{})
	if !ok {
		fmt.Println("No merges found")
		return
	}
	
	for i := 0; i < 10; i++ {
		fmt.Printf("Merge %d: %q\n", i, merges[i])
	}
}
