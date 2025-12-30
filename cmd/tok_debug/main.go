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
	
	fmt.Printf("General Name: %v\n", f.KV["general.name"])
	fmt.Printf("Context Length: %v\n", f.KV["llama.context_length"])
	fmt.Printf("Rope Theta: %v\n", f.KV["llama.rope.freq_base"])
	fmt.Printf("Rope Scaling: %v\n", f.KV["llama.rope.scaling.type"])
	
	tokens, ok := f.KV["tokenizer.ggml.tokens"].([]interface{})
	if !ok {
		fmt.Println("No tokens found")
		return
	}
	
	ids := []int{1, 2, 3, 4}
	for _, id := range ids {
		if id < len(tokens) {
			fmt.Printf("Token %d: %q\n", id, tokens[id])
		}
	}
}
