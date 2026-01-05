//go:build darwin && metal

package main

import (
	"fmt"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
	"strings"
)

func main() {
	f, err := gguf.LoadFile("/Users/rsd/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f")
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	
	tokens, ok := f.KV["tokenizer.ggml.tokens"].([]interface{})
	if !ok {
		fmt.Println("No tokens found")
		return
	}
	
	for i, v := range tokens {
		s := v.(string)
		if strings.Contains(s, "Paris") {
			fmt.Printf("Found %q at ID %d\n", s, i)
		}
	}
}
