//go:build darwin && metal

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

	tokens, ok := f.KV["tokenizer.ggml.tokens"].([]interface{})
	if !ok {
		fmt.Println("No tokens found")
		return
	}

	s := tokens[13290].(string)
	fmt.Printf("Token 13290: %q\n", s)
	for _, r := range s {
		fmt.Printf("  Char: %q, Code: %X\n", r, r)
	}
}
