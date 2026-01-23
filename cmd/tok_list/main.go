//go:build darwin && metal

package main

import (
	"fmt"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

func main() {
	f, err := gguf.LoadFile("/Users/rsd/.ollama/models/blobs/sha256-f5117074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f")
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	tokens, _ := f.KV["tokenizer.ggml.tokens"].([]interface{})
	for i := 1170; i < 1200; i++ {
		fmt.Printf("Token %d: %q\n", i, tokens[i])
	}
}
