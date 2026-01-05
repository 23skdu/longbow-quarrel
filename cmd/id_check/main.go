//go:build darwin && metal

package main

import (
	"fmt"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

func main() {
	f, _ := gguf.LoadFile("/Users/rsd/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f")
	tokens, _ := f.KV["tokenizer.ggml.tokens"].([]interface{})
	ids := []int{1, 1183, 1782, 6233, 31674}
	for _, id := range ids {
		fmt.Printf("ID %d: %q\n", id, tokens[id])
	}
}
