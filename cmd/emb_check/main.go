//go:build darwin && metal

package main

import (
	"flag"
	"fmt"

	"github.com/23skdu/longbow-quarrel/internal/engine"
)

func main() {
	modelPath := "/Users/rsd/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f"
	tokenID := flag.Int("token", -1, "Token ID to check")
	flag.Parse()

	e, err := engine.NewEngine(modelPath, false)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	if *tokenID != -1 {
		id := *tokenID
		tokens := e.Model.KV["tokenizer.ggml.tokens"].([]interface{})
		if id < 0 || id >= len(tokens) {
			fmt.Printf("Token ID %d out of range (0-%d)\n", id, len(tokens))
			return
		}

		word := tokens[id].(string)
		fmt.Printf("Token ID %d = %q\n", id, word)

		// Dump embedding
		data := e.Weights.TokenEmb.EmbeddingLookup(id, 1.0).ToHost()
		max := float32(0.0)
		for _, v := range data {
			abs := v
			if abs < 0 {
				abs = -abs
			}
			if abs > max {
				max = abs
			}
		}
		fmt.Printf("Token %d Embedding Max: %f\n", id, max)
	} else {
		fmt.Println("Please provide -token ID")
	}
}
