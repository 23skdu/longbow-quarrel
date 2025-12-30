package main

import (
	"fmt"
	"github.com/23skdu/longbow-quarrel/internal/engine"
)

func main() {
	modelPath := "/Users/rsd/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f"
	e, err := engine.NewEngine(modelPath, false)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	
	words := []string{"The", "Paris"}
	for _, w := range words {
		tokens := e.Model.KV["tokenizer.ggml.tokens"].([]interface{})
		var id int = -1
		for i, v := range tokens {
			if v.(string) == w || v.(string) == "▁"+w {
				id = i
				break
			}
		}
		if id != -1 {
			emb := e.Weights.TokenEmb.EmbeddingLookup(id)
			max := emb.ScanMax(fmt.Sprintf("Emb(%s)", w))
			fmt.Printf("Token %q (ID %d) Max Emb: %f\n", w, id, max)
		}
	}
}
