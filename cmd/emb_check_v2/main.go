package main

import (
	"fmt"
	"github.com/23skdu/longbow-quarrel/internal/engine"
)

func main() {
	modelPath := "/Users/rsd/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f"
	e, _ := engine.NewEngine(modelPath, false)
	
	ids := []int{1782, 6233}
	for _, id := range ids {
		emb := e.Weights.TokenEmb.EmbeddingLookup(id)
		max := emb.ScanMax(fmt.Sprintf("ID %d", id))
		fmt.Printf("ID %d Max Emb: %f\n", id, max)
	}
}
