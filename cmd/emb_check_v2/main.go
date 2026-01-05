//go:build darwin && metal

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
		data := e.Weights.TokenEmb.EmbeddingLookup(id, 1.0).ToHost()
		max := float32(0.0)
		for _, v := range data {
			if v > max {
				max = v
			}
			if -v > max {
				max = -v
			}
		}
		fmt.Printf("ID %d Max Emb: %f\n", id, max)
	}
}
