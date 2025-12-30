package main

import (
	"fmt"
	"github.com/23skdu/longbow-quarrel/internal/engine"
	"sort"
)

func main() {
	modelPath := "/Users/rsd/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f"
	e, _ := engine.NewEngine(modelPath, false)
	
	prompt := "The capital of France is"
	e.Infer(prompt, 1, 0, 1.1)
}
