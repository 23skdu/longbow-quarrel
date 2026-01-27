package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/engine"
)

func main() {
	home, _ := os.UserHomeDir()
	modelPath := filepath.Join(home, ".ollama/models/blobs/sha256-a70437c41b3b0b768c48737e15f8160c90f13dc963f5226aabb3a160f708d1ce")

	fmt.Printf("Loading model: %s\n", modelPath)

	cfg := config.Default()
	e, err := engine.NewEngine(modelPath, cfg)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	defer e.Close()
	fmt.Println("Model loaded successfully.")
}
