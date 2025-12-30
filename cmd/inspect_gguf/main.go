package main

import (
	"fmt"
	"log"
	"flag"

	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

func main() {
	modelPath := flag.String("model", "", "Path to GGUF model file")
	flag.Parse()

	if *modelPath == "" {
        // Fallback or error
		*modelPath = "/Users/rsd/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f"
        fmt.Printf("Using default path: %s\n", *modelPath)
	}

	fmt.Printf("Loading model header: %s\n", *modelPath)
	f, err := gguf.LoadFile(*modelPath)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	fmt.Println("\n=== GGUF Header Keys ===")
    fmt.Printf("Analysis: Looking for scaling factors, versioning, or custom attributes\n")
    fmt.Println("----------------------------------------------------------------")
	for k, v := range f.KV {
		fmt.Printf("Key: %-30s | Value: %v\n", k, v)
	}
    fmt.Println("----------------------------------------------------------------")
    
    // Also check for any weird tensors like layer_scale
    fmt.Println("\n=== Tensor Search: 'scale' ===")
    foundScale := false
    for _, t := range f.Tensors {
        if contains(t.Name, "scale") {
            fmt.Printf("Tensor: %s (Type: %d, Dims: %v)\n", t.Name, t.Type, t.Dimensions)
            foundScale = true
        }
    }
    if !foundScale {
        fmt.Println("No tensors with 'scale' in name found.")
    }
}

func contains(s, substr string) bool {
    // simple check
    for i := 0; i < len(s)-len(substr)+1; i++ {
        if s[i:i+len(substr)] == substr {
            return true
        }
    }
    return false
}
