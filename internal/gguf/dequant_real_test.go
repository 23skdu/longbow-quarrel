package gguf

import (
	"fmt"
	"os"
	"testing"
)

func TestDequantToken1(t *testing.T) {
	path := "/Users/rsd/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f"
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	// Token 1 at 760000
	data := make([]byte, 144)
	_, err = f.ReadAt(data, 760000)
	if err != nil {
		t.Fatal(err)
	}

	fmt.Printf("Token 1 Raw: %x\n", data[:16])
	
	// Dequantize one block (256 weights)
	weights := DequantizeQ4K(data, 256)
	
	fmt.Printf("Token 1 First 10 floats: %v\n", weights[:10])
	
	var max float32
	for _, v := range weights {
		if v*v > max*max { max = v }
	}
	fmt.Printf("Token 1 Max Abs: %f\n", max)
}
