//go:build darwin && metal

package main

import (
	"fmt"
	"os"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: dump_gguf <path>")
		return
	}
	f, err := gguf.LoadFile(os.Args[1])
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	for _, t := range f.Tensors {
		if t.Name == "output_norm.weight" {
			fmt.Printf("Tensor: %s (Type: %d, Offset: %d)\n", t.Name, t.Type, t.Offset)
			if len(t.Data) >= 16 {
				fmt.Printf("  Data[0:16]: %x\n", t.Data[:16])
			}
		}
	}
}
