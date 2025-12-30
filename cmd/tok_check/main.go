package main

import (
	"fmt"
	"os"
	"strconv"
	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
)

func main() {
	modelPath := "/Users/rsd/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f"
	tok, err := tokenizer.New(modelPath)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	
	if len(os.Args) > 1 {
		fmt.Printf("Decoding IDs from arguments:\n")
		for _, arg := range os.Args[1:] {
			id, err := strconv.Atoi(arg)
			if err != nil {
				continue
			}
			decoded := tok.Decode([]int{id})
			fmt.Printf("  %d -> %q\n", id, decoded)
		}
		return
	}

	words := []string{"Hello", " Hello", "world"}
	for _, w := range words {
		ids := tok.Encode(w)
		fmt.Printf("%q -> %v\n", w, ids)
		for _, id := range ids {
			decoded := tok.Decode([]int{id})
			fmt.Printf("  %d -> %q\n", id, decoded)
		}
	}
}
