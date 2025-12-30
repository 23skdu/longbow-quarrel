package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
)

func main() {
	modelPath := flag.String("model", "mistral", "Model path")
	flag.Parse()

	mPath := *modelPath
	if mPath == "mistral" {
		home, _ := os.UserHomeDir()
		mPath = home + "/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f"
	}

	tok, err := tokenizer.New(mPath)
	if err != nil {
		log.Fatal(err)
	}

	// If args provided, decode them
	args := flag.Args()
	if len(args) > 0 {
		for _, arg := range args {
			var id int
			fmt.Sscanf(arg, "%d", &id)
			if id > 0 || arg == "0" {
				text := tok.Decode([]int{id})
				fmt.Printf("ID %d -> %q\n", id, text)
			} else {
				encoded := tok.Encode(arg)
				fmt.Printf("Text %q -> %v\n", arg, encoded)
			}
		}
		return
	}

	ids := []int{25076, 19072, 4684, 1046, 1}
	for _, id := range ids {
		text := tok.Decode([]int{id})
		fmt.Printf("ID %d -> %q\n", id, text)
	}

	targets := []string{"Paris", " Paris", "aurus"}
	for _, t := range targets {
		encoded := tok.Encode(t)
		fmt.Printf("Text %q -> %v\n", t, encoded)
	}
}
