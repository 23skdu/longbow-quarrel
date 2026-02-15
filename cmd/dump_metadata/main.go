package main

import (
	"flag"
	"fmt"
	"log"
	"sort"

	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

func main() {
	modelPath := flag.String("model", "", "Path to GGUF model file")
	flag.Parse()

	if *modelPath == "" {
		log.Fatal("--model is required")
	}

	f, err := gguf.LoadFile(*modelPath)
	if err != nil {
		log.Fatalf("Failed to load GGUF: %v", err)
	}
	defer func() { _ = f.Close() }()

	fmt.Println("=== Metadata (KV) ===")
	keys := make([]string, 0, len(f.KV))
	for k := range f.KV {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, k := range keys {
		if k == "tokenizer.ggml.merges" || k == "tokenizer.ggml.tokens" || k == "tokenizer.ggml.scores" || k == "tokenizer.ggml.token_type" {
			fmt.Printf("%s: [skipped large data]\n", k)
			continue
		}
		fmt.Printf("%s: %v\n", k, f.KV[k])
	}

	fmt.Println("\n=== Tensors ===")
	for _, t := range f.Tensors {
		fmt.Printf("%s [%v] %s\n", t.Name, t.Dimensions, t.Type)
	}
}
