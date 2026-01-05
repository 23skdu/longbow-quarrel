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
		fmt.Printf("[GGUF Dump] %s: Type=%v Shape=%v Offset=%d\n", t.Name, t.Type, t.Dimensions, t.Offset)
	}

	fmt.Printf("BOS Token ID: %v\n", f.KV["tokenizer.ggml.bos_token_id"])
	fmt.Printf("EOS Token ID: %v\n", f.KV["tokenizer.ggml.eos_token_id"])
	fmt.Printf("Unknown Token ID: %v\n", f.KV["tokenizer.ggml.unknown_token_id"])
	for k := range f.KV {
		fmt.Printf("KV: %s\n", k)
	}

	fmt.Println("\n[GGUF KV] Vocabulary (First 50):")
	if tokens, ok := f.KV["tokenizer.ggml.tokens"].([]interface{}); ok {
		for i := 0; i < 50 && i < len(tokens); i++ {
			fmt.Printf("%d: %v\n", i, tokens[i])
		}
	} else {
		fmt.Println("No vocabulary found in KV")
	}
}
