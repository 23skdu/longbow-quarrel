package main

import (
	"fmt"
	"log"
	"sort"

	"github.com/23skdu/longbow-quarrel/internal/engine"
	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
)

func main() {
	modelPath := "/Users/rsd/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f"
	
	fmt.Println("Loading model...")
	e, err := engine.NewEngine(modelPath)
	if err != nil {
		log.Fatal(err)
	}
	defer e.Close()
	
	// Tokenize prompt
	tok, err := tokenizer.NewTokenizer(modelPath)
	if err != nil {
		log.Fatal(err)
	}
	
	prompt := "The capital of France is"
	tokens := tok.Encode(prompt, true, false)
	fmt.Printf("Prompt: %q\n", prompt)
	fmt.Printf("Tokens: %v\n", tokens)
	
	// Run inference for 1 token
	config := engine.SamplerConfig{
		Temperature: 0.0,
		TopP:        0.95,
		TopK:        40,
	}
	
	results := e.Infer(tokens, 1, config)
	
	// Get logits from last position
	logits := e.GetLastLogits()
	
	// Find token 6233 ("Paris")
	parisToken := 6233
	parisLogit := logits[parisToken]
	
	// Get top-100 tokens
	type tokenLogit struct {
		id    int
		logit float32
	}
	
	all := make([]tokenLogit, len(logits))
	for i, l := range logits {
		all[i] = tokenLogit{i, l}
	}
	
	sort.Slice(all, func(i, j int) bool {
		return all[i].logit > all[j].logit
	})
	
	// Check if Paris is in top-100
	parisRank := -1
	for i := 0; i < len(all); i++ {
		if all[i].id == parisToken {
			parisRank = i + 1
			break
		}
	}
	
	fmt.Printf("\n=== LOGIT ANALYSIS ===\n")
	fmt.Printf("Token 6233 (Paris) logit: %.6f\n", parisLogit)
	if parisRank > 0 {
		fmt.Printf("Token 6233 rank: %d / %d\n", parisRank, len(logits))
	} else {
		fmt.Printf("Token 6233 NOT in top-%d\n", len(all))
	}
	
	fmt.Printf("\nTop-10 tokens:\n")
	for i := 0; i < 10 && i < len(all); i++ {
		tokenStr := tok.Decode([]int{all[i].id})
		fmt.Printf("  %2d. Token %5d (logit=%7.3f): %q\n", 
			i+1, all[i].id, all[i].logit, tokenStr)
	}
	
	if parisRank > 10 && parisRank <= 100 {
		fmt.Printf("\nToken 6233 is at rank %d:\n", parisRank)
		tokenStr := tok.Decode([]int{parisToken})
		fmt.Printf("  Token %5d (logit=%7.3f): %q\n", parisToken, parisLogit, tokenStr)
	}
	
	fmt.Printf("\nGenerated token: %d\n", results[0])
}
