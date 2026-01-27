//go:build darwin && metal

package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/engine"
	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
)

func main() {
	modelPath := flag.String("model", "", "Path to model file or model name (mistral)")
	prompt := flag.String("prompt", "", "Input prompt")
	maxLen := flag.Int("len", 100, "Maximum tokens to generate")
	temp := flag.Float64("temp", 0.7, "Temperature")
	topK := flag.Int("topk", 40, "Top-K")
	topP := flag.Float64("topp", 0.95, "Top-P")
	verbose := flag.Bool("v", false, "Verbose output")
	flag.Parse()

	if *modelPath == "" {
		// Try env var or default
		if os.Getenv("QUARREL_MODEL") != "" {
			*modelPath = os.Getenv("QUARREL_MODEL")
		} else {
			log.Fatal("--model required")
		}
	}

	resolvedPath := *modelPath
	if *modelPath == "mistral" {
		home, _ := os.UserHomeDir()
		resolvedPath = home + "/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f"
	}

	if *verbose {
		fmt.Printf("Loading model: %s\n", resolvedPath)
	}
	conf := config.Default()
	conf.KVCacheSize = 1024
	e, err := engine.NewEngine(resolvedPath, conf)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	defer e.Close()

	tok, err := tokenizer.New(resolvedPath)
	if err != nil {
		log.Fatalf("Failed to load tokenizer: %v", err)
	}

	finalPrompt := *prompt
	if finalPrompt == "" {
		finalPrompt = "The capital of France is"
	}

	tokens := tok.Encode(finalPrompt)
	// Add BOS
	tokens = append([]int{1}, tokens...)

	if *verbose {
		fmt.Printf("Prompt Tokens: %v\n", tokens)
		fmt.Printf("Generating (Temp=%.1f, TopK=%d, TopP=%.2f)...\n", *temp, *topK, *topP)
		fmt.Println("--------------------------------------------------")
	}

	// Stream output
	fmt.Print(finalPrompt)

	config := engine.SamplerConfig{
		Temperature: *temp,
		TopK:        *topK,
		TopP:        *topP,
		RepPenalty:  1.0,
	}

	// generated := 0 (removed)
	start := time.Now()

	// Since Infer generates N tokens, but we want to stream each one,
	// checking if Engine supports streaming callback would be ideal.
	// Current engine.Infer generates ALL tokens in a loop.
	// For this tool, we will just call Infer for the full length,
	// assuming it returns the slice of generated token IDs.
	// Wait, Engine.Infer logic:
	// Phase 1: Prefill (returns first generated token)
	// Phase 2: Generation loop (append to result)
	// So calling Infer(tokens, maxLen, config) will block until done.
	// To stream, we'd need to modify Infer or use a lower-level loop.
	// For now, let's just run it and print result at end, OR improve engine later.
	// Actually, let's just print the result at the end for this validaton step.
	// Streaming is a "nice to have".

	resultIDs, err := e.Infer(tokens, *maxLen, config)
	if err != nil {
		log.Fatalf("Inference failed: %v", err)
	}

	// Decode new tokens
	// resultIDs contains ONLY the generated tokens? Let's check Engine.Infer return.
	// engine.go: result = append(result, nextObj) -> starts empty.
	// Yes, result contains generated tokens.

	totalTime := time.Since(start)

	outputStr := tok.Decode(resultIDs)

	// If the prompt didn't end with space and generated doesn't start with space,
	// it might look glued. Tokenizer decode handles this usually.

	// Print JUST the new output (or full?)
	// We already printed prompt.
	// Mistral tokenizer adds dummy prefix space?
	// Let's just print outputStr.
	fmt.Print(outputStr)
	fmt.Println() // Newline at end

	if *verbose {
		fmt.Println("\n--------------------------------------------------")
		tps := float64(len(resultIDs)) / totalTime.Seconds()
		fmt.Printf("Generated %d tokens in %.2fs (%.2f tokens/s)\n", len(resultIDs), totalTime.Seconds(), tps)
	}
}
