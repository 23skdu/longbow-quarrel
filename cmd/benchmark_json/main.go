//go:build darwin && metal

package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/engine"
	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
)

type Output struct {
	Throughput float64 `json:"throughput_tokens_per_sec"`
	Output     string  `json:"output"`
	Duration   float64 `json:"duration_seconds"`
	Tokens     int     `json:"tokens_generated"`
}

func main() {
	modelPath := flag.String("model", "", "Path to GGUF model file")
	prompt := flag.String("prompt", "Hello world", "Prompt to generate from")
	numTokens := flag.Int("tokens", 20, "Number of tokens to generate")
	outputFormat := flag.String("output", "text", "Output format (text or json)")

	// Ignored flags for compatibility if needed, though we control the script
	// Additional flags
	timeoutSec := flag.Int("timeout", 60, "Timeout in seconds")
	flag.Parse()

	if *modelPath == "" {
		log.Fatal("--model is required")
	}

	// Initialize Tokenizer
	tok, err := tokenizer.New(*modelPath)
	if err != nil {
		log.Fatalf("Failed to initialize tokenizer: %v", err)
	}

	// Initialize Engine
	engineConfig := config.Default()
	// Set reasonable defaults
	engineConfig.KVCacheSize = 256 // Generic size

	e, err := engine.NewEngine(*modelPath, engineConfig)
	if err != nil {
		log.Fatalf("Failed to initialize engine: %v", err)
	}
	defer e.Close()

	// Tokenize
	inputTokens := tok.Encode(*prompt)
	// Add BOS if not present? Usually good practice for Llama/Mistral
	inputTokens = append([]int{1}, inputTokens...)

	samplerConfig := engine.SamplerConfig{
		Temperature: 0.0, // Greedy for coherence check
	}

	start := time.Now()

	// Run inference with timeout
	type inferResult struct {
		tokens []int
		err    error
	}
	resultChan := make(chan inferResult, 1)

	go func() {
		resIds, err := e.Infer(inputTokens, *numTokens, samplerConfig)
		resultChan <- inferResult{tokens: resIds, err: err}
	}()

	var resultIDs []int

	select {
	case res := <-resultChan:
		if res.err != nil {
			log.Fatalf("Inference failed: %v", res.err)
		}
		resultIDs = res.tokens
	case <-time.After(time.Duration(*timeoutSec) * time.Second):
		log.Fatalf("Inference timed out after %d seconds", *timeoutSec)
	}

	duration := time.Since(start)

	decoded := tok.Decode(resultIDs)
	tps := float64(len(resultIDs)) / duration.Seconds()

	// Debug: Log tokens
	log.Printf("DEBUG: Generated Token IDs: %v", resultIDs)

	if *outputFormat == "json" {
		out := Output{
			Throughput: tps,
			Output:     decoded,
			Duration:   duration.Seconds(),
			Tokens:     len(resultIDs),
		}
		jsonEnc := json.NewEncoder(os.Stdout)
		jsonEnc.Encode(out)
	} else {
		fmt.Printf("Output: %s\n", decoded)
		fmt.Printf("TPS: %.2f\n", tps)
	}
}
