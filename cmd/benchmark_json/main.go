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
	Throughput      float64 `json:"throughput_tokens_per_sec"`
	Output          string  `json:"output"`
	TotalDuration   float64 `json:"total_duration_seconds"`
	PrefillDuration float64 `json:"prefill_duration_seconds"`
	GenDuration     float64 `json:"generation_duration_seconds"`
	Tokens          int     `json:"tokens_generated"`
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

	var prefillDuration, genDuration time.Duration
	start := time.Now()
	var firstTokenTime time.Time

	// Run inference with timeout
	type inferResult struct {
		tokens []int
		err    error
	}
	resultChan := make(chan inferResult, 1)

	go func() {
		resIds, err := e.InferWithCallback(inputTokens, *numTokens, samplerConfig, func(token int) {
			if firstTokenTime.IsZero() {
				firstTokenTime = time.Now()
				prefillDuration = time.Since(start)
			}
		})
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

	totalDuration := time.Since(start)
	if !firstTokenTime.IsZero() {
		genDuration = time.Since(firstTokenTime)
	}

	decoded := tok.Decode(resultIDs)

	// TPS calculation for generation phase (excluding prefill)
	tps := 0.0
	if len(resultIDs) > 1 && genDuration > 0 {
		// We generated N tokens. 1 was during prefill loop completion, N-1 during pure generation loop.
		tps = float64(len(resultIDs)-1) / genDuration.Seconds()
	} else if totalDuration > 0 {
		tps = float64(len(resultIDs)) / totalDuration.Seconds()
	}

	// Debug: Log tokens
	log.Printf("DEBUG: Generated Token IDs: %v", resultIDs)

	if *outputFormat == "json" {
		out := Output{
			Throughput:      tps,
			Output:          decoded,
			TotalDuration:   totalDuration.Seconds(),
			PrefillDuration: prefillDuration.Seconds(),
			GenDuration:     genDuration.Seconds(),
			Tokens:          len(resultIDs),
		}
		jsonEnc := json.NewEncoder(os.Stdout)
		jsonEnc.Encode(out)
	} else {
		fmt.Printf("Output: %s\n", decoded)
		fmt.Printf("TPS (gen): %.2f\n", tps)
		fmt.Printf("Prefill: %.2fs, Gen: %.2fs\n", prefillDuration.Seconds(), genDuration.Seconds())
	}
}
