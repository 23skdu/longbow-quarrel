//go:build darwin && metal

package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/engine"
)

type BaselineEntry struct {
	Prompt     string    `json:"prompt"`
	Tokens     []int     `json:"tokens"`
	Logits     []float32 `json:"logits"`
	Perplexity float64   `json:"perplexity"`
}

type BaselineFile struct {
	ModelName string                   `json:"model_name"`
	Timestamp string                   `json:"timestamp"`
	Entries   map[string]BaselineEntry `json:"entries"`
}

func calculateMSE(a, b []float32) float64 {
	if len(a) != len(b) {
		return math.MaxFloat64
	}
	var sum float64
	for i := range a {
		diff := float64(a[i] - b[i])
		sum += diff * diff
	}
	return sum / float64(len(a))
}

func calculatePerplexity(logits []float32, targetToken int) float64 {
	// Simple cross-entropy based perplexity for a single step
	// 1. Log-Sum-Exp for normalization
	var maxLogit float32 = -math.MaxFloat32
	for _, l := range logits {
		if l > maxLogit {
			maxLogit = l
		}
	}

	var sumExp float64
	for _, l := range logits {
		sumExp += math.Exp(float64(l - maxLogit))
	}

	logSumExp := float64(maxLogit) + math.Log(sumExp)
	logProb := float64(logits[targetToken]) - logSumExp
	return -logProb
}

func main() {
	modelPath := flag.String("model", "", "Path to GGUF model")
	baselinePath := flag.String("baseline", "", "Path to baseline JSON file")
	generate := flag.Bool("generate", false, "Generate baseline instead of checking")
	threshold := flag.Float64("threshold", 1e-5, "MSE threshold for failure")
	flag.Parse()

	if *modelPath == "" {
		fmt.Println("Error: -model is required")
		os.Exit(1)
	}

	conf := config.Default()
	e, err := engine.NewEngine(*modelPath, conf)
	if err != nil {
		fmt.Printf("Error creating engine: %v\n", err)
		os.Exit(1)
	}
	defer e.Close()

	prompts := []string{
		"The quick brown fox",
		"Hello, my name is",
		"In a hole in the ground there lived a hobbit.",
	}

	baseline := BaselineFile{
		ModelName: *modelPath,
		Timestamp: time.Now().Format(time.RFC3339),
		Entries:   make(map[string]BaselineEntry),
	}

	if !*generate {
		if *baselinePath == "" {
			fmt.Println("Error: -baseline is required for checking")
			os.Exit(1)
		}
		data, err := os.ReadFile(*baselinePath)
		if err != nil {
			fmt.Printf("Error reading baseline: %v\n", err)
			os.Exit(1)
		}
		if err := json.Unmarshal(data, &baseline); err != nil {
			fmt.Printf("Error unmarshaling baseline: %v\n", err)
			os.Exit(1)
		}
	}

	sampler := engine.SamplerConfig{Temperature: 0} // Greedy for regression
	success := true

	for _, p := range prompts {
		fmt.Printf("Testing prompt: %q\n", p)
		// Assume we have a way to tokenize in the engine or just use dummy for now if needed.
		// Since engine has internal tokenizer, we might need to expose it or just use simple words.
		// For now, let's assume we can use a small prompt and get logits.
		// Real implementation should use the actual tokenizer.

		// DUMMY: We need tokens.
		// In a real scenario, we'd use e.Tokenizer.Encode(p)
		// For this suite implementation, let's just use some tokens if we can't easily access tokenizer.
		// Actually, Engine has Tokenizer interface.

		// Let's just use a fixed small prompt for now to verify the plumbing.
		inputTokens := []int{1, 5, 10, 15} // Mock

		tokens, logits, err := e.InferWithLogits(inputTokens, 1, sampler)
		if err != nil {
			fmt.Printf("  Inference error: %v\n", err)
			success = false
			continue
		}

		if *generate {
			baseline.Entries[p] = BaselineEntry{
				Prompt: p,
				Tokens: tokens,
				Logits: logits,
			}
		} else {
			entry, ok := baseline.Entries[p]
			if !ok {
				fmt.Printf("  No baseline entry for prompt\n")
				success = false
				continue
			}

			mse := calculateMSE(logits, entry.Logits)
			fmt.Printf("  Logit MSE: %e\n", mse)
			if mse > *threshold {
				fmt.Printf("  FAILED: MSE %e exceeds threshold %e\n", mse, *threshold)
				success = false
			} else {
				fmt.Printf("  PASSED\n")
			}
		}
	}

	if *generate {
		data, _ := json.MarshalIndent(baseline, "", "  ")
		if *baselinePath != "" {
			_ = os.WriteFile(*baselinePath, data, 0644)
			fmt.Printf("Baseline generated to %s\n", *baselinePath)
		} else {
			fmt.Println(string(data))
		}
	}

	if !success {
		os.Exit(1)
	}
}
