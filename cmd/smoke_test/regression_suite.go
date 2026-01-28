//go:build darwin && metal

package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
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

func main() {
	modelPath := flag.String("model", "", "Path to GGUF model")
	baselinePath := flag.String("baseline", "", "Path to baseline JSON file")
	generate := flag.Bool("generate", false, "Generate baseline instead of checking")
	threshold := flag.Float64("threshold", 1e-5, "MSE threshold for failure")
	pplThreshold := flag.Float64("ppl-threshold", 0.05, "Perplexity difference threshold")
	flag.Parse()

	success := true
	if *modelPath == "" {
		fmt.Println("Running default regression suite for key models...")
		models := []struct {
			name string
			path string
		}{
			// {"nemotron-3-nano", "/Users/rsd/.ollama/models/blobs/sha256-a70437c41b3b0b768c48737e15f8160c90f13dc963f5226aabb3a160f708d1ce"},
			{"gpt-oss", "/Users/rsd/.ollama/models/blobs/sha256-e7b273f9636059a689e3ddcab3716e4f65abe0143ac978e46673ad0e52d09efb"},
			{"mistral", "/Users/rsd/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f"},
			{"granite4", "/Users/rsd/.ollama/models/blobs/sha256-5c7ac4aead1bcf4c8da9534ed72cc632d005aeed6547f1e8662ccdfae688364e"},
			{"tinyllama", "/Users/rsd/.ollama/models/blobs/sha256-2af3b81862c6be03c769683af18efdadb2c33f60ff32ab6f83e42c043d6c7816"},
		}

		for _, m := range models {
			fmt.Printf("\n=== Testing Model: %s ===\n", m.name)
			if _, err := os.Stat(m.path); os.IsNotExist(err) {
				fmt.Printf("Skipping %s: file not found at %s\n", m.name, m.path)
				continue
			}
			// Recursive call or separate function? Let's use a function.
			if !runModelTest(m.path, *baselinePath, *generate, *threshold, *pplThreshold) {
				success = false
			}
		}
		if !success {
			os.Exit(1)
		}
		return
	}

	if !runModelTest(*modelPath, *baselinePath, *generate, *threshold, *pplThreshold) {
		os.Exit(1)
	}
}

func runModelTest(modelPath, baselinePath string, generate bool, threshold, pplThreshold float64) bool {
	conf := config.Default()
	fmt.Printf("Memory before NewEngine: %d bytes\n", device.AllocatedBytes())
	e, err := engine.NewEngine(modelPath, conf)
	if err != nil {
		fmt.Printf("Error creating engine: %v\n", err)
		return false
	}
	defer func() {
		e.Close()
		fmt.Printf("Memory after Close: %d bytes\n", device.AllocatedBytes())
	}()

	if e.Tokenizer == nil {
		fmt.Printf("Error: Engine has no tokenizer\n")
		return false
	}

	prompts := []string{
		"The quick brown fox",
		"Hello, my name is",
		"In a hole in the ground there lived a hobbit.",
	}

	baseline := BaselineFile{
		ModelName: modelPath,
		Timestamp: time.Now().Format(time.RFC3339),
		Entries:   make(map[string]BaselineEntry),
	}

	if baselinePath == "" {
		// Try to find a default baseline file based on model name
		baselinePath = fmt.Sprintf("cmd/smoke_test/baselines/%s.json", filepath.Base(modelPath))
	}

	if !generate {
		data, err := os.ReadFile(baselinePath)
		if err != nil {
			fmt.Printf("Error reading baseline: %v\n", err)
			return false
		}
		if err := json.Unmarshal(data, &baseline); err != nil {
			fmt.Printf("Error unmarshaling baseline: %v\n", err)
			return false
		}
	}

	sampler := engine.SamplerConfig{Temperature: 0} // Greedy for regression
	success := true

	for _, p := range prompts {
		fmt.Printf("Testing prompt: %q\n", p)
		inputTokens := e.Tokenizer.Encode(p)

		tokens, logits, err := e.InferWithLogits(inputTokens, 1, sampler)
		if err != nil {
			fmt.Printf("  Inference error: %v\n", err)
			success = false
			continue
		}

		if generate {
			baseline.Entries[p] = BaselineEntry{
				Prompt:     p,
				Tokens:     tokens,
				Logits:     logits,
				Perplexity: calculatePerplexity(logits, tokens[0]),
			}
		} else {
			entry, ok := baseline.Entries[p]
			if !ok {
				fmt.Printf("  No baseline entry for prompt\n")
				success = false
				continue
			}

			mse := calculateMSE(logits, entry.Logits)
			ppl := calculatePerplexity(logits, tokens[0])
			pplDiff := math.Abs(ppl - entry.Perplexity)

			fmt.Printf("  Logit MSE: %e\n", mse)
			fmt.Printf("  Perplexity: %.4f (diff: %.4f)\n", ppl, pplDiff)

			if mse > threshold {
				fmt.Printf("  FAILED: MSE %e exceeds threshold %e\n", mse, threshold)
				success = false
			} else if pplDiff > pplThreshold {
				fmt.Printf("  FAILED: Perplexity diff %.4f exceeds threshold %.4f\n", pplDiff, pplThreshold)
				success = false
			} else {
				fmt.Printf("  PASSED\n")
			}
		}
	}

	if generate {
		data, _ := json.MarshalIndent(baseline, "", "  ")
		if baselinePath != "" {
			_ = os.WriteFile(baselinePath, data, 0644)
			fmt.Printf("Baseline generated to %s\n", baselinePath)
		} else {
			fmt.Println(string(data))
		}
	}

	return success
}
