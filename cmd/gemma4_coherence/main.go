//go:build darwin && metal

package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/engine"
	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
)

type TestResult struct {
	Prompt        string  `json:"prompt"`
	QuarrelText   string  `json:"quarrel_text"`
	QuarrelTokens int     `json:"quarrel_tokens"`
	QuarrelTPS    float64 `json:"quarrel_tps"`
	OllamaText    string  `json:"ollama_text"`
	OllamaTokens  int     `json:"ollama_tokens"`
	OllamaTPS     float64 `json:"ollama_tps"`
	Match         bool    `json:"match"`
}

type inferenceResult struct {
	text   string
	tokens int
	tps    float64
}

func main() {
	modelPath := flag.String("model", "", "Path to model GGUF file")
	ollamaModel := flag.String("ollama-model", "gemma4:e4b", "Ollama model name")
	tokens := flag.Int("tokens", 20, "Tokens to generate")
	output := flag.String("output", "coherence_results.json", "Output JSON file")
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintf(os.Stderr, "--model required\n")
		os.Exit(1)
	}

	conf := config.Default()
	conf.KVCacheSize = 8192 // Need more for long battery
	conf.DebugEmbedding = false
	conf.DebugAttention = false
	conf.DebugFFN = false
	conf.DebugLayerOutput = false
	conf.DebugLogits = false
	conf.DebugMemory = false

	fmt.Printf("Loading engine: %s...\n", *modelPath)
	e, err := engine.NewEngine(*modelPath, conf)
	if err != nil {
		log.Fatalf("Failed to load engine: %v", err)
	}
	defer e.Close()

	tok, err := tokenizer.New(*modelPath)
	if err != nil {
		log.Fatalf("Failed to load tokenizer: %v", err)
	}

	prompts := []string{
		"Explain quantum entanglement in simple terms.",
		"Write a Python function to calculate Fibonacci numbers.",
		"What are the three laws of thermodynamics?",
		"Translate 'Where is the library?' into French, Spanish, and German.",
		"Who wrote the play 'Hamlet'?",
		"Explain the difference between a list and a tuple in Python.",
		"Summarize the plot of the movie 'Inception'.",
		"How do photosynthesis and respiration differ?",
		"Write a short poem about a rainy day in a city.",
		"What is the approximate distance from the Earth to the Moon?",
		"List five major cities in Japan.",
		"Explain the concept of 'opportunity cost' in economics.",
		"How does a blockchain work?",
		"Write a SQL query to find all users over age 30 from a 'users' table.",
		"What are the primary ingredients in a classic Margherita pizza?",
		"Describe the process of brewing a cup of espresso.",
		"Who is known as the 'Father of Modern Physics'?",
		"Write a function in JavaScript to reverse a string.",
		"What is the chemical formula for table salt?",
		"Summarize the main theme of the book '1984' by George Orwell.",
		"How many continents are there on Earth?",
		"Explain the difference between 'HTTP' and 'HTTPS'.",
		"Write a short story (3 sentences) about a robot that learns to paint.",
		"What is the value of the mathematical constant 'pi' to 5 decimal places?",
		"List the colors of a rainbow in order.",
		"How do you declare a variable in C++?",
		"What is the capital city of Australia?",
		"Explain the 'Turing Test'.",
		"Write a CSS rule to make all h1 elements blue.",
		"Who painted the 'Mona Lisa'?",
	}

	results := make([]TestResult, 0, len(prompts))

	for i, prompt := range prompts {
		fmt.Printf("[%d/%d] Testing: %q\n", i+1, len(prompts), prompt)

		qResult := testQuarrel(e, tok, prompt, *tokens)
		fmt.Printf("  Quarrel: %s (%.1f t/s)\n", qResult.text, qResult.tps)

		oResult := testOllama(*ollamaModel, prompt, *tokens)
		fmt.Printf("  Ollama:  %s (%.1f t/s)\n", oResult.text, oResult.tps)

		match := strings.EqualFold(qResult.text, oResult.text)
		fmt.Printf("  Match: %v\n\n", match)

		results = append(results, TestResult{
			Prompt:        prompt,
			QuarrelText:   qResult.text,
			QuarrelTokens: qResult.tokens,
			QuarrelTPS:    qResult.tps,
			OllamaText:    oResult.text,
			OllamaTokens:  oResult.tokens,
			OllamaTPS:     oResult.tps,
			Match:         match,
		})
	}

	writeJSON(results, *output)
	printSummary(results)
}

func testQuarrel(e engine.Engine, tok *tokenizer.Tokenizer, prompt string, maxTokens int) inferenceResult {
	inputTokens := tok.Encode(prompt)
	sampler := engine.SamplerConfig{
		Temperature: 0.7,
		TopK:        40,
		TopP:        0.95,
	}

	start := time.Now()
	result, err := e.Infer(inputTokens, maxTokens, sampler)
	elapsed := time.Since(start)

	if err != nil {
		log.Fatalf("Inference failed: %v", err)
	}

	generated := tok.Decode(result)
	tokens := len(result)
	tps := float64(tokens) / elapsed.Seconds()

	return inferenceResult{
		text:   strings.TrimSpace(generated),
		tokens: tokens,
		tps:    tps,
	}
}

func testOllama(model, prompt string, maxTokens int) inferenceResult {
	cmd := exec.Command("ollama", "run", model, prompt)
	cmd.Stdin = strings.NewReader(prompt + "\n")
	var out strings.Builder
	cmd.Stdout = &out
	cmd.Stderr = os.Stderr

	start := time.Now()
	err := cmd.Run()
	elapsed := time.Since(start)

	if err != nil {
		log.Printf("Ollama error: %v", err)
		return inferenceResult{text: "", tokens: 0, tps: 0}
	}

	lines := strings.Split(out.String(), "\n")
	var text string
	for _, line := range lines {
		if strings.TrimSpace(line) != "" {
			text = strings.TrimSpace(line)
		}
	}

	tokens := strings.Count(text, " ") + 1
	if tokens < 1 {
		tokens = maxTokens
	}
	tps := float64(tokens) / elapsed.Seconds()

	return inferenceResult{
		text:   text,
		tokens: tokens,
		tps:    tps,
	}
}

func writeJSON(results []TestResult, path string) {
	data, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		log.Printf("Failed to marshal JSON: %v", err)
		return
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		log.Printf("Failed to write JSON: %v", err)
		return
	}
	fmt.Printf("Results written to: %s\n", path)
}

func printSummary(results []TestResult) {
	fmt.Printf("\n=== SUMMARY ===\n")

	var qTPS, oTPS float64
	matches := 0

	for _, r := range results {
		qTPS += r.QuarrelTPS
		oTPS += r.OllamaTPS
		if r.Match {
			matches++
		}
	}

	fmt.Printf("Quarrel avg TPS: %.1f\n", qTPS/float64(len(results)))
	fmt.Printf("Ollama avg TPS:  %.1f\n", oTPS/float64(len(results)))
	fmt.Printf("Exact matches:  %d/%d (%.1f%%)\n", matches, len(results), float64(matches)/float64(len(results))*100)

	var qlen, olen int
	for _, r := range results {
		qlen += len(r.QuarrelText)
		olen += len(r.OllamaText)
	}
	avgLenDiff := math.Abs(float64(qlen) - float64(olen))
	fmt.Printf("Avg length diff: %.0f chars\n", avgLenDiff/float64(len(results)))
}
