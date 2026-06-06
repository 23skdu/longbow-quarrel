package main

import (
	"flag"
	"fmt"
	"os"
	"runtime"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/engine"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
)

func main() {
	modelPath := flag.String("model", "", "Path to GGUF model file")
	prompt := flag.String("prompt", "The capital of France is", "Prompt for text generation")
	maxTokens := flag.Int("n", 50, "Maximum tokens to generate")
	temp := flag.Float64("temp", 0.7, "Temperature for sampling")
	topK := flag.Int("topk", 40, "Top-K sampling")
	maxMemMB := flag.Int64("max-memory", 0, "Max memory in MB (0 = no limit)")
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "Error: --model flag is required")
		os.Exit(1)
	}

	fmt.Printf("=== Longbow-Quarrel Cross-Platform CLI ===\n")
	fmt.Printf("Go Version: %s\n", runtime.Version())
	fmt.Printf("NumCPU: %d\n", runtime.NumCPU())
	fmt.Printf("Model: %s\n", *modelPath)
	fmt.Printf("Prompt: %s\n", *prompt)
	fmt.Printf("Temp: %.2f, TopK: %d, MaxTokens: %d\n", *temp, *topK, *maxTokens)
	fmt.Println()

	f, err := gguf.LoadFile(*modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load GGUF: %v\n", err)
		os.Exit(1)
	}
	defer func() { _ = f.Close() }()

	fmt.Printf("GGUF Version: %d\n", f.Header.Version)
	fmt.Printf("Tensors: %d\n", len(f.Tensors))

	arch, ok := f.KV["general.architecture"]
	if ok {
		fmt.Printf("Architecture: %v\n", arch)
	}

	vocabSize, ok := f.KV["llama.vocab_size"]
	if ok {
		fmt.Printf("Vocab Size: %v\n", vocabSize)
	}

	layers, ok := f.KV["llama.block_count"]
	if ok {
		fmt.Printf("Layers: %v\n", layers)
	}

	dim, ok := f.KV["llama.embedding_length"]
	if ok {
		fmt.Printf("Embedding Dim: %v\n", dim)
	}

	fmt.Println()
	fmt.Println("Loading tokenizer...")
	tok, err := tokenizer.NewFromGGUF(f)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load tokenizer: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Tokenizer loaded: %d tokens\n", len(tok.Tokens))

	inputTokens := tok.Encode(*prompt)
	if len(inputTokens) == 0 {
		fmt.Fprintln(os.Stderr, "Failed to encode prompt")
		os.Exit(1)
	}
	fmt.Printf("Encoded %d tokens\n", len(inputTokens))

	fmt.Println()
	fmt.Printf("Input: %s\n", *prompt)

	startTime := time.Now()

	cfg := config.Default()
	cfg.MaxMemoryMB = *maxMemMB
	if cfg.MaxMemoryMB > 0 {
		fmt.Printf("Memory limit: %d MB\n", cfg.MaxMemoryMB)
	}

	samplerCfg := engine.SamplerConfig{
		Temperature: 0.8,
		TopK:        40,
		RepPenalty:  1.5,
	}

	e, err := engine.NewEngine(*modelPath, cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create engine: %v\n", err)
		os.Exit(1)
	}
	defer e.Close()

	resultTokens, err := e.Infer(inputTokens, *maxTokens, samplerCfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Inference failed: %v\n", err)
		os.Exit(1)
	}

	elapsed := time.Since(startTime)
	fmt.Printf("Generated: ")
	for _, tokenID := range resultTokens {
		fmt.Printf("%d:%s ", tokenID, tok.Decode([]int{tokenID}))
	}
	fmt.Println()
	fmt.Printf("Generated %d tokens in %v (%.2f tokens/s)\n", len(resultTokens), elapsed, float64(len(resultTokens))/elapsed.Seconds())

	fullText := *prompt + tok.Decode(resultTokens)
	fmt.Printf("\nFull output:\n%s\n", fullText)
}
