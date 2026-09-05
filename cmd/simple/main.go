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
	"github.com/23skdu/longbow-quarrel/internal/models"
	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
)

func main() {
	modelPath := flag.String("model", "", "Path to GGUF model file")
	prompt := flag.String("prompt", "The capital of France is", "Prompt for text generation")
	maxTokens := flag.Int("n", 50, "Maximum tokens to generate")
	temp := flag.Float64("temp", 0.7, "Temperature for sampling")
	topK := flag.Int("topk", 40, "Top-K sampling")
	topP := flag.Float64("topp", 0.95, "Top-P sampling")
	repPenalty := flag.Float64("rep-penalty", 1.1, "Repetition penalty")
	presencePenalty := flag.Float64("presence-penalty", 0.0, "Presence penalty")
	frequencyPenalty := flag.Float64("frequency-penalty", 0.0, "Frequency penalty")
	minP := flag.Float64("min-p", 0.0, "Min-P sampling threshold")
	seed := flag.Int64("seed", 0, "Random seed (0 for auto)")
	streamOutput := flag.Bool("stream", false, "Stream tokens as they are generated")
	maxMemMB := flag.Int64("max-memory", 0, "Max memory in MB (0 = no limit)")
	numGPULayers := flag.Int("ngl", -1, "Number of GPU layers to offload (-1 for all)")
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "Error: --model flag is required")
		os.Exit(1)
	}

	resolved, err := models.ResolveModelPath(*modelPath)
	if err == nil {
		*modelPath = resolved
	}

	fmt.Printf("=== Longbow-Quarrel Cross-Platform CLI ===\n")
	fmt.Printf("Go Version: %s\n", runtime.Version())
	fmt.Printf("NumCPU: %d\n", runtime.NumCPU())
	fmt.Printf("Model: %s\n", *modelPath)
	fmt.Printf("Prompt: %s\n", *prompt)
	fmt.Printf("Temp: %.2f, TopK: %d, TopP: %.2f, MaxTokens: %d\n", *temp, *topK, *topP, *maxTokens)
	fmt.Println()

	f, err := gguf.LoadFile(*modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load GGUF: %v\n", err)
		os.Exit(1)
	}
	defer func() { _ = f.Close() }()

	fmt.Printf("GGUF Version: %d\n", f.Header.Version)
	fmt.Printf("Tensors: %d\n", len(f.Tensors))

	mCfg := engine.ExtractModelConfig(f)
	fmt.Printf("Architecture: %s\n", mCfg.Architecture)
	fmt.Printf("Layers: %d\n", mCfg.Layers)
	fmt.Printf("Embedding Dim: %d\n", mCfg.Dim)
	fmt.Printf("Heads: %d (KV: %d, HeadDim: %d)\n", mCfg.Heads, mCfg.KVHeads, mCfg.HeadDim)
	fmt.Printf("Vocab Size: %d\n", mCfg.VocabSize)

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
	cfg.NumGPULayers = *numGPULayers
	if cfg.MaxMemoryMB > 0 {
		fmt.Printf("Memory limit: %d MB\n", cfg.MaxMemoryMB)
	}

	samplerCfg := engine.SamplerConfig{
		Temperature:      *temp,
		TopK:             *topK,
		TopP:             *topP,
		RepPenalty:       *repPenalty,
		PresencePenalty:  *presencePenalty,
		FrequencyPenalty: *frequencyPenalty,
		MinP:             *minP,
		Seed:             *seed,
	}

	e, err := engine.NewEngine(*modelPath, cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create engine: %v\n", err)
		os.Exit(1)
	}
	defer e.Close()

	var resultTokens []int
	if *streamOutput {
		fmt.Printf("\nGenerating:\n")
		resultTokens, err = e.InferWithCallback(inputTokens, *maxTokens, samplerCfg, func(tokenID int) {
			fmt.Print(tok.Decode([]int{tokenID}))
		})
		fmt.Println()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Inference failed: %v\n", err)
			os.Exit(1)
		}
	} else {
		resultTokens, err = e.Infer(inputTokens, *maxTokens, samplerCfg)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Inference failed: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Generated: ")
		for _, tokenID := range resultTokens {
			fmt.Printf("%d:%s ", tokenID, tok.Decode([]int{tokenID}))
		}
		fmt.Println()
	}

	elapsed := time.Since(startTime)
	fmt.Printf("Generated %d tokens in %v (%.2f tokens/s)\n", len(resultTokens), elapsed, float64(len(resultTokens))/elapsed.Seconds())

	fullText := *prompt + tok.Decode(resultTokens)
	fmt.Printf("\nFull output:\n%s\n", fullText)
}
