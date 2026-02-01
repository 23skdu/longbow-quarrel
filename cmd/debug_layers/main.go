//go:build darwin && metal

package main

import (
	"fmt"
	"os"
	"runtime"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/engine"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
)

type DebugConfig struct {
	EmbeddingLog   bool
	AttentionLog   bool
	FFNLog         bool
	LayerOutputLog bool
	Quantization   string
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: debug_layers <model_path> [quant_type]")
		fmt.Println("  quant_type: auto, fp16, q4_0, q6_k, iq4_nl (default: auto)")
		os.Exit(1)
	}

	modelPath := os.Args[1]
	quantType := "auto"
	if len(os.Args) > 2 {
		quantType = os.Args[2]
	}

	// Configure quantization
	conf := config.Default()
	switch quantType {
	case "fp16":
		conf.PrecisionMode = config.PrecisionFP16
		fmt.Printf("Quantization: FP16\n")
	case "q4_0":
		// Note: Cannot force specific quant types via config
		fmt.Printf("Quantization: Model default (Q4_0 detected by loader)\n")
	case "q6_k":
		fmt.Printf("Quantization: Model default (Q6_K detected by loader)\n")
	case "iq4_nl":
		fmt.Printf("Quantization: Model default (IQ4_NL detected by loader)\n")
	default:
		fmt.Printf("Quantization: Auto (model default)\n")
	}

	fmt.Printf("=== Deep Debug Tool ===\n")
	fmt.Printf("Model: %s\n", modelPath)
	fmt.Printf("Quantization: %s\n", quantType)
	fmt.Printf("\n")

	// Load engine
	start := time.Now()
	e, err := engine.NewEngine(modelPath, conf)
	if err != nil {
		fmt.Printf("ERROR: Failed to create engine: %v\n", err)
		os.Exit(1)
	}
	defer e.Close()

	loadTime := time.Since(start)
	fmt.Printf("Model loaded in %v\n", loadTime)

	// Load tokenizer
	ggufFile, err := gguf.LoadFile(modelPath)
	if err != nil {
		fmt.Printf("ERROR: Failed to load GGUF: %v\n", err)
		os.Exit(1)
	}
	defer ggufFile.Close()

	tok, err := tokenizer.NewFromGGUF(ggufFile)
	if err != nil {
		fmt.Printf("ERROR: Failed to create tokenizer: %v\n", err)
		os.Exit(1)
	}

	// Test prompts
	prompts := []string{
		"The",
		"The capital of",
		"The capital of France is",
	}

	sampler := engine.SamplerConfig{Temperature: 0, TopK: 1}

	for i, prompt := range prompts {
		fmt.Printf("\n=== Prompt %d: %q ===\n", i+1, prompt)
		fmt.Printf("Generating 1 token...\n")

		startGen := time.Now()
		result, err := e.Infer(tok.Encode(prompt), 1, sampler)
		if err != nil {
			fmt.Printf("ERROR: Inference failed: %v\n", err)
			continue
		}
		genTime := time.Since(startGen)

		if len(result) != 1 {
			fmt.Printf("ERROR: Expected 1 token, got %d\n", len(result))
			continue
		}

		tokenID := result[0]
		tokenText := tok.Decode([]int{tokenID})

		fmt.Printf("Token ID: %d\n", tokenID)
		fmt.Printf("Token Text: %q\n", tokenText)
		fmt.Printf("Gen Time: %v\n", genTime)

		// Check for special tokens
		if tokenID == 0 {
			fmt.Printf("⚠️  WARNING: Token ID 0 (likely UNK)\n")
		}
		if tokenID == 128001 {
			fmt.Printf("⚠️  WARNING: Token ID 128001 (EOS token)\n")
		}
		if tokenID == 29999 {
			fmt.Printf("⚠️  WARNING: Token ID 29999 (Chinese character)\n")
		}

		runtime.GC()
	}

	fmt.Printf("\n=== Summary ===\n")
}
