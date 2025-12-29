package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/engine"
	"github.com/23skdu/longbow-quarrel/internal/ollama"
	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
	modelPath   = flag.String("model", "", "Path to GGUF model file")
	prompt      = flag.String("prompt", "Hello world", "Prompt to generate from")
	numTokens   = flag.Int("n", 20, "Number of tokens to generate")
	metricsAddr = flag.String("metrics", ":9090", "Address to serve Prometheus metrics")
)

func main() {
	// Increase File Descriptor Limit (Metal Buffers might use FDs)
	var rLimit syscall.Rlimit
	if err := syscall.Getrlimit(syscall.RLIMIT_NOFILE, &rLimit); err == nil {
		rLimit.Cur = 10240
		if rLimit.Max < 10240 { rLimit.Max = 10240 }
		syscall.Setrlimit(syscall.RLIMIT_NOFILE, &rLimit)
	}

	flag.Parse()

	if *modelPath == "" {
		fmt.Println("Error: --model flag is required")
		flag.Usage()
		os.Exit(1)
	}

	// Try to resolve as Ollama model name first
	resolvedPath, err := ollama.ResolveModelPath(*modelPath)
	if err == nil {
		log.Printf("Resolved Ollama model '%s' to %s", *modelPath, resolvedPath)
		*modelPath = resolvedPath
	} else {
		// Not an Ollama model, treat as direct path
		log.Printf("Using direct model path: %s", *modelPath)
	}

	// Start Metrics Server
	go func() {
		http.Handle("/metrics", promhttp.Handler())
		log.Printf("Metrics serving on %s/metrics", *metricsAddr)
		if err := http.ListenAndServe(*metricsAddr, nil); err != nil {
			log.Printf("Metrics server error: %v", err)
		}
	}()

	// Signal handling
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	// Initialize Tokenizer
	log.Printf("Loading tokenizer from %s...", *modelPath)
	tok, err := tokenizer.New(*modelPath)
	if err != nil {
		log.Fatalf("Failed to initialize tokenizer: %v", err)
	}

	// Initialize Engine
	log.Printf("Loading model from %s...", *modelPath)
	e, err := engine.NewEngine(*modelPath)
	if err != nil {
		log.Fatalf("Failed to initialize engine: %v", err)
	}
	defer e.Close()

	// Tokenize
	inputTokensRaw := tok.Encode(*prompt)
	
	// Prepend BOS for SmolLM2 (1)
	inputTokens := make([]int, 0, len(inputTokensRaw)+1)
	inputTokens = append(inputTokens, 1)
	inputTokens = append(inputTokens, inputTokensRaw...)
	
	log.Printf("Encoded prompt '%s' -> %v (len %d)", *prompt, inputTokens, len(inputTokens))

	log.Printf("Starting inference for %d tokens...", *numTokens)
	
	doneChan := make(chan struct{})
	
	go func() {
		start := time.Now()
		result, err := e.Infer(inputTokens, *numTokens)
		if err != nil {
			log.Printf("Inference error: %v", err)
		} else {
			duration := time.Since(start)
			tokensPerSec := float64(len(result)) / duration.Seconds()
			log.Printf("Inference complete: generated %d tokens in %v (%.2f t/s)", 
				len(result), duration, tokensPerSec)
			
			// Debug print each token
			fmt.Print("Result Tokens Detail: ")
			for _, id := range result {
				decoded := tok.Decode([]int{id})
				fmt.Printf("[%d:'%s'] ", id, decoded)
			}
			fmt.Println()

			log.Printf("Decoded Text: %s", tok.Decode(result))
		}
		close(doneChan)
	}()
	
	select {
	case <-doneChan:
		// done
	case <-sigChan:
		log.Println("Interrupt received, shutting down...")
		// e.Close() called by defer, but we might want early exit
	}
}
