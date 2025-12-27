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

	_ "net/http/pprof"

	"github.com/23skdu/longbow-quarrel/internal/engine"
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
	flag.Parse()

	if *modelPath == "" {
		fmt.Println("Error: --model flag is required")
		flag.Usage()
		os.Exit(1)
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
	inputTokens := tok.Encode(*prompt)
	log.Printf("Encoded prompt '%s' -> %v (len %d)", *prompt, inputTokens, len(inputTokens))

	log.Printf("Starting inference for %d tokens...", *numTokens)
	
	// Run Inference in a goroutine so we can handle signals?
	// Or just run it.
	
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
			log.Printf("Result tokens: %v", result)
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
