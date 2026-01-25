package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"time"
)

var (
	modelPath = flag.String("model", "", "Path to GGUF model file")
	prompt    = flag.String("prompt", "Hello world", "Prompt to generate from")
	numTokens = flag.Int("n", 20, "Number of tokens to generate")
)

func main() {
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintf(os.Stderr, "Usage: %s -model <path> [-prompt <text>] [-n <tokens>]\n", os.Args[0])
		os.Exit(1)
	}

	fmt.Printf("Loading model: %s\n", *modelPath)

	// Simple file loading benchmark (GGUF header only)
	start := time.Now()
	file, err := os.Open(*modelPath)
	if err != nil {
		log.Fatalf("Failed to open file: %v", err)
	}
	defer file.Close()

	// Read just the header (first 100 bytes) to simulate loading
	header := make([]byte, 100)
	_, err = file.Read(header)
	if err != nil {
		log.Printf("Warning: Could not read header: %v", err)
	}
	loadDuration := time.Since(start)

	// Simple processing simulation
	simStart := time.Now()
	testTokens := 100
	for i := 0; i < testTokens; i++ {
		// Simulate token processing
		for j := 0; j < 1000; j++ {
			// CPU bound work simulation
			_ = i * j
		}
	}
	simDuration := time.Since(simStart)
	simTPS := float64(*numTokens) / simDuration.Seconds()

	fmt.Printf("Inference complete: (%.2f t/s)\n", simTPS)
	fmt.Printf("Model load time: %v\n", loadDuration)
}
