//go:build darwin && metal

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

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/engine"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
	"github.com/23skdu/longbow-quarrel/internal/logger"
	"github.com/23skdu/longbow-quarrel/internal/ollama"
	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
	modelPath   = flag.String("model", "", "Path to GGUF model file")
	prompt      = flag.String("prompt", "Hello world", "Prompt to generate from")
	numTokens   = flag.Int("n", 20, "Number of tokens to generate")
	metricsAddr = flag.String("metrics", ":9090", "Address to serve Prometheus metrics")
	kvCacheSize = flag.Int("kv-cache-size", 2048, "KV cache max sequence length")

	temperature  = flag.Float64("temp", 0.7, "Temperature for sampling")
	topK         = flag.Int("topk", 40, "Top-K sampling")
	topP         = flag.Float64("topp", 0.95, "Top-P sampling")
	repPenalty   = flag.Float64("penalty", 1.1, "Repetition penalty")
	streamOutput = flag.Bool("stream", false, "Stream tokens as they are generated")
)

func main() {
	flag.Parse()

	if *modelPath == "" {
		fmt.Println("Error: --model flag is required")
		flag.Usage()
		os.Exit(1)
	}

	fmt.Printf("=== Longbow-Quarrel Metal ===\n")
	fmt.Printf("Model: %s\n", *modelPath)

	resolvedPath, err := ollama.ResolveModelPath(*modelPath)
	if err == nil {
		logger.Log.Info("Resolved Ollama model", "original", *modelPath, "resolved", resolvedPath)
		*modelPath = resolvedPath
	}

	f, err := gguf.LoadFile(*modelPath)
	if err != nil {
		log.Fatalf("Failed to load GGUF: %v", err)
	}
	defer f.Close()

	arch := "unknown"
	if v, ok := f.KV["general.architecture"].(string); ok {
		arch = v
	}

	layers := 1
	if v, ok := f.KV["llama.block_count"].(uint32); ok {
		layers = int(v)
	}

	vocabSize := 49152
	if v, ok := f.KV["llama.vocab_size"].(uint32); ok {
		vocabSize = int(v)
	}

	seqLen := 2048
	if v, ok := f.KV["llama.context_length"].(uint32); ok {
		seqLen = int(v)
	}

	fmt.Printf("Architecture: %s\n", arch)
	fmt.Printf("Layers: %d\n", layers)
	fmt.Printf("Vocab: %d\n", vocabSize)

	go func() {
		http.Handle("/metrics", promhttp.Handler())
		logger.Log.Info("Metrics serving", "address", *metricsAddr+"/metrics")
		if err := http.ListenAndServe(*metricsAddr, nil); err != nil {
			logger.Log.Info("Metrics server error", "error", err)
		}
	}()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	logger.Log.Info("Loading tokenizer", "model", *modelPath)
	tok, err := tokenizer.New(*modelPath)
	if err != nil {
		log.Fatalf("Failed to initialize tokenizer: %v", err)
	}

	cfg := config.Config{
		KVCacheSize: *kvCacheSize,
	}

	e, err := engine.NewEngine(*modelPath, cfg)
	if err != nil {
		log.Fatalf("Failed to initialize engine: %v", err)
	}
	defer e.Close()

	inputTokens := tok.Encode(*prompt)
	inputTokens = append([]int{1}, inputTokens...)

	logger.Log.Info("Starting inference", "num_tokens", *numTokens)

	doneChan := make(chan struct{})

	go func() {
		start := time.Now()

		samplerConfig := engine.SamplerConfig{
			Temperature: *temperature,
			TopK:        *topK,
			TopP:        *topP,
			RepPenalty:  *repPenalty,
			Seed:        time.Now().UnixNano(),
		}

		var result []int
		var err error

		if *streamOutput {
			fmt.Print("Streaming output: ")
			result, err = e.InferWithCallback(inputTokens, *numTokens, samplerConfig, func(token int) {
				decoded := tok.Decode([]int{token})
				fmt.Print(decoded)
			})
			fmt.Println(" [END]")
		} else {
			result, err = e.Infer(inputTokens, *numTokens, samplerConfig)
		}

		if err != nil {
			logger.Log.Info("Inference error", "error", err)
		} else {
			duration := time.Since(start)
			tokensPerSec := float64(len(result)) / duration.Seconds()
			logger.Log.Info("Inference complete", "tokens", len(result), "duration", duration, "tps", tokensPerSec)

			fmt.Printf("\n=== Complete ===\n")
			fmt.Printf("%d tokens in %v (%.1f t/s)\n\n", len(result), duration, tokensPerSec)
			fmt.Println(tok.Decode(result))
		}
		close(doneChan)
	}()

	select {
	case <-doneChan:
	case <-sigChan:
		log.Println("Interrupt received, shutting down...")
	}
}
