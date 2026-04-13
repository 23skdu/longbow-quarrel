//go:build linux && cuda

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
	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/engine"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
	"github.com/23skdu/longbow-quarrel/internal/logger"
	"github.com/23skdu/longbow-quarrel/internal/ollama"
	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
	"github.com/23skdu/longbow-quarrel/internal/metrics"
	"github.com/23skdu/longbow-quarrel/internal/api"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
	modelPath   = flag.String("model", "", "Path to GGUF model file")
	prompt      = flag.String("prompt", "Hello world", "Prompt to generate from")
	numTokens   = flag.Int("n", 20, "Number of tokens to generate")
	metricsAddr  = flag.String("metrics", ":9090", "Address to serve Prometheus metrics")
	kvCacheSize  = flag.Int("kv-cache-size", 2048, "KV cache max sequence length")
	maxBatchSize = flag.Int("max-batch-size", 16, "Maximum number of sequences in a batch")
	blockSize    = flag.Int("block-size", 16, "Paged attention block size")
	totalBlocks  = flag.Int("total-blocks", 256, "Total number of physical blocks in paged cache")

	temperature  = flag.Float64("temp", 0.7, "Temperature for sampling")
	topK         = flag.Int("topk", 40, "Top-K sampling")
	topP         = flag.Float64("topp", 0.95, "Top-P sampling")
	repPenalty   = flag.Float64("penalty", 1.1, "Repetition penalty")
	streamOutput = flag.Bool("stream", false, "Stream tokens as they are generated")
	flightAddr   = flag.String("flight", ":50051", "Address to serve Arrow Flight inference")
)

func main() {
	flag.Parse()

	if *modelPath == "" {
		fmt.Println("Error: --model flag is required")
		flag.Usage()
		os.Exit(1)
	}

	fmt.Printf("=== Longbow-Quarrel CUDA ===\n")
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
	fmt.Printf("Architecture: %s\n", arch)

	layers := 0
	if v, ok := f.KV["llama.block_count"].(uint32); ok {
		layers = int(v)
	}
	fmt.Printf("Layers: %d\n", layers)

	vocabSize := 0
	if v, ok := f.KV["llama.vocab_size"].(uint32); ok {
		vocabSize = int(v)
	}
	fmt.Printf("Vocab: %d\n", vocabSize)

	fmt.Printf("\n=== Initializing CUDA Backend ===\n")

	ctx, err := device.NewCUDAContext()
	if err != nil {
		log.Fatalf("Failed to create CUDA context: %v", err)
	}
	defer ctx.Free()

	cudaModel, err := ctx.NewCUDAModel(f, true, *kvCacheSize)
	if err != nil {
		log.Fatalf("Failed to load model to GPU: %v", err)
	}
	defer cudaModel.Free()

	fmt.Printf("GPU Memory: %.1f MB\n", float64(device.CUDAAllocatedBytes())/1e6)
	fmt.Printf("KV Cache: %d layers\n", len(cudaModel.KCache))

	go func() {
		http.Handle("/metrics", promhttp.Handler())
		if err := http.ListenAndServe(*metricsAddr, nil); err != nil {
		}
	}()

	// Start the background metrics flusher (hotpath-safe)
	metricsFlusher := metrics.NewBgFlusher(5 * time.Second)
	defer metricsFlusher.Stop()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	logger.Log.Info("Loading tokenizer", "model", *modelPath)
	tok, err := tokenizer.New(*modelPath)
	if err != nil {
		log.Fatalf("Failed to initialize tokenizer: %v", err)
	}

	engineConfig := config.Config{
		KVCacheSize: *kvCacheSize,
		SeqLen:      *kvCacheSize,
	}
	// Note: In a real implementation, we would extend config.Config to include these fields.
	// For now, we rely on the engine implementation using these defaults or being updated to use them.

	e, err := engine.NewEngine(*modelPath, engineConfig)
	if err != nil {
		log.Fatalf("Failed to initialize engine: %v", err)
	}
	defer e.Close()

	// Initialize and start Arrow Flight Server
	flightServer := api.NewInferenceFlightServer(*flightAddr, e, tok)
	go func() {
		if err := flightServer.Serve(); err != nil {
			logger.Log.Error("Flight server failed", "error", err)
		}
	}()

	promptTokens := tok.Encode(*prompt)
	fmt.Printf("Prompt tokens: %d\n", len(promptTokens))

	samplerConfig := engine.SamplerConfig{
		Temperature: *temperature,
		TopK:        *topK,
		TopP:        *topP,
		RepPenalty:  *repPenalty,
	}

	startTime := time.Now()

	if *streamOutput {
		fmt.Printf("\nGenerating:\n")
		_, err := e.InferWithCallback(promptTokens, *numTokens, samplerConfig, func(token int) {
			text := tok.Decode([]int{token})
			fmt.Print(text)
		})
		fmt.Printf("\n")
		if err != nil {
			log.Fatalf("Generation failed: %v", err)
		}
	} else {
		outputTokens, err := e.Infer(promptTokens, *numTokens, samplerConfig)
		if err != nil {
			log.Fatalf("Generation failed: %v", err)
		}

		elapsed := time.Since(startTime)
		tokensPerSecond := float64(len(outputTokens)) / elapsed.Seconds()

		fmt.Printf("\n=== Complete ===\n")
		fmt.Printf("%d tokens in %.2fs (%.1f t/s)\n",
			len(outputTokens), elapsed.Seconds(), tokensPerSecond)

		if len(outputTokens) > 0 {
			fmt.Printf("\n%s\n", tok.Decode(outputTokens))
		}
	}
}
