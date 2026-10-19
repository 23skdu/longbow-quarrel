//go:build linux && tpu && !cuda

package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/api"
	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/engine"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
	"github.com/23skdu/longbow-quarrel/internal/logger"
	"github.com/23skdu/longbow-quarrel/internal/metrics"
	"github.com/23skdu/longbow-quarrel/internal/ollama"
	"github.com/23skdu/longbow-quarrel/internal/telemetry"
	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
	modelPath    = flag.String("model", "", "Path to GGUF model file")
	prompt       = flag.String("prompt", "Hello world", "Prompt to generate from")
	numTokens    = flag.Int("n", 20, "Number of tokens to generate")
	metricsAddr  = flag.String("metrics", ":9090", "Address to serve Prometheus metrics")
	apiAddr      = flag.String("api-addr", ":8080", "Address to serve the OpenAI-compatible REST API (empty disables it)")
	kvCacheSize  = flag.Int("kv-cache-size", 2048, "KV cache max sequence length")
	maxBatchSize = flag.Int("max-batch-size", 16, "Maximum number of sequences in a batch")
	blockSize    = flag.Int("block-size", 16, "Paged attention block size")
	totalBlocks  = flag.Int("total-blocks", 256, "Total number of physical blocks in paged cache")

	temperature  = flag.Float64("temp", 0.7, "Temperature for sampling")
	topK         = flag.Int("topk", 40, "Top-K sampling")
	topP         = flag.Float64("topp", 0.95, "Top-P sampling")
	repPenalty   = flag.Float64("rep-penalty", 1.1, "Repetition penalty")
	streamOutput = flag.Bool("stream", false, "Stream tokens as they are generated")
	flightAddr   = flag.String("flight", ":50051", "Address to serve Arrow Flight inference")
	DraftModel   = flag.String("draft-model", "", "Path to a smaller GGUF model used for speculative decoding (optional)")
	DraftLength  = flag.Int("draft-length", 4, "How many tokens the draft model proposes per step")
	DraftPaths   = flag.Int("draft-paths", 2, "How many parallel draft paths to generate per step")
)

func main() {
	// -help / -version are handled before flag.Parse so that the container
	// HEALTHCHECK (`quarrel --help`) exits 0 instead of 2.
	if handleHelpFlag() {
		return
	}
	flag.Parse()
	if handleVersionFlag() {
		return
	}

	if *modelPath == "" {
		fmt.Println("Error: --model flag is required")
		flag.Usage()
		os.Exit(1)
	}

	fmt.Printf("=== Longbow-Quarrel TPU ===\n")
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

	fmt.Printf("\n=== Initializing TPU Backend ===\n")

	ctx, err := device.NewTPUContext()
	if err != nil {
		log.Fatalf("Failed to create TPU context: %v", err)
	}
	defer ctx.Free()

	tpuModel, err := ctx.LoadTPUModel(f, true, *kvCacheSize)
	if err != nil {
		log.Fatalf("Failed to load model to TPU: %v", err)
	}
	defer tpuModel.Free()

	fmt.Printf("TPU Memory: %.1f MB\n", float64(device.TPUAllocatedBytes())/1e6)
	fmt.Printf("KV Cache: %d layers\n", len(tpuModel.KCache))

	go func() {
		http.Handle("/metrics", promhttp.Handler())
		if err := http.ListenAndServe(*metricsAddr, nil); err != nil {
		}
	}()

	metricsFlusher := metrics.NewBgFlusher(5 * time.Second)
	defer metricsFlusher.Stop()

	// OpenTelemetry tracing. The tracer is only really initialised when
	// an OTLP endpoint is configured; without one it stays a no-op so
	// the instrumentation costs nothing.
	shutdownTracer := telemetry.InitTracer()
	if shutdownTracer != nil {
		defer func() { _ = shutdownTracer(context.Background()) }()
	}

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

	e, err := engine.NewEngine(*modelPath, engineConfig)
	if err != nil {
		log.Fatalf("Failed to initialize engine: %v", err)
	}

	// Speculative decoding: a smaller draft model proposes the next tokens
	// and the target model verifies them, which trades draft compute for
	// fewer target forward passes. The defer has to come after the
	// wrapping, because the wrapper owns both engines.
	// The target engine is kept separately: the REST API serves ordinary
	// requests with it and only uses the speculative engine for the
	// requests that ask for it.
	targetEngine := e
	var specEngine engine.Engine
	if *DraftModel != "" {
		engineConfig.SpeculativeDraftModel = *DraftModel
		engineConfig.SpeculativeDraftLength = *DraftLength
		engineConfig.SpeculativeNumPaths = *DraftPaths
		speculative, specErr := engine.NewSpeculativeEngine(e, *DraftModel, engineConfig)
		if specErr != nil {
			logger.Log.Error("Failed to enable speculative decoding", "error", specErr)
		} else {
			specEngine = speculative
			e = speculative
			fmt.Printf("Speculative decoding enabled with draft model: %s\n", *DraftModel)
		}
	}
	defer e.Close()

	flightServer := api.NewInferenceFlightServer(*flightAddr, e, tok)
	go func() {
		if err := flightServer.Serve(); err != nil {
			logger.Log.Error("Flight server failed", "error", err)
		}
	}()

	// OpenAI-compatible REST API. This is the surface documented in README.md
	// and docs/api.md; it was previously never started by any binary.
	apiSrv, err := startAPIServer(*apiAddr, func() int64 { return int64(device.TPUAllocatedBytes()) }, targetEngine, specEngine, tok)
	if err != nil {
		logger.Log.Error("failed to start REST API", "error", err)
	}
	if apiSrv != nil {
		defer apiSrv.Close()
		fmt.Println(apiSrv.describe())
	}

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
