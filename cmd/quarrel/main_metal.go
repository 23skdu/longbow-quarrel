//go:build darwin && metal && !cuda

package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/engine"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
	"github.com/23skdu/longbow-quarrel/internal/logger"
	"github.com/23skdu/longbow-quarrel/internal/metrics"
	"github.com/23skdu/longbow-quarrel/internal/ollama"
	"github.com/23skdu/longbow-quarrel/internal/simd"
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

	temperature     = flag.Float64("temp", 0.7, "Temperature for sampling")
	topK            = flag.Int("topk", 40, "Top-K sampling")
	topP            = flag.Float64("topp", 0.95, "Top-P sampling")
	repPenalty      = flag.Float64("penalty", 1.1, "Repetition penalty")
	streamOutput    = flag.Bool("stream", false, "Stream tokens as they are generated")
	ActivationLog   = flag.String("activation-log", "", "Write a JSON activation log (embedding, per-layer activations, logits) to this path")
	DraftModel      = flag.String("draft-model", "", "Path to a smaller GGUF model used for speculative decoding (optional)")
	DraftLength     = flag.Int("draft-length", 4, "How many tokens the draft model proposes per step")
	DraftPaths      = flag.Int("draft-paths", 2, "How many parallel draft paths to generate per step")
	stopTokensFlag  = flag.String("stop-tokens", "", "Comma-separated list of stop token IDs (e.g. 2,13)")
	kvCacheTypeFlag = flag.String("kv-cache-type", "f16", "Precision of the KV cache: f16, tq1_0, tq2_0, fp8, q8_0")
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

	fmt.Printf("=== Longbow-Quarrel Metal ===\n")
	metrics.RecordSIMDLevel(simd.GetCPULevel())
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

	// Start the background metrics flusher (hotpath-safe)
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
		KVCacheSize:  *kvCacheSize,
		SeqLen:       *kvCacheSize,
		MaxBatchSize: *maxBatchSize,
		BlockSize:    *blockSize,
		TotalBlocks:  *totalBlocks,
		MaxTokens:    *numTokens,
	}
	kvCacheType, err := engine.ParseKVCacheType(*kvCacheTypeFlag)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	engineConfig.KVCacheType = kvCacheType
	if *ActivationLog != "" {
		engineConfig.ActivationLogPath = *ActivationLog
	}
	if *stopTokensFlag != "" {
		for _, s := range strings.Split(*stopTokensFlag, ",") {
			s = strings.TrimSpace(s)
			if s == "" {
				continue
			}
			var id int
			if _, err := fmt.Sscanf(s, "%d", &id); err == nil {
				engineConfig.StopTokens = append(engineConfig.StopTokens, id)
			}
		}
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

	// OpenAI-compatible REST API. This is the surface documented in README.md
	// and docs/api.md; it was previously never started by any binary.
	apiSrv, err := startAPIServer(*apiAddr, device.AllocatedBytes, targetEngine, specEngine, tok)
	if err != nil {
		logger.Log.Error("failed to start REST API", "error", err)
	}
	if apiSrv != nil {
		defer apiSrv.Close()
		fmt.Println(apiSrv.describe())
	}

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
			StopTokens:  engineConfig.StopTokens,
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
