//go:build !cuda && !metal && !tpu

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

	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/engine"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
	"github.com/23skdu/longbow-quarrel/internal/logger"
	"github.com/23skdu/longbow-quarrel/internal/metrics"
	"github.com/23skdu/longbow-quarrel/internal/models"
	"github.com/23skdu/longbow-quarrel/internal/simd"
	"github.com/23skdu/longbow-quarrel/internal/telemetry"
	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
	cpuModelPath     = flag.String("model", "", "Path to GGUF model file")
	cpuPrompt        = flag.String("prompt", "Hello world", "Prompt to generate from")
	cpuNumTokens     = flag.Int("n", 20, "Number of tokens to generate")
	cpuMetricsAddr   = flag.String("metrics", ":9090", "Address to serve Prometheus metrics")
	cpuAPIAddr       = flag.String("api-addr", ":8080", "Address to serve the OpenAI-compatible REST API (empty disables it)")
	cpuKVCacheSize   = flag.Int("kv-cache-size", 2048, "KV cache max sequence length")
	cpuTemp          = flag.Float64("temp", 0.7, "Temperature for sampling")
	cpuTopK          = flag.Int("topk", 40, "Top-K sampling")
	cpuTopP          = flag.Float64("topp", 0.95, "Top-P sampling")
	cpuRepPenalty    = flag.Float64("rep-penalty", 1.1, "Repetition penalty")
	cpuMinP          = flag.Float64("min-p", 0.0, "Min-P sampling threshold")
	cpuSeed          = flag.Int64("seed", 0, "Random seed (0 for auto)")
	cpuStream        = flag.Bool("stream", false, "Stream tokens as they are generated")
	cpuMaxBatchSize  = flag.Int("max-batch-size", 16, "Maximum number of sequences in a batch")
	cpuLoraPath      = flag.String("lora", "", "Path to LoRA adapter .gguf file (optional)")
	cpuActivationLog = flag.String("activation-log", "", "Write a JSON activation log (embedding, per-layer activations, logits) to this path")
	cpuDraftModel    = flag.String("draft-model", "", "Path to a smaller GGUF model used for speculative decoding (optional)")
	cpuDraftLength   = flag.Int("draft-length", 4, "How many tokens the draft model proposes per step")
	cpuDraftPaths    = flag.Int("draft-paths", 2, "How many parallel draft paths to generate per step")
	cpuBlockSize     = flag.Int("block-size", 16, "Paged attention block size")
	cpuTotalBlocks   = flag.Int("total-blocks", 256, "Total number of physical blocks in paged cache")
	cpuStopTokens    = flag.String("stop-tokens", "", "Comma-separated list of stop token IDs (e.g. 2,13)")
)

// GPU-offload flags are accepted for command-line compatibility with the CUDA
// build but have no effect: the CPU engine is always fully host-resident.
// They are registered without binding a variable so existing scripts and
// Docker CMDs that pass -ngl / -gpu-layers keep working on the CPU image.
var _ = flag.Int("ngl", 0, "Ignored on the CPU build; accepted for flag compatibility")
var _ = flag.Int("gpu-layers", 0, "Alias for -ngl (ignored on the CPU build)")

func main() {
	flag.Usage = func() {
		out := flag.CommandLine.Output()
		_, _ = fmt.Fprintf(out, "Longbow-Quarrel - CPU inference server\n\nUsage: %s -model <model.gguf> [options]\n\nOptions:\n", os.Args[0])
		flag.PrintDefaults()
	}

	// flag.Parse() exits 2 on an undefined flag, which made `--help` fail in
	// every container HEALTHCHECK. Intercept the help flags before parsing so
	// they print usage and exit 0.
	if handleHelpFlag() {
		return
	}
	flag.Parse()
	if handleVersionFlag() {
		return
	}

	if *cpuModelPath == "" {
		fmt.Fprintln(os.Stderr, "Error: --model flag is required")
		flag.Usage()
		os.Exit(1)
	}

	printVersion()
	metrics.RecordSIMDLevel(simd.GetCPULevel())
	fmt.Printf("=== Longbow-Quarrel (CPU) ===\nModel: %s\n", *cpuModelPath)

	// Universal model resolution: accept a bare name like "qwen3" as well as a
	// path, searching the well-known local model caches.
	if resolved, err := models.ResolveModelPath(*cpuModelPath); err == nil {
		logger.Log.Info("Resolved model path", "original", *cpuModelPath, "resolved", resolved)
		*cpuModelPath = resolved
	}

	f, err := gguf.LoadFile(*cpuModelPath)
	if err != nil {
		log.Fatalf("Failed to load GGUF: %v", err)
	}
	defer func() {
		if cerr := f.Close(); cerr != nil {
			logger.Log.Warn("failed to close GGUF file", "error", cerr)
		}
	}()

	arch := "unknown"
	if v, ok := f.KV["general.architecture"].(string); ok {
		arch = v
	}
	fmt.Printf("Architecture: %s\n", arch)

	engineConfig := engine.ExtractModelConfig(f)
	if *cpuActivationLog != "" {
		engineConfig.ActivationLogPath = *cpuActivationLog
	}
	if *cpuKVCacheSize > 0 {
		engineConfig.KVCacheSize = *cpuKVCacheSize
		engineConfig.SeqLen = *cpuKVCacheSize
	}
	if *cpuMaxBatchSize > 0 {
		engineConfig.MaxBatchSize = *cpuMaxBatchSize
	}
	if *cpuBlockSize > 0 {
		engineConfig.BlockSize = *cpuBlockSize
	}
	if *cpuTotalBlocks > 0 {
		engineConfig.TotalBlocks = *cpuTotalBlocks
	}
	if *cpuStopTokens != "" {
		for _, s := range strings.Split(*cpuStopTokens, ",") {
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
	// The CPU engine has no device offload; force a fully host-resident run so
	// the registered factory cannot select a GPU backend.
	engineConfig.NumGPULayers = 0

	fmt.Printf("Layers: %d\n", engineConfig.Layers)
	fmt.Printf("Vocab: %d\n", engineConfig.VocabSize)
	fmt.Printf("Dim: %d, Heads: %d, KV Heads: %d\n", engineConfig.Dim, engineConfig.Heads, engineConfig.KVHeads)

	// Prometheus metrics endpoint.
	go func() {
		mux := http.NewServeMux()
		mux.Handle("/metrics", promhttp.Handler())
		srv := &http.Server{
			Addr:              *cpuMetricsAddr,
			Handler:           mux,
			ReadHeaderTimeout: 5 * time.Second,
		}
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Log.Warn("Metrics server exited", "error", err)
		}
	}()
	fmt.Printf("Metrics: http://localhost%s/metrics\n", *cpuMetricsAddr)

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

	logger.Log.Info("Loading tokenizer", "model", *cpuModelPath)
	tok, err := tokenizer.New(*cpuModelPath)
	if err != nil {
		log.Fatalf("Failed to initialize tokenizer: %v", err)
	}

	e, err := engine.NewEngine(*cpuModelPath, engineConfig)
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
	if *cpuDraftModel != "" {
		engineConfig.SpeculativeDraftModel = *cpuDraftModel
		engineConfig.SpeculativeDraftLength = *cpuDraftLength
		engineConfig.SpeculativeNumPaths = *cpuDraftPaths
		speculative, specErr := engine.NewSpeculativeEngine(e, *cpuDraftModel, engineConfig)
		if specErr != nil {
			logger.Log.Error("Failed to enable speculative decoding", "error", specErr)
		} else {
			specEngine = speculative
			e = speculative
			fmt.Printf("Speculative decoding enabled with draft model: %s\n", *cpuDraftModel)
		}
	}
	defer e.Close()

	if *cpuLoraPath != "" {
		if err := e.LoadAdapter(*cpuLoraPath, "cli-lora"); err != nil {
			logger.Log.Warn("Failed to load LoRA adapter", "error", err)
		} else {
			fmt.Printf("Loaded LoRA adapter: %s\n", *cpuLoraPath)
		}
	}

	// OpenAI-compatible REST API. This is the surface documented in README.md
	// and docs/api.md; it was previously never started by any binary.
	apiSrv, err := startAPIServer(*cpuAPIAddr, device.CPUAllocatedBytes, targetEngine, specEngine, tok)
	if err != nil {
		logger.Log.Error("failed to start REST API", "error", err)
	}
	if apiSrv != nil {
		defer apiSrv.Close()
		fmt.Println(apiSrv.describe())
	}

	promptTokens := tok.Encode(*cpuPrompt)
	fmt.Printf("Prompt tokens: %d\n", len(promptTokens))

	samplerConfig := engine.SamplerConfig{
		Temperature: *cpuTemp,
		TopK:        *cpuTopK,
		TopP:        *cpuTopP,
		RepPenalty:  *cpuRepPenalty,
		MinP:        *cpuMinP,
		Seed:        *cpuSeed,
		StopTokens:  engineConfig.StopTokens,
	}

	startTime := time.Now()

	if *cpuStream {
		fmt.Printf("\nGenerating:\n")
		if _, err := e.InferWithCallback(promptTokens, *cpuNumTokens, samplerConfig, func(token int) {
			fmt.Print(tok.Decode([]int{token}))
		}); err != nil {
			log.Fatalf("Generation failed: %v", err)
		}
		fmt.Println()
		return
	}

	outputTokens, err := e.Infer(promptTokens, *cpuNumTokens, samplerConfig)
	if err != nil {
		log.Fatalf("Generation failed: %v", err)
	}

	elapsed := time.Since(startTime)
	tps := 0.0
	if elapsed.Seconds() > 0 {
		tps = float64(len(outputTokens)) / elapsed.Seconds()
	}

	fmt.Printf("\n=== Complete ===\n")
	fmt.Printf("%d tokens in %.2fs (%.1f t/s)\n", len(outputTokens), elapsed.Seconds(), tps)
	if len(outputTokens) > 0 {
		fmt.Printf("\n%s\n", tok.Decode(outputTokens))
	}
}
