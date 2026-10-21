//go:build linux && amd64 && cuda && cgo

package main

import (
	"context"
	"encoding/base64"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/api"
	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/engine"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
	"github.com/23skdu/longbow-quarrel/internal/logger"
	"github.com/23skdu/longbow-quarrel/internal/metrics"
	"github.com/23skdu/longbow-quarrel/internal/models"
	"github.com/23skdu/longbow-quarrel/internal/simd"
	"github.com/23skdu/longbow-quarrel/internal/telemetry"
	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
	"github.com/23skdu/longbow-quarrel/internal/vlm"
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

	temperature      = flag.Float64("temp", 0.7, "Temperature for sampling")
	topK             = flag.Int("topk", 40, "Top-K sampling")
	topP             = flag.Float64("topp", 0.95, "Top-P sampling")
	repPenalty       = flag.Float64("rep-penalty", 1.1, "Repetition penalty")
	presencePenalty  = flag.Float64("presence-penalty", 0.0, "Presence penalty")
	frequencyPenalty = flag.Float64("frequency-penalty", 0.0, "Frequency penalty")
	minP             = flag.Float64("min-p", 0.0, "Min-P sampling threshold")
	seed             = flag.Int64("seed", 0, "Random seed (0 for auto)")
	streamOutput     = flag.Bool("stream", false, "Stream tokens as they are generated")
	flightAddr       = flag.String("flight", ":50051", "Address to serve Arrow Flight inference")
	numGPULayers     = flag.Int("ngl", -1, "Number of layers to offload to GPU (-1 for all)")
	gpuLayers        = flag.Int("gpu-layers", -1, "Alias for -ngl")
	loraPath         = flag.String("lora", "", "Path to LoRA adapter .gguf file (optional)")
	ActivationLog    = flag.String("activation-log", "", "Write a JSON activation log (embedding, per-layer activations, logits) to this path")
	DraftModel       = flag.String("draft-model", "", "Path to a smaller GGUF model used for speculative decoding (optional)")
	DraftLength      = flag.Int("draft-length", 4, "How many tokens the draft model proposes per step")
	DraftPaths       = flag.Int("draft-paths", 2, "How many parallel draft paths to generate per step")
	imagePath        = flag.String("image", "", "Path to image file for VLM inference (optional)")
	stopTokensFlag   = flag.String("stop-tokens", "", "Comma-separated list of stop token IDs (e.g. 2,13)")
	kvCacheTypeFlag  = flag.String("kv-cache-type", "f16", "Precision of the KV cache: f16, tq1_0, tq2_0, fp8, q8_0")

	// Multi-GPU Flags
	devicesFlag  = flag.String("devices", "", "Comma-separated list of CUDA device IDs for multi-GPU (e.g. 0,1)")
	tpSizeFlag   = flag.Int("tp-size", 1, "Tensor parallel degree (default 1)")
	ppStagesFlag = flag.Int("pp-stages", 1, "Pipeline parallel stages (default 1)")
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

	fmt.Printf("=== Longbow-Quarrel CUDA ===\n")
	metrics.RecordSIMDLevel(simd.GetCPULevel())
	fmt.Printf("Model: %s\n", *modelPath)

	resolvedPath, err := models.ResolveModelPath(*modelPath)
	if err == nil {
		logger.Log.Info("Resolved model path", "original", *modelPath, "resolved", resolvedPath)
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

	engineConfig := engine.ExtractModelConfig(f)
	if *ActivationLog != "" {
		engineConfig.ActivationLogPath = *ActivationLog
	}
	if *kvCacheSize > 0 {
		engineConfig.KVCacheSize = *kvCacheSize
		engineConfig.SeqLen = *kvCacheSize
	}
	if *gpuLayers >= 0 && *numGPULayers < 0 {
		*numGPULayers = *gpuLayers
	}
	if *numGPULayers >= 0 {
		engineConfig.NumGPULayers = *numGPULayers
	}
	if *maxBatchSize > 0 {
		engineConfig.MaxBatchSize = *maxBatchSize
	}
	if *blockSize > 0 {
		engineConfig.BlockSize = *blockSize
	}
	if *totalBlocks > 0 {
		engineConfig.TotalBlocks = *totalBlocks
	}
	kvCacheType, err := engine.ParseKVCacheType(*kvCacheTypeFlag)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	engineConfig.KVCacheType = kvCacheType
	engineConfig.MaxTokens = *numTokens
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

	var devIDs []int
	if *devicesFlag != "" {
		for _, s := range strings.Split(*devicesFlag, ",") {
			s = strings.TrimSpace(s)
			if s == "" {
				continue
			}
			var id int
			if _, err := fmt.Sscanf(s, "%d", &id); err == nil {
				devIDs = append(devIDs, id)
			}
		}
	}

	engineConfig.Devices = devIDs
	engineConfig.TensorParallelSize = *tpSizeFlag
	engineConfig.PipelineStages = *ppStagesFlag

	// A single device is a real setting: it selects the CUDA device the
	// engine runs on.
	//
	// A topology that spans more than one device is not. The managers in
	// internal/device/multi_gpu.go are complete, but the inference engine
	// does not use them: it loads the whole model onto one device and
	// never calls a collective. Asking for tensor or pipeline parallelism
	// used to print a topology line and then run the model on a single
	// GPU, which is worse than an error because the user believes their
	// model was split.
	multiDevice := len(devIDs) > 1 || *tpSizeFlag > 1 || *ppStagesFlag > 1
	if multiDevice {
		fmt.Fprintf(os.Stderr,
			"error: multi-device parallelism is not implemented: the inference engine loads the whole model onto a single device.\n"+
				"       -devices=%v -tp-size=%d -pp-stages=%d was requested, but no model sharding, no cross-device\n"+
				"       activation transfer and no collective is performed. The supported way to spread a model\n"+
				"       across devices is the Arrow Flight distributed path (see docs/roadmap.md, Step 35).\n",
			devIDs, *tpSizeFlag, *ppStagesFlag)
		os.Exit(1)
	}
	if len(devIDs) == 1 {
		count, err := device.GetDeviceCount()
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
		if devIDs[0] < 0 || devIDs[0] >= count {
			fmt.Fprintf(os.Stderr, "error: CUDA device %d does not exist (the host has %d)\n", devIDs[0], count)
			os.Exit(1)
		}
		fmt.Printf("CUDA device: %d\n", devIDs[0])
	}

	fmt.Printf("Layers: %d (GPU: %d)\n", engineConfig.Layers, engineConfig.NumGPULayers)
	fmt.Printf("Vocab: %d\n", engineConfig.VocabSize)
	fmt.Printf("Dim: %d, Heads: %d, KV Heads: %d\n", engineConfig.Dim, engineConfig.Heads, engineConfig.KVHeads)

	fmt.Printf("\n=== Initializing CUDA Backend ===\n")

	go func() {
		mux := http.NewServeMux()
		mux.Handle("/metrics", promhttp.Handler())
		srv := &http.Server{
			Addr:              *metricsAddr,
			Handler:           mux,
			ReadHeaderTimeout: 5 * time.Second,
		}
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Log.Warn("Metrics server exited", "error", err)
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

	if *loraPath != "" {
		if err := e.LoadAdapter(*loraPath, "cli-lora"); err != nil {
			logger.Log.Warn("Failed to load LoRA adapter", "error", err)
		} else {
			fmt.Printf("Loaded LoRA adapter: %s\n", *loraPath)
		}
	}

	fmt.Printf("GPU Memory: %.1f MB\n", float64(device.CUDAAllocatedBytes())/1e6)

	// Initialize and start Arrow Flight Server
	flightServer := api.NewInferenceFlightServer(*flightAddr, e, tok)
	go func() {
		if err := flightServer.Serve(); err != nil {
			logger.Log.Error("Flight server failed", "error", err)
		}
	}()

	// OpenAI-compatible REST API. This is the surface documented in README.md
	// and docs/api.md; it was previously never started by any binary.
	apiSrv, err := startAPIServer(*apiAddr, device.CUDAAllocatedBytes, targetEngine, specEngine, tok)
	if err != nil {
		logger.Log.Error("failed to start REST API", "error", err)
	}
	if apiSrv != nil {
		defer apiSrv.Close()
		fmt.Println(apiSrv.describe())
	}

	promptTokens := tok.Encode(*prompt)
	fmt.Printf("Prompt tokens: %d\n", len(promptTokens))

	// VLM: If image provided, encode through vision encoder and prepend to prompt
	if *imagePath != "" {
		imgData, err := os.ReadFile(*imagePath)
		if err != nil {
			logger.Log.Warn("Failed to read image file", "error", err)
		} else {
			_ = base64.StdEncoding.EncodeToString(imgData)
			vlmArch := arch
			if vlmArch == "" || vlmArch == "unknown" {
				vlmArch = "clip"
			}
			devCtx := device.NewContext()
			defer devCtx.Free()
			vlmCfg := vlm.VLMConfig{
				Architecture: vlmArch,
				ImageSize:    224,
				PatchSize:    14,
				HiddenDim:    engineConfig.Dim,
			}
			vDecoder, err := vlm.NewVLMDecoder(devCtx, vlmCfg)
			if err != nil {
				logger.Log.Warn("VLM decoder not available", "arch", vlmArch, "error", err)
			} else {
				visionTensor, vErr := vDecoder.Decode(imgData)
				if vErr != nil {
					logger.Log.Warn("Vision encoding failed", "error", vErr)
				} else if visionTensor != nil {
					fmt.Printf("Vision features: %d patches encoded\n", visionTensor.Rows())
					promptTokens = append([]int{0}, promptTokens...)
				}
			}
		}
	}

	samplerConfig := engine.SamplerConfig{
		Temperature:      *temperature,
		TopK:             *topK,
		TopP:             *topP,
		RepPenalty:       *repPenalty,
		PresencePenalty:  *presencePenalty,
		FrequencyPenalty: *frequencyPenalty,
		MinP:             *minP,
		Seed:             *seed,
		StopTokens:       engineConfig.StopTokens,
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
