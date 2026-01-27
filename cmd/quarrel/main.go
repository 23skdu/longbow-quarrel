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
	kvCacheSize = flag.Int("kv-cache-size", 22, "KV cache size in MiB (default: 22)")
	benchmark   = flag.Bool("benchmark", false, "Run performance benchmark")

	// Sampling flags
	temperature      = flag.Float64("temp", 0.7, "Temperature for sampling (0.0 = greedy)")
	topK             = flag.Int("topk", 40, "Top-K sampling")
	topP             = flag.Float64("topp", 0.95, "Top-P (Nucleus) sampling")
	repPenalty       = flag.Float64("penalty", 1.1, "Repetition penalty")
	qualityMode      = flag.Bool("quality", false, "Enable advanced quality-guided sampling")
	streamOutput     = flag.Bool("stream", false, "Stream tokens as they are generated")
	chatML           = flag.Bool("chatml", false, "Wrap prompt in ChatML template")
	debugDequant     = flag.Bool("debug-dequant", false, "Enable dequantization debug dump")
	debugActivations = flag.Bool("debug-activations", false, "Enable layer-by-layer activation dumping")
	debugPerf        = flag.Bool("debug-perf", false, "Enable performance metric logging for kernels")
	gpu              = flag.Bool("gpu", false, "Use Metal GPU acceleration")
)

func main() {
	// Increase File Descriptor Limit (Metal Buffers might use FDs)
	var rLimit syscall.Rlimit
	if err := syscall.Getrlimit(syscall.RLIMIT_NOFILE, &rLimit); err == nil {
		rLimit.Cur = 10240
		if rLimit.Max < 10240 {
			rLimit.Max = 10240
		}
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
		logger.Log.Info("Resolved Ollama model", "original", *modelPath, "resolved", resolvedPath)
		*modelPath = resolvedPath
	} else {
		// Not an Ollama model, treat as direct path
		logger.Log.Info("Using direct model path", "path", *modelPath)
	}

	// Start Metrics Server
	go func() {
		http.Handle("/metrics", promhttp.Handler())
		logger.Log.Info("Metrics serving", "address", *metricsAddr+"/metrics")
		if err := http.ListenAndServe(*metricsAddr, nil); err != nil {
			logger.Log.Info("Metrics server error", "error", err)
		}
	}()

	// Signal handling
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	// Initialize Tokenizer
	logger.Log.Info("Loading tokenizer", "model", *modelPath)
	tok, err := tokenizer.New(*modelPath)
	if err != nil {
		log.Fatalf("Failed to initialize tokenizer: %v", err)
	}

	// Initialize Engine
	logger.Log.Info("Loading model", "model", *modelPath)
	engineConfig := config.Default()
	engineConfig.KVCacheSize = *kvCacheSize
	engineConfig.DebugActivations = *debugActivations

	e, err := engine.NewEngine(*modelPath, engineConfig)
	if err != nil {
		log.Fatalf("Failed to initialize engine: %v", err)
	}
	defer e.Close()

	// Parse ChatML Special Tokens from Tokenizer if possible, or use defaults for SmolLM2
	// <|im_start|> = 1
	// <|im_end|> = 2
	// \n = 198 (Standard GPT-2/Llama)
	const (
		ID_IM_START = 1
		ID_IM_END   = 2
		ID_NEWLINE  = 198
	)

	// Tokenize prompt
	var inputTokens []int

	if *chatML {
		// Manual ChatML Construction to avoid tokenizer splitting special tokens
		// <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n

		inputTokens = append(inputTokens, ID_IM_START)
		inputTokens = append(inputTokens, tok.Encode("user")...)
		inputTokens = append(inputTokens, ID_NEWLINE)

		inputTokens = append(inputTokens, tok.Encode(*prompt)...)

		inputTokens = append(inputTokens, ID_IM_END)
		inputTokens = append(inputTokens, ID_NEWLINE)

		inputTokens = append(inputTokens, ID_IM_START)
		inputTokens = append(inputTokens, tok.Encode("assistant")...)
		inputTokens = append(inputTokens, ID_NEWLINE)

	} else {
		// Raw prompt
		inputTokens = tok.Encode(*prompt)
		// Prepend BOS if needed? SmolLM2 might not need BOS for raw completion, but usually yes.
		// Let's prepend 1 just in case, unless it's ChatML where we explicitly added it.
		// Actually, standard Llama 2 adds BOS.
		// Let's add it.
		inputTokens = append([]int{1}, inputTokens...)
	}

	logger.Log.Info("Encoded tokens", "tokens", inputTokens, "length", len(inputTokens))

	logger.Log.Info("Starting inference", "num_tokens", *numTokens)

	doneChan := make(chan struct{})

	go func() {
		start := time.Now()

		samplerConfig := engine.SamplerConfig{
			Temperature:      *temperature,
			TopK:             *topK,
			TopP:             *topP,
			RepPenalty:       *repPenalty,
			Seed:             time.Now().UnixNano(),
			DebugActivations: *debugActivations,
			QualityMode:      *qualityMode,
		}

		logger.Log.Info("Sampling Config",
			"temp", *temperature,
			"top_k", *topK,
			"top_p", *topP,
			"penalty", *repPenalty,
			"quality_mode", *qualityMode,
			"stream", *streamOutput,
			"debug_activations", *debugActivations)

		var result []int
		var err error

		if *streamOutput {
			// Streaming mode: output tokens as they are generated
			fmt.Print("Streaming output: ")
			result, err = e.InferWithCallback(inputTokens, *numTokens, samplerConfig, func(token int) {
				decoded := tok.Decode([]int{token})
				fmt.Print(decoded)
				// Flush output immediately for streaming effect
				// Note: In Go, stdout is line-buffered by default, but we want immediate output
			})
			fmt.Println(" [END]") // Mark end of streaming
		} else {
			// Batch mode: collect all tokens then output
			result, err = e.Infer(inputTokens, *numTokens, samplerConfig)
		}
		if err != nil {
			logger.Log.Info("Inference error", "error", err)
		} else {
			duration := time.Since(start)
			tokensPerSec := float64(len(result)) / duration.Seconds()
			logger.Log.Info("Inference complete", "tokens", len(result), "duration", duration, "tps", tokensPerSec)

			// Debug print each token
			fmt.Print("Result Tokens Detail: ")
			for _, id := range result {
				decoded := tok.Decode([]int{id})
				fmt.Printf("[%d:'%s'] ", id, decoded)
			}
			fmt.Println()

			logger.Log.Info("Decoded Text", "text", tok.Decode(result))
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
