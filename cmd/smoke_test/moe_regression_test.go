//go:build darwin && metal

package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/engine"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
)

// MOEBaselineEntry stores MOE-specific test results
type MOEBaselineEntry struct {
	Prompt           string  `json:"prompt"`
	Tokens           []int   `json:"tokens"`
	GeneratedText    string  `json:"generated_text"`
	Perplexity       float64 `json:"perplexity"`
	TokensPerSecond  float64 `json:"tokens_per_second"`
	ExpertSelections []int   `json:"expert_selections"` // Which experts were selected
}

type MOEBaselineFile struct {
	ModelName      string                      `json:"model_name"`
	ModelType      string                      `json:"model_type"` // "nemotron", "nemotron-mini", "gpt-oss", "mixtral"
	Timestamp      string                      `json:"timestamp"`
	QuarrelVersion string                      `json:"quarrel_version"`
	Entries        map[string]MOEBaselineEntry `json:"entries"`
	Performance    MOEPerformanceMetrics       `json:"performance"`
}

type MOEPerformanceMetrics struct {
	AvgTokensPerSecond   float64 `json:"avg_tokens_per_second"`
	AvgMoELayerLatencyMs float64 `json:"avg_moe_layer_latency_ms"`
	AvgRoutingLatencyMs  float64 `json:"avg_routing_latency_ms"`
	MemoryPeakMB         float64 `json:"memory_peak_mb"`
}

// llamaCPPBaseline represents expected output from llama.cpp for comparison
type llamaCPPBaseline struct {
	ModelName string `json:"model_name"`
	Entries   map[string]struct {
		Tokens        []int   `json:"tokens"`
		GeneratedText string  `json:"generated_text"`
		Perplexity    float64 `json:"perplexity"`
	} `json:"entries"`
}

// standardTestPrompts for MOE models
var standardTestPrompts = []string{
	"The quick brown fox",
	"Hello, my name is",
	"In a hole in the ground there lived a hobbit.",
	"The capital of France is",
	"Once upon a time",
	"What is 2+2?",
}

// TestMOELLamaCPPComparison compares MOE model outputs with llama.cpp baselines
func TestMOELLamaCPPComparison(t *testing.T) {
	// Test against each MOE model
	testCases := []struct {
		name         string
		modelPath    string
		baselinePath string
		description  string
	}{
		{
			name:         "Nemotron-3-Nano",
			modelPath:    "/Users/rsd/.ollama/models/blobs/sha256-e7b273f9636059a689e3ddcab3716e4f65abe0143ac978e46673ad0e52d09efb",
			baselinePath: "cmd/smoke_test/baselines/llamacpp_nemotron_nano.json",
			description:  "128 experts, top-6 routing, 1 shared expert",
		},
		{
			name:         "Nemotron-Mini-4B",
			modelPath:    "/Users/rsd/.ollama/models/blobs/sha256-nemotron-mini-4b", // Placeholder path
			baselinePath: "cmd/smoke_test/baselines/llamacpp_nemotron_mini.json",
			description:  "128 experts, top-6 routing, smaller FFN",
		},
		{
			name:         "GPT-OSS",
			modelPath:    "/Users/rsd/.ollama/models/blobs/sha256-gpt-oss", // Placeholder path
			baselinePath: "cmd/smoke_test/baselines/llamacpp_gpt_oss.json",
			description:  "64 experts, top-4 routing, no shared experts",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Check if model exists
			if _, err := os.Stat(tc.modelPath); os.IsNotExist(err) {
				t.Skipf("Model not found at %s, skipping comparison test", tc.modelPath)
			}

			// Check if baseline exists
			baselineData, err := os.ReadFile(tc.baselinePath)
			if err != nil {
				t.Skipf("llama.cpp baseline not found at %s: %v", tc.baselinePath, err)
			}

			var llamaBaseline llamaCPPBaseline
			if err := json.Unmarshal(baselineData, &llamaBaseline); err != nil {
				t.Fatalf("Failed to parse llama.cpp baseline: %v", err)
			}

			t.Logf("Testing %s (%s)", tc.name, tc.description)

			conf := config.Default()
			e, err := engine.NewEngine(tc.modelPath, conf)
			if err != nil {
				t.Fatalf("Failed to initialize engine: %v", err)
			}
			defer e.Close()

			// Load tokenizer
			ggufFile, err := gguf.LoadFile(tc.modelPath)
			if err != nil {
				t.Fatalf("Failed to load GGUF: %v", err)
			}
			defer ggufFile.Close()

			tok, err := tokenizer.NewFromGGUF(ggufFile)
			if err != nil {
				t.Fatalf("Failed to create tokenizer: %v", err)
			}

			sampler := engine.SamplerConfig{Temperature: 0}

			// Compare outputs for each standard prompt
			for _, prompt := range standardTestPrompts {
				llamaEntry, ok := llamaBaseline.Entries[prompt]
				if !ok {
					t.Logf("No llama.cpp baseline for prompt: %q", prompt)
					continue
				}

				t.Logf("Comparing prompt: %q", prompt)

				inputTokens := tok.Encode(prompt)
				genLen := len(llamaEntry.Tokens)

				// Generate with Quarrel
				quarrelTokens, err := e.Infer(inputTokens, genLen, sampler)
				if err != nil {
					t.Errorf("Quarrel inference failed: %v", err)
					continue
				}

				quarrelText := tok.Decode(quarrelTokens)
				llamaText := llamaEntry.GeneratedText

				// Compare token sequences
				tokenMatch := compareTokenSequences(quarrelTokens, llamaEntry.Tokens)
				t.Logf("  Token match: %.1f%%", tokenMatch*100)

				// Check perplexity difference (allow 10% variance)
				ppl := calculatePerplexityForTokens(e, inputTokens, quarrelTokens)
				pplDiff := math.Abs(ppl - llamaEntry.Perplexity)
				pplPercentDiff := pplDiff / llamaEntry.Perplexity * 100
				t.Logf("  Perplexity: Quarrel=%.4f, llama.cpp=%.4f (diff=%.1f%%)",
					ppl, llamaEntry.Perplexity, pplPercentDiff)

				if tokenMatch < 0.8 {
					t.Errorf("  Low token match (%.1f%%) - outputs diverge significantly", tokenMatch*100)
				}
				if pplPercentDiff > 10.0 {
					t.Errorf("  Perplexity difference too high (%.1f%%)", pplPercentDiff)
				}
			}
		})
	}
}

// TestMOEPerformanceBenchmark benchmarks MOE model performance
func TestMOEPerformanceBenchmark(t *testing.T) {
	benchmarkCases := []struct {
		name      string
		modelPath string
		modelType string
	}{
		{
			name:      "Nemotron-3-Nano",
			modelPath: "/Users/rsd/.ollama/models/blobs/sha256-e7b273f9636059a689e3ddcab3716e4f65abe0143ac978e46673ad0e52d09efb",
			modelType: "nemotron",
		},
		{
			name:      "Nemotron-Mini-4B",
			modelPath: "/Users/rsd/.ollama/models/blobs/sha256-nemotron-mini-4b",
			modelType: "nemotron-mini",
		},
		{
			name:      "GPT-OSS",
			modelPath: "/Users/rsd/.ollama/models/blobs/sha256-gpt-oss",
			modelType: "gpt-oss",
		},
	}

	for _, tc := range benchmarkCases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := os.Stat(tc.modelPath); os.IsNotExist(err) {
				t.Skipf("Model not found at %s, skipping benchmark", tc.modelPath)
			}

			t.Logf("Benchmarking %s", tc.name)

			conf := config.Default()
			e, err := engine.NewEngine(tc.modelPath, conf)
			if err != nil {
				t.Fatalf("Failed to initialize engine: %v", err)
			}
			defer e.Close()

			ggufFile, err := gguf.LoadFile(tc.modelPath)
			if err != nil {
				t.Fatalf("Failed to load GGUF: %v", err)
			}
			defer ggufFile.Close()

			tok, err := tokenizer.NewFromGGUF(ggufFile)
			if err != nil {
				t.Fatalf("Failed to create tokenizer: %v", err)
			}

			sampler := engine.SamplerConfig{Temperature: 0}
			prompt := "The quick brown fox jumps over the lazy dog. This is a test of"
			inputTokens := tok.Encode(prompt)

			// Warmup
			_, _ = e.Infer(inputTokens, 5, sampler)

			// Benchmark
			numTokens := 50
			numRuns := 3

			var totalTime time.Duration
			var totalTokens int

			for i := 0; i < numRuns; i++ {
				start := time.Now()
				tokens, err := e.Infer(inputTokens, numTokens, sampler)
				elapsed := time.Since(start)

				if err != nil {
					t.Errorf("Inference failed: %v", err)
					continue
				}

				totalTime += elapsed
				totalTokens += len(tokens)
			}

			avgTime := totalTime / time.Duration(numRuns)
			tokensPerSec := float64(numTokens) / avgTime.Seconds()

			t.Logf("Performance Results:")
			t.Logf("  Average time: %v", avgTime)
			t.Logf("  Tokens/sec: %.2f", tokensPerSec)
			t.Logf("  Total memory: %d bytes", device.AllocatedBytes())

			// Save benchmark results
			result := MOEBaselineFile{
				ModelName:      tc.name,
				ModelType:      tc.modelType,
				Timestamp:      time.Now().Format(time.RFC3339),
				QuarrelVersion: "dev",
				Performance: MOEPerformanceMetrics{
					AvgTokensPerSecond: tokensPerSec,
					MemoryPeakMB:       float64(device.AllocatedBytes()) / (1024 * 1024),
				},
			}

			resultPath := fmt.Sprintf("cmd/smoke_test/baselines/quarrel_%s_perf.json", tc.modelType)
			data, _ := json.MarshalIndent(result, "", "  ")
			_ = os.MkdirAll(filepath.Dir(resultPath), 0755)
			_ = os.WriteFile(resultPath, data, 0644)

			t.Logf("Results saved to %s", resultPath)
		})
	}
}

// Helper functions

func compareTokenSequences(a, b []int) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	minLen := len(a)
	if len(b) < minLen {
		minLen = len(b)
	}

	matches := 0
	for i := 0; i < minLen; i++ {
		if a[i] == b[i] {
			matches++
		}
	}

	return float64(matches) / float64(minLen)
}

func calculatePerplexityForTokens(e *engine.Engine, inputTokens, generatedTokens []int) float64 {
	// Simplified perplexity calculation
	// In a real implementation, this would use the actual logits
	return 0.0
}
