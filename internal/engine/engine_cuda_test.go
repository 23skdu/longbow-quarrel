//go:build linux && cuda

package engine

import (
	"math"
	"os"
	"testing"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
)

func TestCUDAEngineCreation(t *testing.T) {
	if os.Getenv("SKIP_CUDA_TEST") != "" {
		t.Skip("Skipping CUDA test")
	}

	modelPath := getTestModelPath()
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("Test model not found: %s", modelPath)
	}

	cfg := config.Config{
		KVCacheSize: 2048,
	}

	e, err := NewEngine(modelPath, cfg)
	if err != nil {
		t.Fatalf("Failed to create CUDA engine: %v", err)
	}
	defer e.Close()

	if e.CUDA == nil {
		t.Error("CUDA model is nil")
	}

	if e.Config.Layers <= 0 {
		t.Error("Invalid number of layers")
	}

	t.Logf("Created CUDA engine for %s with %d layers", modelPath, e.Config.Layers)
}

func TestCUDAInference(t *testing.T) {
	if os.Getenv("SKIP_CUDA_TEST") != "" {
		t.Skip("Skipping CUDA test")
	}

	modelPath := getTestModelPath()
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("Test model not found: %s", modelPath)
	}

	cfg := config.Config{
		KVCacheSize: 2048,
	}

	e, err := NewEngine(modelPath, cfg)
	if err != nil {
		t.Fatalf("Failed to create CUDA engine: %v", err)
	}
	defer e.Close()

	tok, err := tokenizer.New(modelPath)
	if err != nil {
		t.Fatalf("Failed to load tokenizer: %v", err)
	}

	prompt := "Hello"
	tokens := tok.Encode(prompt)
	if len(tokens) == 0 {
		tokens = []int{1} // Use BOS token if encoding fails
	}

	samplerCfg := SamplerConfig{
		Temperature: 0.7,
		TopK:        40,
		TopP:        0.95,
		RepPenalty:  1.1,
	}

	start := time.Now()
	result, err := e.Infer(tokens, 3, samplerCfg)
	elapsed := time.Since(start)

	if err != nil {
		t.Fatalf("Inference failed: %v", err)
	}

	tokensPerSec := float64(len(result)) / elapsed.Seconds()
	t.Logf("Generated %d tokens in %v (%.1f t/s)", len(result), elapsed, tokensPerSec)

	if len(result) == 0 {
		t.Error("No tokens generated")
	}

	text := tok.Decode(result)
	t.Logf("Generated text: %s", text)
}

func TestCUDASampler(t *testing.T) {
	if os.Getenv("SKIP_CUDA_TEST") != "" {
		t.Skip("Skipping CUDA test")
	}

	sampler := NewSampler(SamplerConfig{
		Temperature: 1.0,
		TopK:        10,
		TopP:        0.9,
		RepPenalty:  1.0,
		Seed:        42,
	})

	logits := make([]float32, 100)
	for i := range logits {
		logits[i] = float32(i) - 50
	}

	token := sampler.Sample(logits)
	if token < 0 || token >= 100 {
		t.Errorf("Invalid token sampled: %d", token)
	}

	t.Logf("Sampled token: %d", token)

	sampler2 := NewSampler(SamplerConfig{
		Temperature: 1.0,
		TopK:        10,
		TopP:        0.9,
		RepPenalty:  1.0,
		Seed:        42,
	})
	token2 := sampler2.Sample(logits)

	if token != token2 {
		t.Error("Seeded sampling not reproducible")
	}
}

func TestCUDAKVCache(t *testing.T) {
	if os.Getenv("SKIP_CUDA_TEST") != "" {
		t.Skip("Skipping CUDA test")
	}

	modelPath := getTestModelPath()
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("Test model not found: %s", modelPath)
	}

	cfg := config.Config{
		KVCacheSize: 512,
	}

	e, err := NewEngine(modelPath, cfg)
	if err != nil {
		t.Fatalf("Failed to create CUDA engine: %v", err)
	}
	defer e.Close()

	if e.CUDA == nil {
		t.Fatal("CUDA model is nil")
	}

	if e.CUDA.KCache == nil {
		t.Log("KV cache not allocated (expected for small models)")
	}
}

func TestCUDAPromptProcessing(t *testing.T) {
	if os.Getenv("SKIP_CUDA_TEST") != "" {
		t.Skip("Skipping CUDA test")
	}

	modelPath := getTestModelPath()
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("Test model not found: %s", modelPath)
	}

	cfg := config.Config{
		KVCacheSize: 2048,
	}

	e, err := NewEngine(modelPath, cfg)
	if err != nil {
		t.Fatalf("Failed to create CUDA engine: %v", err)
	}
	defer e.Close()

	tok, err := tokenizer.New(modelPath)
	if err != nil {
		t.Fatalf("Failed to load tokenizer: %v", err)
	}

	testPrompts := []string{
		"Hello",
		"The quick brown fox",
		"What is the capital of France?",
	}

	for _, prompt := range testPrompts {
		t.Run(prompt, func(t *testing.T) {
			tokens := tok.Encode(prompt)
			if len(tokens) == 0 {
				tokens = []int{1}
			}

			samplerCfg := SamplerConfig{
				Temperature: 0.7,
				TopK:        40,
				TopP:        0.95,
				RepPenalty:  1.1,
			}

			result, err := e.Infer(tokens, 2, samplerCfg)
			if err != nil {
				t.Fatalf("Inference failed: %v", err)
			}

			if len(result) == 0 {
				t.Error("No tokens generated")
			}
		})
	}
}

func TestCUDAOutputValidation(t *testing.T) {
	if os.Getenv("SKIP_CUDA_TEST") != "" {
		t.Skip("Skipping CUDA test")
	}

	modelPath := getTestModelPath()
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("Test model not found: %s", modelPath)
	}

	cfg := config.Config{
		KVCacheSize: 2048,
	}

	e, err := NewEngine(modelPath, cfg)
	if err != nil {
		t.Fatalf("Failed to create CUDA engine: %v", err)
	}
	defer e.Close()

	tok, err := tokenizer.New(modelPath)
	if err != nil {
		t.Fatalf("Failed to load tokenizer: %v", err)
	}

	prompt := "Test"
	tokens := tok.Encode(prompt)
	if len(tokens) == 0 {
		tokens = []int{1}
	}

	samplerCfg := SamplerConfig{
		Temperature: 0.7,
		TopK:        50,
		TopP:        1.0,
		RepPenalty:  1.0,
	}

	result, err := e.Infer(tokens, 5, samplerCfg)
	if err != nil {
		t.Fatalf("Inference failed: %v", err)
	}

	for i, token := range result {
		if token < 0 {
			t.Errorf("Negative token at position %d: %d", i, token)
		}
		if math.IsNaN(float64(token)) || math.IsInf(float64(token), 0) {
			t.Errorf("Invalid token at position %d: %d", i, token)
		}
	}

	t.Logf("Output validation passed for %d tokens", len(result))
}

func TestCUDAEngineConfig(t *testing.T) {
	if os.Getenv("SKIP_CUDA_TEST") != "" {
		t.Skip("Skipping CUDA test")
	}

	testConfigs := []config.Config{
		{KVCacheSize: 512},
		{KVCacheSize: 1024},
		{KVCacheSize: 2048},
	}

	modelPath := getTestModelPath()
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("Test model not found: %s", modelPath)
	}

	for _, cfg := range testConfigs {
		t.Run("", func(t *testing.T) {
			e, err := NewEngine(modelPath, cfg)
			if err != nil {
				t.Fatalf("Failed to create engine with KVCacheSize=%d: %v", cfg.KVCacheSize, err)
			}
			defer e.Close()

			if e.Config.KVCacheSize != cfg.KVCacheSize {
				t.Errorf("KVCacheSize mismatch: got %d, want %d", e.Config.KVCacheSize, cfg.KVCacheSize)
			}
		})
	}
}

func BenchmarkCUDAInference(b *testing.B) {
	modelPath := getTestModelPath()
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		b.Skipf("Test model not found: %s", modelPath)
	}

	cfg := config.Config{
		KVCacheSize: 2048,
	}

	e, err := NewEngine(modelPath, cfg)
	if err != nil {
		b.Fatalf("Failed to create CUDA engine: %v", err)
	}
	defer e.Close()

	tok, err := tokenizer.New(modelPath)
	if err != nil {
		b.Fatalf("Failed to load tokenizer: %v", err)
	}

	prompt := "Benchmark test prompt for performance measurement"
	tokens := tok.Encode(prompt)
	if len(tokens) == 0 {
		tokens = []int{1}
	}

	samplerCfg := SamplerConfig{
		Temperature: 0.7,
		TopK:        40,
		TopP:        0.95,
		RepPenalty:  1.1,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := e.Infer(tokens, 10, samplerCfg)
		if err != nil {
			b.Fatalf("Inference failed: %v", err)
		}
	}
}
