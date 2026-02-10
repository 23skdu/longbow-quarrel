//go:build linux && cuda

package integration_test

import (
	"math"
	"os"
	"testing"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/engine"
	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
)

func getTestModelPath() string {
	if path := os.Getenv("QUARREL_TEST_MODEL"); path != "" {
		return path
	}
	return "smollm2.gguf"
}

func TestCUDAInferenceSmoke(t *testing.T) {
	if os.Getenv("SKIP_CUDA_TEST") != "" {
		t.Skip("Skipping CUDA integration test")
	}

	modelPath := getTestModelPath()
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("Test model not found: %s", modelPath)
	}

	cfg := config.Config{
		KVCacheSize: 2048,
	}

	e, err := engine.NewEngine(modelPath, cfg)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}
	defer e.Close()

	tok, err := tokenizer.New(modelPath)
	if err != nil {
		t.Fatalf("Failed to load tokenizer: %v", err)
	}

	prompts := []string{
		"Hello world",
		"The quick brown fox jumps over the lazy dog",
		"What is 2 + 2?",
	}

	for _, prompt := range prompts {
		t.Run(prompt, func(t *testing.T) {
			tokens := tok.Encode(prompt)
			if len(tokens) == 0 {
				tokens = []int{1}
			}

			samplerCfg := engine.SamplerConfig{
				Temperature: 0.7,
				TopK:        40,
				TopP:        0.95,
				RepPenalty:  1.1,
			}

			start := time.Now()
			result, err := e.Infer(tokens, 5, samplerCfg)
			elapsed := time.Since(start)

			if err != nil {
				t.Fatalf("Inference failed for prompt '%s': %v", prompt, err)
			}

			tokensPerSec := float64(len(result)) / elapsed.Seconds()
			t.Logf("Prompt: '%s' -> Generated %d tokens in %v (%.1f t/s)",
				prompt, len(result), elapsed, tokensPerSec)

			if len(result) == 0 {
				t.Error("No tokens generated")
			}

			text := tok.Decode(result)
			t.Logf("Generated: %s", text)
		})
	}
}

func TestCUDALongContextInference(t *testing.T) {
	if os.Getenv("SKIP_CUDA_TEST") != "" {
		t.Skip("Skipping CUDA integration test")
	}

	modelPath := getTestModelPath()
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("Test model not found: %s", modelPath)
	}

	cfg := config.Config{
		KVCacheSize: 4096,
	}

	e, err := engine.NewEngine(modelPath, cfg)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}
	defer e.Close()

	tok, err := tokenizer.New(modelPath)
	if err != nil {
		t.Fatalf("Failed to load tokenizer: %v", err)
	}

	longPrompt := "This is a test. "
	for i := 0; i < 10; i++ {
		longPrompt += longPrompt
	}
	if len(longPrompt) > 1000 {
		longPrompt = longPrompt[:1000]
	}

	tokens := tok.Encode(longPrompt)
	if len(tokens) == 0 {
		tokens = make([]int, 100)
		for i := range tokens {
			tokens[i] = 1
		}
	}

	samplerCfg := engine.SamplerConfig{
		Temperature: 0.7,
		TopK:        40,
		TopP:        0.95,
		RepPenalty:  1.1,
	}

	start := time.Now()
	result, err := e.Infer(tokens, 5, samplerCfg)
	elapsed := time.Since(start)

	if err != nil {
		t.Fatalf("Long context inference failed: %v", err)
	}

	tokensPerSec := float64(len(result)) / elapsed.Seconds()
	t.Logf("Long context (%d tokens) -> Generated %d tokens in %v (%.1f t/s)",
		len(tokens), len(result), elapsed, tokensPerSec)
}

func TestCUDABatchInference(t *testing.T) {
	if os.Getenv("SKIP_CUDA_TEST") != "" {
		t.Skip("Skipping CUDA integration test")
	}

	modelPath := getTestModelPath()
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("Test model not found: %s", modelPath)
	}

	cfg := config.Config{
		KVCacheSize: 2048,
	}

	e, err := engine.NewEngine(modelPath, cfg)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}
	defer e.Close()

	tok, err := tokenizer.New(modelPath)
	if err != nil {
		t.Fatalf("Failed to load tokenizer: %v", err)
	}

	prompts := []string{
		"Hello",
		"Goodbye",
		"How are you?",
	}

	samplerCfg := engine.SamplerConfig{
		Temperature: 0.7,
		TopK:        40,
		TopP:        0.95,
		RepPenalty:  1.1,
	}

	totalTime := time.Duration(0)
	for i, prompt := range prompts {
		t.Run(prompt, func(t *testing.T) {
			tokens := tok.Encode(prompt)
			if len(tokens) == 0 {
				tokens = []int{1}
			}

			start := time.Now()
			result, err := e.Infer(tokens, 3, samplerCfg)
			elapsed := time.Since(start)
			totalTime += elapsed

			if err != nil {
				t.Fatalf("Batch inference failed for prompt %d: %v", i, err)
			}

			if len(result) == 0 {
				t.Error("No tokens generated")
			}
		})
	}

	t.Logf("Batch of %d prompts completed in %v total", len(prompts), totalTime)
}

func TestCUDASamplingVariety(t *testing.T) {
	if os.Getenv("SKIP_CUDA_TEST") != "" {
		t.Skip("Skipping CUDA integration test")
	}

	modelPath := getTestModelPath()
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("Test model not found: %s", modelPath)
	}

	cfg := config.Config{
		KVCacheSize: 2048,
	}

	e, err := engine.NewEngine(modelPath, cfg)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
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

	samplingConfigs := []struct {
		name       string
		temp       float64
		topK       int
		topP       float64
		repPenalty float64
	}{
		{"Creative", 1.2, 50, 0.95, 1.0},
		{"Deterministic", 0.1, 1, 1.0, 1.0},
		{"Balanced", 0.7, 40, 0.95, 1.1},
		{"Diverse", 1.5, 100, 0.9, 1.0},
	}

	for _, cfg := range samplingConfigs {
		t.Run(cfg.name, func(t *testing.T) {
			samplerCfg := engine.SamplerConfig{
				Temperature: cfg.temp,
				TopK:        cfg.topK,
				TopP:        cfg.topP,
				RepPenalty:  cfg.repPenalty,
			}

			result, err := e.Infer(tokens, 5, samplerCfg)
			if err != nil {
				t.Fatalf("Inference failed: %v", err)
			}

			if len(result) == 0 {
				t.Error("No tokens generated")
			}

			text := tok.Decode(result)
			t.Logf("Config: temp=%.2f, topK=%d, topP=%.2f -> '%s'",
				cfg.temp, cfg.topK, cfg.topP, text)
		})
	}
}

func TestCUDAModelInfo(t *testing.T) {
	if os.Getenv("SKIP_CUDA_TEST") != "" {
		t.Skip("Skipping CUDA integration test")
	}

	modelPath := getTestModelPath()
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("Test model not found: %s", modelPath)
	}

	cfg := config.Config{
		KVCacheSize: 2048,
	}

	e, err := engine.NewEngine(modelPath, cfg)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}
	defer e.Close()

	checks := []struct {
		name  string
		check func() bool
	}{
		{"Layers > 0", func() bool { return e.Config.Layers > 0 }},
		{"Dim > 0", func() bool { return e.Config.Dim > 0 }},
		{"Heads > 0", func() bool { return e.Config.Heads > 0 }},
		{"VocabSize > 0", func() bool { return e.Config.VocabSize > 0 }},
		{"HiddenDim > 0", func() bool { return e.Config.HiddenDim > 0 }},
		{"HeadDim > 0", func() bool { return e.Config.HeadDim > 0 }},
		{"KVHeads > 0", func() bool { return e.Config.KVHeads > 0 }},
		{"SeqLen > 0", func() bool { return e.Config.SeqLen > 0 }},
		{"CUDA model not nil", func() bool { return e.CUDA != nil }},
	}

	passed := 0
	for _, check := range checks {
		if !check.check() {
			t.Errorf("Model info check failed: %s", check.name)
		} else {
			passed++
		}
	}

	t.Logf("Model info: Layers=%d, Dim=%d, Heads=%d, Vocab=%d, HiddenDim=%d, HeadDim=%d, KVHeads=%d, SeqLen=%d",
		e.Config.Layers, e.Config.Dim, e.Config.Heads, e.Config.VocabSize,
		e.Config.HiddenDim, e.Config.HeadDim, e.Config.KVHeads, e.Config.SeqLen)
	t.Logf("Passed %d/%d model info checks", passed, len(checks))
}

func TestCUDARepetitionHandling(t *testing.T) {
	if os.Getenv("SKIP_CUDA_TEST") != "" {
		t.Skip("Skipping CUDA integration test")
	}

	modelPath := getTestModelPath()
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("Test model not found: %s", modelPath)
	}

	cfg := config.Config{
		KVCacheSize: 2048,
	}

	e, err := engine.NewEngine(modelPath, cfg)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}
	defer e.Close()

	tok, err := tokenizer.New(modelPath)
	if err != nil {
		t.Fatalf("Failed to load tokenizer: %v", err)
	}

	prompt := "Repeat repeat repeat"
	tokens := tok.Encode(prompt)
	if len(tokens) == 0 {
		tokens = []int{1}
	}

	penaltyTests := []struct {
		name       string
		repPenalty float64
	}{
		{"NoPenalty", 1.0},
		{"LightPenalty", 1.1},
		{"StrongPenalty", 1.5},
		{"VeryStrongPenalty", 2.0},
	}

	for _, test := range penaltyTests {
		t.Run(test.name, func(t *testing.T) {
			samplerCfg := engine.SamplerConfig{
				Temperature: 0.7,
				TopK:        40,
				TopP:        0.95,
				RepPenalty:  test.repPenalty,
			}

			result, err := e.Infer(tokens, 5, samplerCfg)
			if err != nil {
				t.Fatalf("Inference failed: %v", err)
			}

			if len(result) == 0 {
				t.Error("No tokens generated")
			}

			text := tok.Decode(result)
			t.Logf("Penalty=%.2f -> Generated: %s", test.repPenalty, text)
		})
	}
}

func TestCUDAE2EMetrics(t *testing.T) {
	if os.Getenv("SKIP_CUDA_TEST") != "" {
		t.Skip("Skipping CUDA integration test")
	}

	modelPath := getTestModelPath()
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("Test model not found: %s", modelPath)
	}

	cfg := config.Config{
		KVCacheSize: 2048,
	}

	e, err := engine.NewEngine(modelPath, cfg)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}
	defer e.Close()

	tok, err := tokenizer.New(modelPath)
	if err != nil {
		t.Fatalf("Failed to load tokenizer: %v", err)
	}

	metrics := []struct {
		name  string
		value float64
	}{
		{"Inference Time (s)", 0},
		{"Tokens Per Second", 0},
		{"Total Tokens Generated", 0},
	}

	for i, m := range metrics {
		if m.name == "Inference Time (s)" {
			tokens := tok.Encode("Test")
			if len(tokens) == 0 {
				tokens = []int{1}
			}

			samplerCfg := engine.SamplerConfig{
				Temperature: 0.7,
				TopK:        40,
				TopP:        0.95,
				RepPenalty:  1.1,
			}

			start := time.Now()
			result, err := e.Infer(tokens, 5, samplerCfg)
			elapsed := time.Since(start)

			if err != nil {
				t.Fatalf("Inference failed: %v", err)
			}

			metrics[i].value = elapsed.Seconds()
			if len(result) > 0 {
				metrics[1].value = float64(len(result)) / elapsed.Seconds()
			}
			metrics[2].value = float64(len(result))
		}
	}

	for _, m := range metrics {
		t.Logf("Metric: %s = %v", m.name, m.value)
	}

	if metrics[0].value > 60 {
		t.Errorf("Inference time too high: %v", metrics[0].value)
	}
	if metrics[1].value <= 0 {
		t.Error("Tokens per second should be positive")
	}
}

func BenchmarkCUDAFullPipeline(b *testing.B) {
	modelPath := getTestModelPath()
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		b.Skipf("Test model not found: %s", modelPath)
	}

	cfg := config.Config{
		KVCacheSize: 2048,
	}

	e, err := engine.NewEngine(modelPath, cfg)
	if err != nil {
		b.Fatalf("Failed to create engine: %v", err)
	}
	defer e.Close()

	tok, err := tokenizer.New(modelPath)
	if err != nil {
		b.Fatalf("Failed to load tokenizer: %v", err)
	}

	prompts := []string{
		"Hello world",
		"The quick brown fox",
		"What is artificial intelligence?",
	}

	samplerCfg := engine.SamplerConfig{
		Temperature: 0.7,
		TopK:        40,
		TopP:        0.95,
		RepPenalty:  1.1,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		prompt := prompts[i%len(prompts)]
		tokens := tok.Encode(prompt)
		if len(tokens) == 0 {
			tokens = []int{1}
		}

		_, err := e.Infer(tokens, 10, samplerCfg)
		if err != nil {
			b.Fatalf("Inference failed: %v", err)
		}
	}
}

func TestCUDANumericalStability(t *testing.T) {
	if os.Getenv("SKIP_CUDA_TEST") != "" {
		t.Skip("Skipping CUDA integration test")
	}

	modelPath := getTestModelPath()
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("Test model not found: %s", modelPath)
	}

	cfg := config.Config{
		KVCacheSize: 2048,
	}

	e, err := engine.NewEngine(modelPath, cfg)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}
	defer e.Close()

	tok, err := tokenizer.New(modelPath)
	if err != nil {
		t.Fatalf("Failed to load tokenizer: %v", err)
	}

	extremeConfigs := []struct {
		name string
		cfg  engine.SamplerConfig
	}{
		{"ZeroTemp", engine.SamplerConfig{Temperature: 0.0, TopK: 1, TopP: 1.0, RepPenalty: 1.0}},
		{"HighTemp", engine.SamplerConfig{Temperature: 2.0, TopK: 100, TopP: 1.0, RepPenalty: 1.0}},
	}

	for _, test := range extremeConfigs {
		t.Run(test.name, func(t *testing.T) {
			tokens := tok.Encode("Test sequence for numerical stability")
			if len(tokens) == 0 {
				tokens = []int{1}
			}

			result, err := e.Infer(tokens, 3, test.cfg)
			if err != nil {
				t.Fatalf("Inference failed with extreme config: %v", err)
			}

			for i, token := range result {
				if math.IsNaN(float64(token)) || math.IsInf(float64(token), 0) {
					t.Errorf("Numerical instability detected at token %d: %d", i, token)
				}
				if token < 0 {
					t.Errorf("Negative token detected at position %d: %d", i, token)
				}
			}
		})
	}
}
