//go:build linux && cuda

package engine

import (
	"os"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
)

func getTestModelPath() string {
	if path := os.Getenv("QUARREL_TEST_MODEL"); path != "" {
		return path
	}
	return "smollm2.gguf"
}

func FuzzCUDAEngineCreation(f *testing.F) {
	if os.Getenv("SKIP_CUDA_TEST") != "" {
		f.Skip("Skipping CUDA fuzz test")
	}

	modelPath := getTestModelPath()
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		f.Skipf("Test model not found: %s", modelPath)
	}

	f.Fuzz(func(t *testing.T, kvCacheSize int) {
		if kvCacheSize < 1 || kvCacheSize > 8192 {
			t.Skip("Invalid KV cache size")
		}

		cfg := config.Config{
			KVCacheSize: kvCacheSize,
		}

		e, err := NewEngine(modelPath, cfg)
		if err != nil {
			t.Fatalf("Failed to create CUDA engine: %v", err)
		}
		defer e.Close()

		if e.CUDA == nil {
			t.Error("CUDA model is nil")
		}
	})
}

func FuzzCUDAInference(f *testing.F) {
	if os.Getenv("SKIP_CUDA_TEST") != "" {
		f.Skip("Skipping CUDA fuzz test")
	}

	modelPath := getTestModelPath()
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		f.Skipf("Test model not found: %s", modelPath)
	}

	cfg := config.Config{
		KVCacheSize: 2048,
	}

	e, err := NewEngine(modelPath, cfg)
	if err != nil {
		f.Fatalf("Failed to create CUDA engine: %v", err)
	}
	defer e.Close()

	f.Fuzz(func(t *testing.T, numTokens int, temp float64, topK int, topP float64, repPenalty float64) {
		if numTokens < 1 || numTokens > 100 {
			t.Skip("Invalid number of tokens")
		}
		if temp < 0 || temp > 2 {
			t.Skip("Invalid temperature")
		}
		if topK < 1 || topK > 100 {
			t.Skip("Invalid topK")
		}
		if topP < 0 || topP > 1 {
			t.Skip("Invalid topP")
		}
		if repPenalty < 0.5 || repPenalty > 2 {
			t.Skip("Invalid repetition penalty")
		}

		tokens := []int{1, 2, 3}
		samplerCfg := SamplerConfig{
			Temperature: temp,
			TopK:        topK,
			TopP:        topP,
			RepPenalty:  repPenalty,
		}

		_, err := e.Infer(tokens, numTokens, samplerCfg)
		if err != nil {
			t.Fatalf("Inference failed: %v", err)
		}
	})
}

func FuzzCUDASampler(f *testing.F) {
	f.Fuzz(func(t *testing.T, seed int64, temp float64, topK int, topP float64) {
		if temp < 0 || temp > 2 {
			t.Skip("Invalid temperature")
		}
		if topK < 1 || topK > 100 {
			t.Skip("Invalid topK")
		}
		if topP < 0 || topP > 1 {
			t.Skip("Invalid topP")
		}

		sampler := NewSampler(SamplerConfig{
			Temperature: temp,
			TopK:        topK,
			TopP:        topP,
			RepPenalty:  1.1,
			Seed:        seed,
		})

		logits := make([]float32, 50)
		for i := range logits {
			logits[i] = float32(i%10) - 5
		}

		token := sampler.Sample(logits)
		if token < 0 || token >= 50 {
			t.Errorf("Invalid token sampled: %d", token)
		}
	})
}

func FuzzCUDAPromptProcessing(f *testing.F) {
	if os.Getenv("SKIP_CUDA_TEST") != "" {
		f.Skip("Skipping CUDA fuzz test")
	}

	modelPath := getTestModelPath()
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		f.Skipf("Test model not found: %s", modelPath)
	}

	cfg := config.Config{
		KVCacheSize: 2048,
	}

	e, err := NewEngine(modelPath, cfg)
	if err != nil {
		f.Fatalf("Failed to create CUDA engine: %v", err)
	}
	defer e.Close()

	f.Fuzz(func(t *testing.T, token1, token2, token3 int) {
		tokens := []int{token1, token2, token3}

		for i, token := range tokens {
			if token < 0 {
				tokens[i] = 1
			}
		}

		samplerCfg := SamplerConfig{
			Temperature: 0.7,
			TopK:        40,
			TopP:        0.95,
			RepPenalty:  1.1,
		}

		_, err := e.Infer(tokens, 2, samplerCfg)
		if err != nil {
			t.Fatalf("Inference failed: %v", err)
		}
	})
}

func FuzzCUDAKVCacheHandling(f *testing.F) {
	if os.Getenv("SKIP_CUDA_TEST") != "" {
		f.Skip("Skipping CUDA fuzz test")
	}

	modelPath := getTestModelPath()
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		f.Skipf("Test model not found: %s", modelPath)
	}

	f.Fuzz(func(t *testing.T, cacheSize int) {
		if cacheSize < 128 || cacheSize > 4096 {
			t.Skip("Invalid cache size")
		}

		cfg := config.Config{
			KVCacheSize: cacheSize,
		}

		e, err := NewEngine(modelPath, cfg)
		if err != nil {
			t.Fatalf("Failed to create CUDA engine: %v", err)
		}
		defer e.Close()

		tokens := []int{1, 2, 3, 4, 5}
		samplerCfg := SamplerConfig{
			Temperature: 0.7,
			TopK:        40,
			TopP:        0.95,
			RepPenalty:  1.1,
		}

		_, err = e.Infer(tokens, 3, samplerCfg)
		if err != nil {
			t.Fatalf("Inference failed: %v", err)
		}
	})
}
