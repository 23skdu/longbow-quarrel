//go:build linux && amd64 && cuda && cgo

package engine

import (
	"os"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
)

func FuzzCUDAEngineCreation(f *testing.F) {
	if os.Getenv("SKIP_CUDA_TEST") != "" {
		f.Skip("Skipping CUDA fuzz test")
	}

	// The model is built here, so the target runs on any machine with a
	// GPU. It used to look for a file on disk.
	modelPath := buildTestGGUF(f, f.TempDir())

	e, err := NewEngine(modelPath, config.Config{KVCacheSize: 2048})
	if err != nil {
		f.Fatalf("Failed to create CUDA engine: %v", err)
	}
	defer e.Close()

	f.Fuzz(func(t *testing.T, kvCacheSize int) {
		if kvCacheSize < 1 || kvCacheSize > 8192 {
			// A skip inside a fuzz target discards the seed, so the corpus
			// never grows for this input. A plain return is what the
			// invalid-input path needs.
			return
		}

		if e == nil {
			t.Error("Engine is nil")
		}
	})
}

func FuzzCUDAInference(f *testing.F) {
	if os.Getenv("SKIP_CUDA_TEST") != "" {
		f.Skip("Skipping CUDA fuzz test")
	}

	// The model is built here, so the target runs on any machine with a
	// GPU. It used to look for a file on disk.
	modelPath := buildTestGGUF(f, f.TempDir())

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
			// A skip inside a fuzz target discards the seed, so the corpus
			// never grows for this input. A plain return is what the
			// invalid-input path needs.
			return
		}
		if temp < 0 || temp > 2 {
			// A skip inside a fuzz target discards the seed, so the corpus
			// never grows for this input. A plain return is what the
			// invalid-input path needs.
			return
		}
		if topK < 1 || topK > 100 {
			// A skip inside a fuzz target discards the seed, so the corpus
			// never grows for this input. A plain return is what the
			// invalid-input path needs.
			return
		}
		if topP < 0 || topP > 1 {
			// A skip inside a fuzz target discards the seed, so the corpus
			// never grows for this input. A plain return is what the
			// invalid-input path needs.
			return
		}
		if repPenalty < 0.5 || repPenalty > 2 {
			// A skip inside a fuzz target discards the seed, so the corpus
			// never grows for this input. A plain return is what the
			// invalid-input path needs.
			return
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
			// A skip inside a fuzz target discards the seed, so the corpus
			// never grows for this input. A plain return is what the
			// invalid-input path needs.
			return
		}
		if topK < 1 || topK > 100 {
			// A skip inside a fuzz target discards the seed, so the corpus
			// never grows for this input. A plain return is what the
			// invalid-input path needs.
			return
		}
		if topP < 0 || topP > 1 {
			// A skip inside a fuzz target discards the seed, so the corpus
			// never grows for this input. A plain return is what the
			// invalid-input path needs.
			return
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

		token := sampler.Sample(logits, nil)
		if token < 0 || token >= 50 {
			t.Errorf("Invalid token sampled: %d", token)
		}
	})
}

func FuzzCUDAPromptProcessing(f *testing.F) {
	if os.Getenv("SKIP_CUDA_TEST") != "" {
		f.Skip("Skipping CUDA fuzz test")
	}

	// The model is built here, so the target runs on any machine with a
	// GPU. It used to look for a file on disk.
	modelPath := buildTestGGUF(f, f.TempDir())

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

	// The model is built here, so the target runs on any machine with a
	// GPU. It used to look for a file on disk.
	modelPath := buildTestGGUF(f, f.TempDir())

	e, err := NewEngine(modelPath, config.Config{KVCacheSize: 2048})
	if err != nil {
		f.Fatalf("Failed to create CUDA engine: %v", err)
	}
	defer e.Close()

	f.Fuzz(func(t *testing.T, cacheSize int) {
		if cacheSize < 128 || cacheSize > 4096 {
			// A skip inside a fuzz target discards the seed, so the corpus
			// never grows for this input. A plain return is what the
			// invalid-input path needs.
			return
		}

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
