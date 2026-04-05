//go:build darwin && metal

package engine

import (
	"os"
	"sync"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
)

func FuzzEngineHotSwap(f *testing.F) {
	// Setup dummy models
	modelPath1 := "test_model_fuzz_1.gguf"
	if err := generateTestGGUF(modelPath1); err != nil {
		f.Fatalf("Failed to generate test GGUF 1: %v", err)
	}
	defer os.Remove(modelPath1)

	modelPath2 := "test_model_fuzz_2.gguf"
	if err := generateTestGGUF(modelPath2); err != nil {
		f.Fatalf("Failed to generate test GGUF 2: %v", err)
	}
	defer os.Remove(modelPath2)

	f.Add(uint(10), uint(5)) // iterations, concurrent inferences
	f.Add(uint(50), uint(10))

	f.Fuzz(func(t *testing.T, iterations uint, concurrency uint) {
		if iterations > 100 {
			iterations = 100
		}
		if concurrency > 50 {
			concurrency = 50
		}
		if concurrency == 0 {
			concurrency = 1
		}

		conf := config.Default()
		conf.KVCacheSize = 256

		e, err := NewEngine(modelPath1, conf)
		if err != nil {
			t.Fatalf("Failed to create engine: %v", err)
		}
		me := e.(*metalEngine)
		defer me.ctx.Free()

		var wg sync.WaitGroup

		for i := uint(0); i < iterations; i++ {
			// Launch concurrent inferences
			for c := uint(0); c < concurrency; c++ {
				wg.Add(1)
				go func() {
					defer wg.Done()
					inputTokens := []int{1, 2, 3}
					sampler := SamplerConfig{Temperature: 0}
					e.Infer(inputTokens, 5, sampler)
				}()
			}

			// Perform swap interchangeably
			path := modelPath1
			if i%2 == 0 {
				path = modelPath2
			}

			err := e.SwapModel(path, conf)
			if err != nil {
				t.Fatalf("SwapModel failed: %v", err)
			}
		}

		wg.Wait()
	})
}
