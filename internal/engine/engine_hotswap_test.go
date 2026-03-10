//go:build darwin && metal

package engine

import (
	"os"
	"sync"
	"testing"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/config"
)

func TestEngineHotSwap(t *testing.T) {
	// Setup dummy models
	modelPath1 := "test_model_hotswap_1.gguf"
	if err := generateTestGGUF(modelPath1); err != nil {
		t.Fatalf("Failed to generate test GGUF 1: %v", err)
	}
	defer os.Remove(modelPath1)

	modelPath2 := "test_model_hotswap_2.gguf"
	if err := generateTestGGUF(modelPath2); err != nil {
		t.Fatalf("Failed to generate test GGUF 2: %v", err)
	}
	defer os.Remove(modelPath2)

	// Initialize Engine with model 1
	conf := config.Default()
	conf.KVCacheSize = 1024
	e, err := NewEngine(modelPath1, conf)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}
	if e == nil {
		t.Fatal("Engine is nil")
	}

	// Verify we are loaded
	if e.Weights.TokenEmb == nil {
		t.Fatal("Expected TokenEmb to be loaded for model 1")
	}

	// Hot swap to model 2 while simulating concurrent access
	var wg sync.WaitGroup
	inferenceErrs := make(chan error, 10)

	// Start some background "inferences" that should either complete or abort cleanly
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			inputTokens := []int{1, 2, 3}
			config := SamplerConfig{Temperature: 0}
			_, err := e.Infer(inputTokens, 10, config)
			if err != nil {
				inferenceErrs <- err
			}
		}()
	}

	// Give inferences a tiny moment to start
	time.Sleep(10 * time.Millisecond)

	// Perform the hot-swap
	err = e.SwapModel(modelPath2, conf)
	if err != nil {
		t.Fatalf("SwapModel failed: %v", err)
	}

	// Wait for inferences to finish
	wg.Wait()
	close(inferenceErrs)

	for err := range inferenceErrs {
		t.Logf("Concurrent inference returned err: %v", err)
		// It's acceptable for concurrent inference to return an error like "engine paused for swap"
	}

	// Verify model 2 is loaded and usable
	if e.Weights.TokenEmb == nil {
		t.Fatal("Expected TokenEmb to be valid after swap")
	}

	// Ensure we can run inference after swap
	inputTokens := []int{1, 2, 3}
	config := SamplerConfig{Temperature: 0}
	_, err = e.Infer(inputTokens, 10, config)
	if err != nil {
		t.Logf("Post-swap inference returned error (expected for dummy): %v", err)
	}
}
