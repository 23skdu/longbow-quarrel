//go:build darwin && metal

package engine

import (
	"fmt"
	"testing"
	"time"
)

// Fuzz target for tokenizer input validation
func FuzzTokenizerInput(f *testing.F) {
	// Seed with interesting inputs
	f.Add([]byte("hello world"))
	f.Add([]byte(""))
	f.Add([]byte("Hello, 世界! 🌍"))
	f.Add([]byte{0, 1, 2, 3, 4})
	f.Add([]byte("a very long string that might cause buffer overflows or other memory issues when processed by the tokenizer without proper bounds checking"))
	f.Add(make([]byte, 1000)) // Large input

	// Create engine for testing
	engine := createTestEngine()
	if engine == nil {
		f.Skip("Cannot create test engine")
	}
	defer engine.Close()

	f.Fuzz(func(t *testing.T, input []byte) {
		// Test tokenizer with fuzzed input
		prompt := string(input)

		// Should not panic
		defer func() {
			if r := recover(); r != nil {
				t.Errorf("Tokenizer panicked on input %q: %v", prompt, r)
			}
		}()

		// Encode should handle invalid input gracefully
		tokens := engine.Tokenizer().Encode(prompt)
		if len(tokens) > 0 {
			// Decode should not panic
			decoded := engine.Tokenizer().Decode(tokens)
			_ = decoded // Just ensure it doesn't panic
		}
	})
}

// Fuzz target for sampling parameters
func FuzzSamplingParams(f *testing.F) {
	// Seed with interesting parameter combinations
	f.Add(float64(0.0), 1, 0.9, 1.0)      // Normal values
	f.Add(float64(-1.0), 0, 0.0, 0.0)     // Edge cases
	f.Add(float64(100.0), 100, 2.0, 10.0) // Extreme values
	f.Add(float64(0.5), 40, 0.95, 1.1)    // Default values

	engine := createTestEngine()
	if engine == nil {
		f.Skip("Cannot create test engine")
	}
	defer engine.Close()

	f.Fuzz(func(t *testing.T, temperature float64, topK int, topP float64, repPenalty float64) {
		// Clamp to reasonable ranges to avoid infinite loops
		if temperature < 0 || temperature > 100 || topK < 0 || topK > 1000 ||
			topP < 0 || topP > 100 || repPenalty < 0 || repPenalty > 100 {
			t.Skip("Unreasonable parameter values")
		}

		// Should not panic
		defer func() {
			if r := recover(); r != nil {
				t.Errorf("Sampling panicked with params temp=%f, k=%d, p=%f, penalty=%f: %v",
					temperature, topK, topP, repPenalty, r)
			}
		}()

		// Create sampling config
		config := SamplingConfig{
			Temperature:       temperature,
			TopK:              topK,
			TopP:              topP,
			RepetitionPenalty: repPenalty,
		}

		// Test with a simple prompt
		tokens := engine.Tokenizer().Encode("Hello")
		if len(tokens) == 0 {
			t.Skip("Cannot encode test prompt")
		}

		// Should not panic during sampling
		start := time.Now()
		_ = engine.Sample(tokens, 1, config)
		duration := time.Since(start)

		// Sampling should complete in reasonable time
		if duration > 5*time.Second {
			t.Errorf("Sampling took too long: %v", duration)
		}
	})
}

// Fuzz target for model input validation
func FuzzModelInput(f *testing.F) {
	// Seed with interesting token sequences
	f.Add([]int{1, 2, 3})
	f.Add([]int{})                             // Empty
	f.Add([]int{999999, -1, 0})                // Invalid tokens
	f.Add(make([]int, 100))                    // Long sequence
	f.Add([]int{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}) // Repetitive

	engine := createTestEngine()
	if engine == nil {
		f.Skip("Cannot create test engine")
	}
	defer engine.Close()

	f.Fuzz(func(t *testing.T, tokens []int) {
		// Should not panic
		defer func() {
			if r := recover(); r != nil {
				t.Errorf("Model inference panicked on tokens %v: %v", tokens, r)
			}
		}()

		// Validate tokens
		for _, token := range tokens {
			if token < 0 || token >= engine.Tokenizer().VocabSize() {
				// Should handle out-of-vocab tokens gracefully
				return
			}
		}

		// Generate a few tokens - should not panic
		start := time.Now()
		_ = engine.Infer(tokens, 1, SamplingConfig{})
		duration := time.Since(start)

		// Should complete in reasonable time
		if duration > 10*time.Second {
			t.Errorf("Inference took too long: %v", duration)
		}
	})
}

// Fuzz target for JSON configuration parsing
func FuzzConfigJSON(f *testing.F) {
	// Seed with various JSON inputs
	f.Add([]byte(`{"temperature": 0.7}`))
	f.Add([]byte(`{invalid json`))
	f.Add([]byte(``)) // Empty
	f.Add([]byte(`{"temperature": "not a number"}`))
	f.Add([]byte(`{"temperature": null, "top_k": 40, "top_p": 0.95, "repetition_penalty": 1.1}`))
	f.Add([]byte(make([]byte, 1000))) // Large input

	f.Fuzz(func(t *testing.T, jsonInput []byte) {
		defer func() {
			if r := recover(); r != nil {
				t.Errorf("Config parsing panicked on JSON %q: %v", string(jsonInput), r)
			}
		}()

		// Try to parse as sampling config
		config := SamplingConfig{}
		_ = config.UnmarshalJSON(jsonInput) // Should not panic

		// Also test engine config parsing
		engineConfig := EngineConfig{}
		_ = engineConfig.UnmarshalJSON(jsonInput) // Should not panic
	})
}

// Helper function to create test engine
func createTestEngine() *Engine {
	// Use a small test model - would need to have one available
	// For now, return nil to indicate we need a test model
	// This would be implemented with an actual test model file
	return nil
}
