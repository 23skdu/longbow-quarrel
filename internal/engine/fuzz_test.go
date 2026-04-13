//go:build darwin && metal

package engine

import (
	"encoding/json"
	"testing"

	conf "github.com/23skdu/longbow-quarrel/internal/config"
)

// Fuzz target for tokenizer input validation
func FuzzTokenizerInput(f *testing.F) {
	f.Add([]byte("hello world"))
	f.Add([]byte(""))
	f.Add([]byte("Hello, 世界! 🌍"))
	f.Add([]byte{0, 1, 2, 3, 4})
	f.Add([]byte("a very long string that might cause buffer overflows"))
	f.Add(make([]byte, 1000))

	f.Fuzz(func(t *testing.T, input []byte) {
		if len(input) > 10000 {
			t.Skip("Input too large")
		}
		_ = string(input)
	})
}

// Fuzz target for sampling parameters
func FuzzSamplingParams(f *testing.F) {
	f.Add(float64(0.0), 1, 0.9, 1.0)
	f.Add(float64(-1.0), 0, 0.0, 0.0)
	f.Add(float64(100.0), 100, 2.0, 10.0)
	f.Add(float64(0.5), 40, 0.95, 1.1)

	f.Fuzz(func(t *testing.T, temperature float64, topK int, topP float64, repPenalty float64) {
		if temperature < 0 || temperature > 100 || topK < 0 || topK > 1000 ||
			topP < 0 || topP > 100 || repPenalty < 0 || repPenalty > 100 {
			t.Skip("Unreasonable parameter values")
		}

		config := SamplerConfig{
			Temperature: temperature,
			TopK:        topK,
			TopP:        topP,
			RepPenalty:  repPenalty,
		}
		_ = config
	})
}

// Fuzz target for model input validation
func FuzzModelInput(f *testing.F) {
	f.Add([]byte{1, 2, 3})

	f.Fuzz(func(t *testing.T, tokenBytes []byte) {
		if len(tokenBytes) > 1000 {
			t.Skip("Token sequence too large")
		}

		tokens := make([]int, len(tokenBytes))
		for i, b := range tokenBytes {
			tokens[i] = int(b)
		}
		_ = tokens
	})
}

// Fuzz target for JSON configuration parsing
func FuzzConfigJSON(f *testing.F) {
	f.Add([]byte(`{"temperature": 0.7}`))
	f.Add([]byte(`{invalid json`))
	f.Add([]byte(``))
	f.Add([]byte(`{"temperature": "not a number"}`))
	f.Add([]byte(`{"temperature": null, "top_k": 40, "top_p": 0.95, "repetition_penalty": 1.1}`))
	f.Add([]byte(make([]byte, 1000)))

	f.Fuzz(func(t *testing.T, jsonInput []byte) {
		if len(jsonInput) > 10000 {
			t.Skip("JSON too large")
		}

		var config SamplerConfig
		_ = json.Unmarshal(jsonInput, &config)

		var engineConfig conf.Config
		_ = json.Unmarshal(jsonInput, &engineConfig)
	})
}

// Helper function to create test engine
func createTestEngine() Engine {
	// Use a small test model - would need to have one available
	// For now, return nil to indicate we need a test model
	// This would be implemented with an actual test model file
	return nil
}
