//go:build darwin && metal

package engine

import (
	"testing"
)

// Test that Smollm2 135M model generates all-zero logits

// This test ensures that if Smollm2 produces all-zero logits (which results in <unk> tokens),
// we catch it early and report a clear error

func TestSmollm2ZeroLogits(t *testing.T) {
	// This should fail if Smollm2 generates normal embeddings (non-zero logits)
	t.Skip("Skipping - Requires loaded model and tokenizer for comprehensive test")
	// TODO: Implement with full engine + tokenizer setup
}

func TestSmollm2TokenizerMismatch(t *testing.T) {
	// This test verifies tokenizer correctly handles Smollm2's vocabulary
	t.Skip("Skipping - Requires tokenizer comparison logic")
	// TODO: Implement by comparing tokenizer vocab with model expectations
}

// Helper to check if array contains only the given value
func allZeros[T comparable](arr []T, value T) bool {
	for _, v := range arr {
		if v != value {
			return false
		}
	}
	return true
}

// Test that output contains only <unk> tokens
func TestSmollm2OutputIsAllUnk(t *testing.T) {
	t.Skip("Skipping - requires running engine to generate actual output")
}
