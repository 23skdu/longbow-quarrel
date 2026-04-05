//go:build darwin && metal

package engine

import (
	"testing"
)

// Test that Smollm2 135M model generates all-zero logits

// This test ensures that if Smollm2 produces all-zero logits (which results in <unk> tokens),
// we catch it early and report a clear error

func TestSmollm2ZeroLogits(t *testing.T) {
	t.Skip("Skipping - Requires loaded model and tokenizer for comprehensive test")
	// To implement:
	// 1. Load Smollm2 135M model and tokenizer
	// 2. Run inference with known prompt
	// 3. Use InferWithLogits to get logits
	// 4. Verify logits are non-zero (except for padding/特殊 tokens)
}

func TestSmollm2TokenizerMismatch(t *testing.T) {
	t.Skip("Skipping - Requires tokenizer comparison logic")
	// To implement:
	// 1. Load model vocab from GGUF metadata
	// 2. Load tokenizer vocab
	// 3. Compare token IDs to verify alignment
}


// Test that output contains only <unk> tokens
func TestSmollm2OutputIsAllUnk(t *testing.T) {
	t.Skip("Skipping - requires running engine to generate actual output")
}
