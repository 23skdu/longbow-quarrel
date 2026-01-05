//go:build darwin && metal

package engine

import (
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

// TestRoPEFrequency_Mistral verifies that Mistral model's RoPE frequency is correctly extracted
// and that the model works with the correct theta value.
func TestRoPEFrequency_Mistral(t *testing.T) {
	mistralPath := "/Users/rsd/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f"

	// Load GGUF file
	file, err := gguf.LoadFile(mistralPath)
	if err != nil {
		t.Skipf("Mistral model not found: %v", err)
	}
	defer file.Close()

	// Extract RoPE frequency base
	var ropeFreqBase float64
	if val, ok := file.KV["llama.rope.freq_base"].(float32); ok {
		ropeFreqBase = float64(val)
	} else if val, ok := file.KV["llama.rope.freq_base"].(float64); ok {
		ropeFreqBase = val
	} else if val, ok := file.KV["llama.rope.freq_base"].(uint32); ok {
		ropeFreqBase = float64(val)
	} else {
		t.Logf("Available KV keys: %v", getKeys(file.KV))
		t.Fatalf("Could not extract llama.rope.freq_base from metadata")
	}

	t.Logf("Mistral RoPE freq_base from GGUF: %.0f", ropeFreqBase)

	// Verify it's one of the expected values
	if ropeFreqBase != 10000.0 && ropeFreqBase != 1000000.0 {
		t.Errorf("Unexpected RoPE freq_base: %.0f (expected 10000 or 1000000)", ropeFreqBase)
	}

	// Test that our config uses the correct value
	engine, err := NewEngine(mistralPath, false)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}

	engineTheta := float64(engine.Config.RopeTheta)
	if engineTheta != ropeFreqBase {
		t.Errorf("Engine RopeTheta mismatch: got %.0f, want %.0f", engineTheta, ropeFreqBase)
	}

	t.Logf("✓ Engine correctly configured with RoPE theta: %.0f", engineTheta)
}

func getKeys(m map[string]interface{}) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}
