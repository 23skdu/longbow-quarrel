package sampler

import (
	"testing"
)

func TestGrammar_Basic(t *testing.T) {
	t.Run("Initialize", func(t *testing.T) {
		// Mock a simple grammar rule
		_ = &Grammar{Active: true}
	})
	
	t.Run("ApplyMask", func(t *testing.T) {
		g := NewJSONGrammar([]string{"{", "}", "a"})
		logits := []float32{0.1, 0.2, 0.3}
		err := g.Apply(logits)
		if err != nil {
			t.Errorf("Apply failed: %v", err)
		}
		// In JSON grammar empty stack, only '{' or '[' allowed.
		// '{' is index 0. Index 2 ('a') should be masked.
		if logits[2] > -1e8 {
			t.Errorf("expected 'a' to be masked, got %f", logits[2])
		}
	})
}
