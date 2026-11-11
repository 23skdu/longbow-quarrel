package sampler

import (
	"testing"
)

func FuzzRegexGrammar(f *testing.F) {
	f.Add("a|b", "ab")
	f.Fuzz(func(t *testing.T, pattern, token string) {
		vocab := []string{"a", "b", "c", token}
		g, err := NewRegexGrammar(pattern, vocab)
		if err != nil {
			return
		}
		logits := make([]float32, len(vocab))
		for i := range logits {
			logits[i] = 1
		}
		if err := g.Apply(logits); err != nil {
			t.Fatalf("Apply failed: %v", err)
		}
		g.Update(token)
	})
}
