package engine

import (
	"testing"
)

func TestQualityEvaluator_Exhaustive(t *testing.T) {
	qe := NewQualityEvaluatorSimple()

	t.Run("Perplexity", func(t *testing.T) {
		res := qe.CalculatePerplexity([]int{1, 2, 3, 4, 5})
		if res.Perplexity <= 0 {
			t.Errorf("invalid perplexity: %f", res.Perplexity)
		}
		
		// Short case
		short := qe.CalculatePerplexity([]int{1})
		if short.Perplexity != 1.0 {
			t.Errorf("expected 1.0 for short seq, got %f", short.Perplexity)
		}
	})

	t.Run("BLEU_Simple", func(t *testing.T) {
		score := qe.CalculateBLEU("hello world", "hello world")
		if score.BLEU1 < 0.99 {
			t.Errorf("expected perfect BLEU for match, got %f", score.BLEU1)
		}

		scoreDiff := qe.CalculateBLEU("cat", "dog")
		if scoreDiff.BLEU1 > 0.5 {
			t.Errorf("expected low BLEU for mismatch, got %f", scoreDiff.BLEU1)
		}
	})

	t.Run("ROUGE_Simple", func(t *testing.T) {
		score := qe.CalculateROUGE("hello world", "hello world")
		if score.F1 < 0.99 {
			t.Errorf("expected perfect ROUGE for match, got %f", score.F1)
		}
		
		// Match logic hit
		_ = qe.CalculateROUGE("a b c", "a d c")
	})
}
