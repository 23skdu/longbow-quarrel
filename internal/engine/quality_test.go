//go:build darwin && metal

package engine

import (
	"fmt"
	"testing"
)

func TestQualityEvaluator_BLEU(t *testing.T) {
	// Create a simple quality evaluator without tokenizer for basic testing
	eval := NewQualityEvaluatorSimple()

	// Test identical texts (should have perfect BLEU)
	candidate := "the cat sat on the mat"
	reference := "the cat sat on the mat"

	score := eval.CalculateBLEU(candidate, reference)

	// With identical texts, BLEU should be close to 1.0
	if score.BLEU4 < 0.9 {
		t.Errorf("Expected BLEU-4 close to 1.0 for identical texts, got %.4f", score.BLEU4)
	}

	// Test different texts
	candidate2 := "the dog ran in the park"
	reference2 := "a cat slept on a mat"

	score2 := eval.CalculateBLEU(candidate2, reference2)

	// Should be lower but not necessarily zero (depends on character overlap)
	// Just check it's a valid score between 0 and 1
	if score2.BLEU4 < 0.0 || score2.BLEU4 > 1.0 {
		t.Errorf("Expected BLEU-4 between 0 and 1 for different texts, got %.4f", score2.BLEU4)
	}

	// BLEU should be lower for different texts than identical texts
	if score2.BLEU4 >= score.BLEU4 {
		t.Errorf("Expected BLEU-4 to be lower for different texts (%.4f) than identical texts (%.4f)", score2.BLEU4, score.BLEU4)
	}
}

func TestQualityEvaluator_ROUGE(t *testing.T) {
	eval := NewQualityEvaluatorSimple()

	// Test identical texts
	candidate := "the quick brown fox"
	reference := "the quick brown fox"

	score := eval.CalculateROUGE(candidate, reference)

	if score.F1 < 0.9 {
		t.Errorf("Expected ROUGE F1 close to 1.0 for identical texts, got %.4f", score.F1)
	}

	// Test partial overlap
	candidate2 := "the quick brown fox jumps"
	reference2 := "the quick red fox runs"

	score2 := eval.CalculateROUGE(candidate2, reference2)

	if score2.F1 <= 0.0 {
		t.Errorf("Expected positive ROUGE F1 for partially overlapping texts, got %.4f", score2.F1)
	}
}

func TestQualityEvaluator_Perplexity(t *testing.T) {
	eval := NewQualityEvaluatorSimple()

	// Test with token sequence
	tokens := []int{1, 2, 3, 4, 5}

	result := eval.CalculatePerplexity(tokens)

	if result.Perplexity <= 0 {
		t.Errorf("Expected positive perplexity, got %.4f", result.Perplexity)
	}

	if result.TotalTokens != 4 { // tokens[1:] = 4 tokens
		t.Errorf("Expected 4 total tokens, got %d", result.TotalTokens)
	}
}

// Mock tokenizer for testing
type mockTokenizer struct{}

func (m *mockTokenizer) Encode(text string) []int {
	// Simple word-based tokenization for testing
	words := []string{}
	current := ""
	for _, r := range text {
		if r == ' ' {
			if current != "" {
				words = append(words, current)
				current = ""
			}
		} else {
			current += string(r)
		}
	}
	if current != "" {
		words = append(words, current)
	}

	// Convert words to token IDs (simple hash for testing)
	tokens := make([]int, len(words))
	for i, word := range words {
		// Simple hash function for testing
		hash := 0
		for _, r := range word {
			hash = hash*31 + int(r)
		}
		tokens[i] = hash % 1000 // Keep IDs small
	}

	return tokens
}

func (m *mockTokenizer) Decode(tokens []int) string {
	// For testing, just return a placeholder
	return fmt.Sprintf("decoded_%d_tokens", len(tokens))
}

func (m *mockTokenizer) VocabSize() int {
	return 1000
}

func (m *mockTokenizer) BOS() int {
	return 1
}

func (m *mockTokenizer) EOS() int {
	return 2
}
