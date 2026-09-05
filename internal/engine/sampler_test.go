package engine

import (
	"math"
	"testing"
)

func TestSampler_Greedy(t *testing.T) {
	s := NewSampler(SamplerConfig{Temperature: 0})

	// Tokens: 0, 1, 2, 3
	// Logits: 1.0, 5.0, 2.0, 0.5
	logits := []float32{1.0, 5.0, 2.0, 0.5}

	val := s.Sample(logits, nil)
	if val != 1 {
		t.Errorf("Greedy failed. Expected 1 (logit 5.0), got %d", val)
	}
}

func TestSampler_TopK(t *testing.T) {
	// K=1 should be identical to Greedy
	s := NewSampler(SamplerConfig{Temperature: 1.0, TopK: 1})

	logits := []float32{2.0, 10.0, 5.0, 1.0}

	// Even with temp 1.0, TopK=1 forces selection of the max
	val := s.Sample(logits, nil)
	if val != 1 {
		t.Errorf("TopK=1 failed. Expected 1, got %d", val)
	}
}

func TestSampler_TopK_Filtering(t *testing.T) {
	// K=2. Top 2 are ID 1 (10.0) and ID 2 (5.0).
	// ID 0 (2.0) and ID 3 (1.0) should be impossible.
	s := NewSampler(SamplerConfig{Temperature: 1.0, TopK: 2})

	logits := []float32{2.0, 10.0, 5.0, 1.0}

	// Run many times to ensure 0 and 3 never appear
	for i := 0; i < 100; i++ {
		val := s.Sample(logits, nil)
		if val == 0 || val == 3 {
			t.Errorf("TopK=2 failed. Got excluded token %d", val)
		}
	}
}

func TestSampler_TopP(t *testing.T) {
	logits := []float32{-0.91, -1.20, -1.61, -2.30}

	s := NewSampler(SamplerConfig{Temperature: 1.0, TopP: 0.5})
	for i := 0; i < 100; i++ {
		val := s.Sample(logits, nil)
		if val == 2 || val == 3 {
			t.Errorf("TopP=0.5 failed. Got excluded token %d", val)
		}
	}
}

func TestSampler_RepetitionPenalty(t *testing.T) {
	s := NewSampler(SamplerConfig{Temperature: 0, RepPenalty: 2.0}) // Greedy + Penalty

	logits := []float32{0.8, 1.0, 0.8}
	history := []int{1}

	val := s.Sample(logits, history)
	if val == 1 {
		t.Errorf("RepPenalty failed. Penalized token 1 was selected over higher prob tokens.")
	}
}

func TestSampler_PresenceAndFrequencyPenalty(t *testing.T) {
	// Token 1 has logit 2.0, Token 0 has logit 1.5.
	// Without penalty, Token 1 wins.
	// With presence penalty 1.0, Token 1 becomes 1.0 < 1.5. Token 0 wins.
	sPresence := NewSampler(SamplerConfig{Temperature: 0, PresencePenalty: 1.0})
	logits := []float32{1.5, 2.0}
	history := []int{1}
	val := sPresence.Sample(logits, history)
	if val != 0 {
		t.Errorf("PresencePenalty failed. Got %d, want 0", val)
	}

	// With frequency penalty 0.5 and count 3, penalty is 1.5. Token 1 becomes 0.5 < 1.5.
	sFreq := NewSampler(SamplerConfig{Temperature: 0, FrequencyPenalty: 0.5})
	logits2 := []float32{1.5, 2.0}
	history2 := []int{1, 1, 1}
	val2 := sFreq.Sample(logits2, history2)
	if val2 != 0 {
		t.Errorf("FrequencyPenalty failed. Got %d, want 0", val2)
	}
}

func TestSampler_EmptyAndInvalidLogitsSafety(t *testing.T) {
	s := NewSampler(SamplerConfig{Temperature: 0.7, TopK: 40})

	// 1. Empty logits should safely return 0 without panic
	if val := s.Sample([]float32{}, nil); val != 0 {
		t.Errorf("Expected 0 for empty logits, got %d", val)
	}
	if val := s.SampleAdvanced([]float32{}, nil, true); val != 0 {
		t.Errorf("Expected 0 for empty logits in advanced mode, got %d", val)
	}

	// 2. All NaNs should not panic
	nanLogits := []float32{float32(math.NaN()), float32(math.NaN())}
	val := s.Sample(nanLogits, nil)
	if val < 0 || val > 1 {
		t.Errorf("Unexpected token for NaN logits: %d", val)
	}

	// 3. Extreme logits with clamping
	extremeLogits := []float32{10000.0, -10000.0, 0.0}
	valExt := s.Sample(extremeLogits, nil)
	if valExt != 0 {
		t.Errorf("Expected extreme positive logit to win: got %d", valExt)
	}
}

func TestSampler_MinP(t *testing.T) {
	// Top token has high logit, token 2 has low logit
	s := NewSampler(SamplerConfig{Temperature: 1.0, MinP: 0.2})
	logits := []float32{10.0, 9.8, 1.0}
	for i := 0; i < 50; i++ {
		val := s.Sample(logits, nil)
		if val == 2 {
			t.Errorf("MinP failed: excluded low prob token 2 was chosen")
		}
	}
}

