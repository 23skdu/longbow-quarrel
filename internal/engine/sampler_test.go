//go:build darwin && metal

package engine

import (
	"testing"
)

func TestSampler_Greedy(t *testing.T) {
	s := NewSampler(SamplerConfig{Temperature: 0})

	// Tokens: 0, 1, 2, 3
	// Logits: 1.0, 5.0, 2.0, 0.5
	logits := []float32{1.0, 5.0, 2.0, 0.5}

	val := s.Sample(logits, nil, len(logits))
	if val != 1 {
		t.Errorf("Greedy failed. Expected 1 (logit 5.0), got %d", val)
	}
}

func TestSampler_TopK(t *testing.T) {
	// K=1 should be identical to Greedy
	s := NewSampler(SamplerConfig{Temperature: 1.0, TopK: 1})

	logits := []float32{2.0, 10.0, 5.0, 1.0}

	// Even with temp 1.0, TopK=1 forces selection of the max
	val := s.Sample(logits, nil, len(logits))
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
		val := s.Sample(logits, nil, len(logits))
		if val == 0 || val == 3 {
			t.Errorf("TopK=2 failed. Got excluded token %d", val)
		}
	}
}

func TestSampler_TopP(t *testing.T) {
	// P=0.5.
	// Logits -> Probs.
	// temp := 1.0 (Unused var removal)
	logits := []float32{10.0, 5.0, 2.0, 1.0}
	// Exp: 22026, 148, 7.3, 2.7
	// Sum ~22184
	// Probs: ~0.99, ~0.006, ...
	// Here TopP 0.9 would include just index 0.

	// Let's make it closer.
	// 0: 0.4
	// 1: 0.3
	// 2: 0.2
	// 3: 0.1
	// Logits need to produce these probs. ln(0.4) ~= -0.91
	logits = []float32{-0.91, -1.20, -1.61, -2.30}

	// TopP 0.5 should include 0 (0.4) and 1 (0.3) because 0.4 < 0.5, so we add next?
	// Standard Logic: Include smallest set summing to >= P.
	// So 0 (0.4) -> sum 0.4 < 0.5. Include 1 (0.3) -> sum 0.7 >= 0.5. Stop.
	// Candidates: 0, 1.
	// 2 and 3 should be excluded.

	s := NewSampler(SamplerConfig{Temperature: 1.0, TopP: 0.5})
	for i := 0; i < 100; i++ {
		val := s.Sample(logits, nil, len(logits))
		if val == 2 || val == 3 {
			t.Errorf("TopP=0.5 failed. Got excluded token %d", val)
		}
	}
}

func TestSampler_RepetitionPenalty(t *testing.T) {
	s := NewSampler(SamplerConfig{Temperature: 0, RepPenalty: 2.0}) // Greedy + Penalty

	logits := []float32{1.0, 1.0, 1.0} // IDs 0, 1, 2 equal
	history := []int{1}                // 1 was seen

	// With penalty 1.2 on ID 1:
	// If logit > 0: logit /= 2.0 -> 0.5
	// ID 0: 1.0
	// ID 1: 0.5
	// ID 2: 1.0
	// Greedy should pick 0 or 2. 1 is penalized.

	// But we need to make 1 the default winner first.
	logits = []float32{0.8, 1.0, 0.8}
	// Normal: 1 wins.
	// Penalized (1.0/2.0 = 0.5): 1 becomes 0.5. 0 and 2 (0.8) win.

	val := s.Sample(logits, history, len(logits))
	if val == 1 {
		t.Errorf("RepPenalty failed. Penalized token 1 was selected over higher prob tokens.")
	}
}
