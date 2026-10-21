package engine

import (
	"math"
	"testing"
)

func TestSampler_Logic_Full(t *testing.T) {
	cfg := SamplerConfig{
		Temperature: 0.8,
		TopK:        2,
		TopP:        0.9,
		RepPenalty:  1.1,
	}
	sampler := NewSampler(cfg)

	logits := []float32{1.0, 2.0, 5.0, 2.0, 1.0}
	history := []int{2, 2, 1}

	// 1. Test standard Sample
	token := sampler.Sample(logits, history)
	if token < 0 || token >= 5 {
		t.Errorf("Invalid token from Sample: %d", token)
	}
}

func TestSampler_EdgeCases(t *testing.T) {
	cfg := SamplerConfig{Temperature: 0.0}
	sampler := NewSampler(cfg)

	logits := []float32{1.0, 10.0, 5.0}
	// Case temp=0 -> argMax
	token := sampler.Sample(logits, nil)
	if token != 1 {
		t.Errorf("Expected token 1 for temp=0, got %d", token)
	}

	// Case NaN logits
	l_nan := []float32{1.0, float32(math.NaN()), 3.0}
	_ = sampler.Sample(l_nan, nil) // Should handle via validateLogits
}
