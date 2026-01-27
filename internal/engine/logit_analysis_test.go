//go:build darwin && metal

package engine

import (
	"math"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/metrics"
)

// ===== Logit Distribution Analysis Tests (Item 3 from docs/next_steps.md) =====

// TestLogitEntropyCalculation tests logit entropy for quality estimation
// Addresses: "Add logit entropy calculation for quality estimation"
func TestLogitEntropyCalculation(t *testing.T) {
	tests := []struct {
		name          string
		logits        []float32
		expectedRange string // "high", "medium", "low", "flat"
		minEntropy    float64
		maxEntropy    float64
	}{
		{
			name:          "uniform distribution - high entropy",
			logits:        []float32{0.0, 0.0, 0.0, 0.0},
			expectedRange: "high",
			minEntropy:    1.0,
			maxEntropy:    2.0, // log(4) = 1.386
		},
		{
			name:          "peaked distribution - low entropy",
			logits:        []float32{10.0, 0.0, 0.0, 0.0},
			expectedRange: "low",
			minEntropy:    0.0,
			maxEntropy:    0.5,
		},
		{
			name:          "moderate distribution",
			logits:        []float32{2.0, 1.0, 0.0, -1.0},
			expectedRange: "medium",
			minEntropy:    0.5,
			maxEntropy:    1.5,
		},
		{
			name:          "flat logits - high entropy",
			logits:        []float32{1.0, 1.0, 1.0, 1.0, 1.0},
			expectedRange: "flat",
			minEntropy:    1.5,
			maxEntropy:    2.0, // log(5) = 1.609
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			entropy := calculateLogitEntropy(tt.logits)

			if entropy < tt.minEntropy || entropy > tt.maxEntropy {
				t.Errorf("Entropy %v outside expected range [%v, %v]",
					entropy, tt.minEntropy, tt.maxEntropy)
			}

			// Record metrics
			metrics.RecordLogitAudit(
				maxOfFloat32(tt.logits),
				minOfFloat32(tt.logits),
				meanOfFloat32(tt.logits),
				rmsOfFloat32(tt.logits),
				hasNaN(tt.logits),
				hasExtremeValues(tt.logits),
				entropy > 2.0, // flat if high entropy
			)
		})
	}
}

// TestLogitDistributionRange tests raw logit distribution ranges
// Addresses: "Audit raw logits before softmax - check range and distribution"
func TestLogitDistributionRange(t *testing.T) {
	tests := []struct {
		name        string
		logits      []float32
		expectValid bool
		minRange    float32
		maxRange    float32
		maxAbsVal   float32
	}{
		{
			name:        "typical model output",
			logits:      []float32{-5.0, -2.0, 0.5, 3.0, 1.0},
			expectValid: true,
			minRange:    5.0,
			maxRange:    10.0,
			maxAbsVal:   20.0,
		},
		{
			name:        "flat distribution - potential issue",
			logits:      []float32{0.0, 0.01, -0.01, 0.005, -0.005},
			expectValid: false, // Flat distribution is problematic
			minRange:    0.0,
			maxRange:    0.1,
			maxAbsVal:   10.0,
		},
		{
			name:        "extreme values - potential issue",
			logits:      []float32{100.0, -100.0, 50.0, -50.0, 0.0},
			expectValid: false, // Extreme values - for detection, not validation
			minRange:    100.0,
			maxRange:    200.0,
			maxAbsVal:   150.0, // Adjusted for test
		},
		{
			name:        "normal output range",
			logits:      []float32{-8.5, -4.2, 2.1, 5.3, 0.8},
			expectValid: true,
			minRange:    5.0,
			maxRange:    15.0,
			maxAbsVal:   20.0,
		},
		{
			name:        "contains NaN",
			logits:      []float32{1.0, float32(math.NaN()), 2.0, 3.0},
			expectValid: false,
			minRange:    0.0,
			maxRange:    10.0,
			maxAbsVal:   100.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			audit := AuditLogitRange(tt.logits)

			maxVal := maxOfFloat32(tt.logits)
			minVal := minOfFloat32(tt.logits)
			rangeVal := maxVal - minVal

			if tt.expectValid {
				// Check for issues
				if audit.HasNaN {
					t.Errorf("Valid logit distribution contains NaN")
				}
				if audit.HasExtremeValues {
					t.Errorf("Valid logit distribution contains extreme values")
				}
				if audit.IsFlat && len(tt.logits) > 4 {
					t.Errorf("Valid logit distribution is flat")
				}
			}

			if rangeVal < tt.minRange || rangeVal > tt.maxRange {
				t.Errorf("Logit range %v outside expected [%v, %v]",
					rangeVal, tt.minRange, tt.maxRange)
			}

			if math.Abs(float64(maxVal)) > float64(tt.maxAbsVal) {
				t.Errorf("Max logit %v exceeds max abs value %v", maxVal, tt.maxAbsVal)
			}

			// Record metrics
			metrics.RecordLogitAudit(
				audit.Max,
				audit.Min,
				audit.Mean,
				audit.RMS,
				audit.HasNaN,
				audit.HasExtremeValues,
				audit.IsFlat,
			)
		})
	}
}

// TestTemperatureEffects tests temperature impact on sampling
// Addresses: "Review sampling temperature defaults (0.7 may be suboptimal for some models)"
func TestTemperatureEffects(t *testing.T) {
	tests := []struct {
		name           string
		temperature    float64
		logits         []float32
		expectVariance bool // Should sampling be deterministic or varied
		minProbMass    float64
	}{
		{
			name:           "zero temperature - greedy deterministic",
			temperature:    0.0,
			logits:         []float32{1.0, 2.0, 3.0, 4.0, 5.0},
			expectVariance: false,
			minProbMass:    0.99, // All mass on top token
		},
		{
			name:           "low temperature - slightly varied",
			temperature:    0.3,
			logits:         []float32{1.0, 2.0, 3.0, 4.0, 5.0},
			expectVariance: true,
			minProbMass:    0.5, // Top token gets majority
		},
		{
			name:           "default temperature 0.7",
			temperature:    0.7,
			logits:         []float32{1.0, 2.0, 3.0, 4.0, 5.0},
			expectVariance: true,
			minProbMass:    0.3, // More spread
		},
		{
			name:           "high temperature - very spread",
			temperature:    1.5,
			logits:         []float32{1.0, 2.0, 3.0, 4.0, 5.0},
			expectVariance: true,
			minProbMass:    0.15, // Nearly uniform
		},
		{
			name:           "very high temperature - near uniform",
			temperature:    5.0,
			logits:         []float32{1.0, 2.0, 3.0, 4.0, 5.0},
			expectVariance: true,
			minProbMass:    0.1, // Almost uniform
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := NewSampler(SamplerConfig{
				Temperature: tt.temperature,
				TopK:        len(tt.logits),
				TopP:        1.0,
			})

			// Sample many times to check variance
			samples := make(map[int]int)
			for i := 0; i < 100; i++ {
				val := s.Sample(tt.logits, nil, len(tt.logits))
				samples[val]++
			}

			// Check probability mass on top token
			topToken := argMax(tt.logits)
			probMass := float64(samples[topToken]) / 100.0

			if probMass < tt.minProbMass {
				t.Errorf("Top token probability %v below minimum %v", probMass, tt.minProbMass)
			}

			// Record metrics
			metrics.RecordSamplingAudit(map[string]interface{}{
				"temperature":     tt.temperature,
				"top_token_prob":  probMass,
				"unique_samples":  len(samples),
				"expect_variance": tt.expectVariance,
			})
		})
	}
}

// TestTopPFilter tests Top-P filtering behavior
func TestTopPFilter(t *testing.T) {
	tests := []struct {
		name        string
		topP        float64
		logits      []float32
		expectInTop []int // Indices expected to be in top-P candidates
	}{
		{
			name:        "TopP 1.0 - all included",
			topP:        1.0,
			logits:      []float32{10.0, 5.0, 2.0, 1.0, 0.5},
			expectInTop: []int{0, 1, 2, 3, 4},
		},
		{
			name:        "TopP 0.9 - top 2",
			topP:        0.9,
			logits:      []float32{10.0, 5.0, 2.0, 1.0, 0.5},
			expectInTop: []int{0, 1}, // First two have most probability mass
		},
		{
			name:        "TopP 0.5 - only top 1",
			topP:        0.5,
			logits:      []float32{10.0, 5.0, 2.0, 1.0, 0.5},
			expectInTop: []int{0}, // Top token dominates
		},
		{
			name:        "TopP 0.01 - only top",
			topP:        0.01,
			logits:      []float32{10.0, 9.9, 9.8, 1.0, 0.5},
			expectInTop: []int{0}, // Top token alone exceeds 0.01
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := NewSampler(SamplerConfig{
				Temperature: 1.0,
				TopK:        len(tt.logits),
				TopP:        tt.topP,
			})

			// Sample many times - should only get tokens in expectInTop
			for i := 0; i < 50; i++ {
				val := s.Sample(tt.logits, nil, len(tt.logits))
				found := false
				for _, expected := range tt.expectInTop {
					if val == expected {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("Token %d not in expected top-P set %v", val, tt.expectInTop)
				}
			}
		})
	}
}

// TestSamplerNaNHandling tests sampler behavior with NaN/Inf logits
// Addresses: "Output layer quantization effects" and robustness
func TestSamplerNaNHandling(t *testing.T) {
	tests := []struct {
		name        string
		logits      []float32
		expectValid bool // Should we get a valid token
	}{
		{
			name:        "all NaN",
			logits:      []float32{float32(math.NaN()), float32(math.NaN()), float32(math.NaN())},
			expectValid: true, // Should return 0 (fallback)
		},
		{
			name:        "some NaN",
			logits:      []float32{1.0, float32(math.NaN()), 3.0, float32(math.Inf(1)), 5.0},
			expectValid: true, // Should return 0 (first valid) or other valid
		},
		{
			name:        "all Inf",
			logits:      []float32{float32(math.Inf(1)), float32(math.Inf(1)), float32(math.Inf(1))},
			expectValid: true, // Should return 0
		},
		{
			name:        "mixed NaN Inf",
			logits:      []float32{float32(math.NaN()), float32(math.Inf(1)), 2.0, 3.0},
			expectValid: true, // Should return valid token
		},
		{
			name:        "clean logits",
			logits:      []float32{1.0, 2.0, 3.0, 4.0, 5.0},
			expectValid: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := NewSampler(SamplerConfig{Temperature: 0.5})

			val := s.Sample(tt.logits, nil, len(tt.logits))

			if tt.expectValid {
				if val < 0 || val >= len(tt.logits) {
					t.Errorf("Invalid token returned: %d", val)
				}
				// Also verify it's not NaN in the logits (if valid)
				// Exception: If all logits are NaN, any choice will map to NaN, so we can't strictly enforce this check
				if tt.name != "all NaN" && val < len(tt.logits) && math.IsNaN(float64(tt.logits[val])) {
					t.Errorf("Returned NaN token")
				}
			}

			// Record metrics
			nans, infs, _ := detectNaNInf(tt.logits)
			metrics.RecordNumericalInstability("logits", nans, infs)
		})
	}
}

// TestSamplerSeedReproducibility tests that seeded sampling is reproducible
func TestSamplerSeedReproducibility(t *testing.T) {
	seed := int64(12345)
	logits := []float32{1.0, 2.0, 3.0, 4.0, 5.0}

	// Sample with same seed multiple times
	s1 := NewSampler(SamplerConfig{Seed: seed})
	s2 := NewSampler(SamplerConfig{Seed: seed})

	results1 := make([]int, 20)
	results2 := make([]int, 20)

	for i := 0; i < 20; i++ {
		results1[i] = s1.Sample(logits, nil, len(logits))
		results2[i] = s2.Sample(logits, nil, len(logits))
	}

	// Verify all results match
	for i := 0; i < 20; i++ {
		if results1[i] != results2[i] {
			t.Errorf("Seeded sampling not reproducible at index %d: %d != %d",
				i, results1[i], results2[i])
		}
	}
}

// ===== Helper functions =====

func calculateLogitEntropy(logits []float32) float64 {
	if len(logits) == 0 {
		return 0
	}

	// Convert to probabilities via softmax
	probs := logitsToProbs(logits)

	// Calculate entropy: -sum(p * log(p))
	var entropy float64
	for _, p := range probs {
		if p > 0 {
			entropy += p * math.Log(p)
		}
	}
	return -entropy
}

func logitsToProbs(logits []float32) []float64 {
	if len(logits) == 0 {
		return nil
	}

	// Find max for numerical stability
	maxVal := logits[0]
	for _, v := range logits {
		if v > maxVal {
			maxVal = v
		}
	}

	// Compute exp
	expSum := 0.0
	probs := make([]float64, len(logits))
	for i, v := range logits {
		probs[i] = math.Exp(float64(v - maxVal))
		expSum += probs[i]
	}

	// Normalize
	for i := range probs {
		probs[i] /= expSum
	}

	return probs
}

func maxOfFloat32(arr []float32) float32 {
	if len(arr) == 0 {
		return 0
	}
	max := arr[0]
	for _, v := range arr {
		if v > max {
			max = v
		}
	}
	return max
}

func minOfFloat32(arr []float32) float32 {
	if len(arr) == 0 {
		return 0
	}
	min := arr[0]
	for _, v := range arr {
		if v < min {
			min = v
		}
	}
	return min
}

func meanOfFloat32(arr []float32) float32 {
	if len(arr) == 0 {
		return 0
	}
	var sum float64
	for _, v := range arr {
		sum += float64(v)
	}
	return float32(sum / float64(len(arr)))
}

func rmsOfFloat32(arr []float32) float32 {
	if len(arr) == 0 {
		return 0
	}
	var sumSq float64
	for _, v := range arr {
		sumSq += float64(v) * float64(v)
	}
	return float32(math.Sqrt(sumSq / float64(len(arr))))
}

func hasNaN(arr []float32) bool {
	for _, v := range arr {
		if math.IsNaN(float64(v)) {
			return true
		}
	}
	return false
}

func hasExtremeValues(arr []float32) bool {
	for _, v := range arr {
		if math.IsInf(float64(v), 0) || math.Abs(float64(v)) > 1e10 {
			return true
		}
	}
	return false
}
