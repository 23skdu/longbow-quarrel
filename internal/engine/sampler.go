package engine

import (
	"log"
	"math"
	"math/rand"
	"sort"
	"time"
)

type SamplerConfig struct {
	Temperature      float64
	TopK             int
	TopP             float64
	RepPenalty       float64 // 1.0 = no penalty, > 1.0 = penalty
	Seed             int64
	DebugActivations bool
	QualityMode      bool // Enable advanced sampling with nucleus sampling and adaptive temperature
}

type Sampler struct {
	Config SamplerConfig
	rng    *rand.Rand
}

func NewSampler(cfg SamplerConfig) *Sampler {
	if cfg.Seed == 0 {
		cfg.Seed = time.Now().UnixNano()
	}
	return &Sampler{
		Config: cfg,
		rng:    rand.New(rand.NewSource(cfg.Seed)),
	}
}

// SampleAdvanced provides enhanced sampling with nucleus sampling and dynamic temperature
func (s *Sampler) SampleAdvanced(logits []float32, history []int, vocabSize int, qualityMode bool) int {
	if qualityMode {
		return s.sampleWithQualityControl(logits, history, vocabSize)
	}
	return s.Sample(logits, history, vocabSize)
}

// sampleWithQualityControl implements advanced sampling with nucleus sampling and quality controls
func (s *Sampler) sampleWithQualityControl(logits []float32, history []int, vocabSize int) int {
	// Check for NaN/Inf in logits and handle gracefully
	hasNaN := false
	hasInf := false
	for _, v := range logits {
		if math.IsNaN(float64(v)) {
			hasNaN = true
			break
		}
		if math.IsInf(float64(v), 0) {
			hasInf = true
			break
		}
	}

	if hasNaN || hasInf {
		// Find the first valid (non-NaN, non-Inf) logit
		for i, v := range logits {
			if !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) {
				return i
			}
		}
		return 0
	}

	// 1. Repetition Penalty (enhanced)
	if s.Config.RepPenalty > 1.0 && len(history) > 0 {
		s.applySmartRepetitionPenalty(logits, history)
	}

	// 2. Dynamic Temperature based on confidence
	temp := s.calculateAdaptiveTemperature(logits)
	if temp == 0 {
		// Greedy sampling for high confidence
		return argMax(logits)
	}

	// 3. Apply temperature
	probs := make([]float64, len(logits))
	maxLogit := float64(logits[0])
	for _, v := range logits {
		if float64(v) > maxLogit {
			maxLogit = float64(v)
		}
	}

	for i, v := range logits {
		probs[i] = float64(v) / temp
	}

	// 4. Softmax
	softmax(probs)

	// 5. Quality-guided filtering: nucleus sampling with top-p
	candidates := make([]tokenProb, 0, len(probs))
	for i, p := range probs {
		if p > 1e-10 && !math.IsNaN(p) && !math.IsInf(p, 0) {
			candidates = append(candidates, tokenProb{id: i, prob: p})
		}
	}

	if len(candidates) == 0 {
		return argMax(logits)
	}

	// Sort by probability (descending)
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].prob > candidates[j].prob
	})

	// Apply nucleus sampling (top-p)
	candidates = applyTopP(candidates, s.Config.TopP)

	// Apply top-k if specified
	if s.Config.TopK > 0 {
		candidates = applyTopK(candidates, s.Config.TopK)
	}

	if len(candidates) == 0 {
		return argMax(logits)
	}

	// Sample from remaining candidates
	r := s.rng.Float64()
	acc := 0.0
	for _, c := range candidates {
		acc += c.prob
		if r < acc {
			return c.id
		}
	}

	// Fallback
	return candidates[0].id
}

// calculateAdaptiveTemperature adjusts temperature based on logit distribution confidence
func (s *Sampler) calculateAdaptiveTemperature(logits []float32) float64 {
	if len(logits) == 0 {
		return s.Config.Temperature
	}

	// Calculate entropy as a measure of confidence
	maxLogit := logits[0]
	sum := float64(0)
	for _, v := range logits {
		if v > maxLogit {
			maxLogit = v
		}
		sum += math.Exp(float64(v - maxLogit))
	}

	// Normalize to get probabilities
	entropy := float64(0)
	for _, v := range logits {
		prob := math.Exp(float64(v-maxLogit)) / sum
		if prob > 0 {
			entropy -= prob * math.Log(prob)
		}
	}

	// High entropy (uncertain) = higher temperature
	// Low entropy (confident) = lower temperature
	baseTemp := s.Config.Temperature
	if entropy > 2.0 {
		// High uncertainty - increase temperature for diversity
		return baseTemp * 1.5
	} else if entropy < 0.5 {
		// High confidence - decrease temperature for consistency
		return math.Max(baseTemp*0.5, 0.1)
	}

	return baseTemp
}

// applySmartRepetitionPenalty applies context-aware repetition penalties
func (s *Sampler) applySmartRepetitionPenalty(logits []float32, history []int) {
	if len(history) == 0 {
		return
	}

	// Create frequency map for recent tokens
	freqMap := make(map[int]int)
	windowSize := 32 // Look at last 32 tokens
	start := len(history) - windowSize
	if start < 0 {
		start = 0
	}

	for i := start; i < len(history); i++ {
		if history[i] < len(logits) {
			freqMap[history[i]]++
		}
	}

	// Apply frequency-based penalty
	for tokenID, freq := range freqMap {
		if tokenID < len(logits) {
			penalty := math.Pow(s.Config.RepPenalty, float64(freq))
			val := logits[tokenID]
			if val > 0 {
				logits[tokenID] /= float32(penalty)
			} else {
				logits[tokenID] *= float32(penalty)
			}
		}
	}
}

func (s *Sampler) Sample(logits []float32, history []int, vocabSize int) int {
	// Check for NaN/Inf in logits and handle gracefully
	hasNaN := false
	hasInf := false
	for _, v := range logits {
		if math.IsNaN(float64(v)) {
			hasNaN = true
			break
		}
		if math.IsInf(float64(v), 0) {
			hasInf = true
			break
		}
	}

	if hasNaN || hasInf {
		// Find the first valid (non-NaN, non-Inf) logit
		for i, v := range logits {
			if !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) {
				return i
			}
		}
		// All logits are NaN/Inf, return 0 as last resort
		return 0
	}

	// 1. Repetition Penalty
	if s.Config.RepPenalty > 1.0 && len(history) > 0 {
		// Apply penalty to tokens in recent history
		// We scan the last N tokens (e.g. 64) or all history
		// Simple approach: Apply to all history for short contexts
		seen := make(map[int]struct{})
		start := 0
		if len(history) > 64 {
			start = len(history) - 64
		}

		for _, id := range history[start:] {
			if _, ok := seen[id]; ok {
				continue
			}
			seen[id] = struct{}{}

			if id < len(logits) {
				val := logits[id]
				// If positive, divide. If negative, multiply.
				// Penalty > 1.0 reduces probability.
				if val > 0 {
					logits[id] /= float32(s.Config.RepPenalty)
				} else {
					logits[id] *= float32(s.Config.RepPenalty)
				}
			}
		}
	}

	// 2. Temperature
	temp := s.Config.Temperature
	if temp == 0 {
		// Greedy
		return argMax(logits)
	}

	// Apply Temp
	// Also convert to float64 for precision
	probs := make([]float64, len(logits))
	for i, v := range logits {
		probs[i] = float64(v) / temp
	}

	// 3. Softmax
	softmax(probs)

	// 4. Top-K / Top-P
	candidates := make([]tokenProb, 0, len(probs))
	for i, p := range probs {
		if p > 1e-10 && !math.IsNaN(p) && !math.IsInf(p, 0) { // Optimization: ignore near-zero and NaN/Inf
			candidates = append(candidates, tokenProb{id: i, prob: p})
		}
	}

	// Handle empty candidates
	if len(candidates) == 0 {
		return argMax(logits)
	}

	// Sort desc
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].prob > candidates[j].prob
	})

	// Apply filters
	candidates = applyTopK(candidates, s.Config.TopK)
	candidates = applyTopP(candidates, s.Config.TopP)

	// Handle empty candidates after filtering
	if len(candidates) == 0 {
		return argMax(logits)
	}

	// 5. Select
	// Re-normalize filtered candidates
	sum := 0.0
	for _, c := range candidates {
		sum += c.prob
	}

	r := s.rng.Float64() * sum
	acc := 0.0
	for _, c := range candidates {
		acc += c.prob
		if r < acc {
			return c.id
		}
	}

	// Fallback: return highest probability candidate
	return candidates[0].id
}

type tokenProb struct {
	id   int
	prob float64
}

func argMax(logits []float32) int {
	if len(logits) == 0 {
		panic("argMax: empty logits slice")
	}

	maxIdx := 0
	maxVal := logits[0]

	// Handle all-NaN case
	allNaN := true
	for i, v := range logits {
		if !math.IsNaN(float64(v)) {
			allNaN = false
			if v > maxVal || math.IsNaN(float64(maxVal)) {
				maxVal = v
				maxIdx = i
			}
		}
	}

	if allNaN {
		// All values are NaN, return first index as fallback
		log.Printf("[WARN] argMax: all logits are NaN, returning index 0")
		return 0
	}

	return maxIdx
}

func softmax(x []float64) {
	max := x[0]
	for _, v := range x {
		if v > max {
			max = v
		}
	}

	sum := 0.0
	for i := range x {
		x[i] = math.Exp(x[i] - max)
		sum += x[i]
	}

	for i := range x {
		x[i] /= sum
	}
}

func applyTopK(candidates []tokenProb, k int) []tokenProb {
	if k <= 0 || k >= len(candidates) {
		return candidates
	}
	return candidates[:k]
}

func applyTopP(candidates []tokenProb, p float64) []tokenProb {
	if p >= 1.0 || p <= 0.0 {
		return candidates
	}

	// Calculate cumulative probabilities to find cutoff point
	sum := 0.0
	for i, c := range candidates {
		sum += c.prob
		if sum >= p {
			// Include tokens up to this point
			selected := candidates[:i+1]

			// Renormalize probabilities to sum to 1.0
			totalProb := 0.0
			for _, c := range selected {
				totalProb += c.prob
			}
			for i := range selected {
				selected[i].prob /= totalProb
			}

			return selected
		}
	}
	return candidates
}
