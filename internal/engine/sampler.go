package engine

import (
	"log"
	"math"
	"math/rand"
	"sort"
	"time"
)

type Sampler struct {
	Config     SamplerConfig
	rng        *rand.Rand
	mirostatMu float64
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

func (s *Sampler) SampleAdvanced(logits []float32, history []int, qualityMode bool) int {
	if qualityMode {
		return s.sampleWithQualityControl(logits, history)
	}
	if s.Config.Grammar != nil {
		s.Config.Grammar.Apply(logits)
	}
	return s.Sample(logits, history)
}

func (s *Sampler) sampleWithQualityControl(logits []float32, history []int) int {
	if !s.validateLogits(logits) {
		return s.findFirstValidToken(logits)
	}

	if s.Config.RepPenalty > 1.0 && len(history) > 0 {
		s.applySmartRepetitionPenalty(logits, history)
	}

	temp := s.calculateAdaptiveTemperature(logits)
	if temp == 0 {
		return argMax(logits)
	}

	probs := s.applyTemperatureAndSoftmax(logits, temp)

	candidates := s.filterValidCandidates(probs)
	if len(candidates) == 0 {
		return argMax(logits)
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].prob > candidates[j].prob
	})

	candidates = applyTopP(candidates, s.Config.TopP)
	if s.Config.TopK > 0 {
		candidates = applyTopK(candidates, s.Config.TopK)
	}

	if len(candidates) == 0 {
		return argMax(logits)
	}

	return s.sampleFromCandidates(candidates)
}

func (s *Sampler) Sample(logits []float32, history []int) int {
	if s.Config.Grammar != nil {
		s.Config.Grammar.Apply(logits)
	}
	if !s.validateLogits(logits) {
		return s.findFirstValidToken(logits)
	}

	if s.Config.RepPenalty > 1.0 && len(history) > 0 {
		s.applyRepetitionPenalty(logits, history)
	}

	temp := s.Config.Temperature
	if temp == 0 {
		return argMax(logits)
	}

	probs := s.applyTemperatureAndSoftmax(logits, temp)

	candidates := s.filterValidCandidates(probs)
	if len(candidates) == 0 {
		return argMax(logits)
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].prob > candidates[j].prob
	})

	candidates = applyTopK(candidates, s.Config.TopK)
	candidates = applyTopP(candidates, s.Config.TopP)

	if s.Config.MinP > 0 && len(candidates) > 0 {
		maxProb := candidates[0].prob
		filtered := make([]tokenProb, 0, len(candidates))
		for _, c := range candidates {
			if c.prob >= s.Config.MinP*maxProb {
				filtered = append(filtered, c)
			}
		}
		if len(filtered) > 0 {
			candidates = filtered
		}
	}

	if len(candidates) == 0 {
		return argMax(logits)
	}

	if s.Config.MirostatTau > 0 && s.Config.MirostatEta > 0 && s.mirostatMu == 0 {
		s.mirostatMu = 2 * s.Config.MirostatTau
	}

	selected := s.sampleFromCandidates(candidates)

	if s.Config.MirostatTau > 0 && s.Config.MirostatEta > 0 {
		selectedProb := 0.0
		for _, c := range candidates {
			if c.id == selected {
				selectedProb = c.prob
				break
			}
		}
		if selectedProb > 0 {
			surprise := -math.Log2(selectedProb)
			s.mirostatMu -= s.Config.MirostatEta * (surprise - s.Config.MirostatTau)
			if s.mirostatMu < 0 {
				s.mirostatMu = 0
			}
		}
	}

	return selected
}

func (s *Sampler) validateLogits(logits []float32) bool {
	for _, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return false
		}
	}
	return true
}

func (s *Sampler) findFirstValidToken(logits []float32) int {
	for i, v := range logits {
		if !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) {
			return i
		}
	}
	return 0
}

func (s *Sampler) applyTemperatureAndSoftmax(logits []float32, temperature float64) []float64 {
	probs := make([]float64, len(logits))
	maxLogit := float64(logits[0])
	for _, v := range logits {
		if float64(v) > maxLogit {
			maxLogit = float64(v)
		}
	}

	for i, v := range logits {
		probs[i] = float64(v) / temperature
	}

	maxVal := probs[0]
	for _, v := range probs {
		if v > maxVal {
			maxVal = v
		}
	}

	sum := 0.0
	for i := range probs {
		probs[i] = math.Exp(probs[i] - maxVal)
		sum += probs[i]
	}

	for i := range probs {
		probs[i] /= sum
	}

	return probs
}

func (s *Sampler) filterValidCandidates(probs []float64) []tokenProb {
	candidates := make([]tokenProb, 0, len(probs))
	for i, p := range probs {
		if p > 1e-10 && !math.IsNaN(p) && !math.IsInf(p, 0) {
			candidates = append(candidates, tokenProb{id: i, prob: p})
		}
	}
	return candidates
}

func (s *Sampler) sampleFromCandidates(candidates []tokenProb) int {
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

	return candidates[0].id
}

func (s *Sampler) calculateAdaptiveTemperature(logits []float32) float64 {
	if len(logits) == 0 {
		return s.Config.Temperature
	}

	maxLogit := logits[0]
	sum := float64(0)
	for _, v := range logits {
		if v > maxLogit {
			maxLogit = v
		}
		sum += math.Exp(float64(v - maxLogit))
	}

	entropy := float64(0)
	for _, v := range logits {
		prob := math.Exp(float64(v-maxLogit)) / sum
		if prob > 0 {
			entropy -= prob * math.Log(prob)
		}
	}

	baseTemp := s.Config.Temperature
	if entropy > 2.0 {
		return baseTemp * 1.5
	} else if entropy < 0.5 {
		return math.Max(baseTemp*0.5, 0.1)
	}

	return baseTemp
}

func (s *Sampler) applySmartRepetitionPenalty(logits []float32, history []int) {
	if len(history) == 0 {
		return
	}

	freqMap := make(map[int]int)
	windowSize := 32
	start := len(history) - windowSize
	if start < 0 {
		start = 0
	}

	for i := start; i < len(history); i++ {
		if history[i] < len(logits) {
			freqMap[history[i]]++
		}
	}

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

func (s *Sampler) applyRepetitionPenalty(logits []float32, history []int) {
	if len(history) == 0 {
		return
	}

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
			if val > 0 {
				logits[id] /= float32(s.Config.RepPenalty)
			} else {
				logits[id] *= float32(s.Config.RepPenalty)
			}
		}
	}
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
		log.Printf("[WARN] argMax: all logits are NaN, returning index 0")
		return 0
	}

	return maxIdx
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

	sum := 0.0
	for i, c := range candidates {
		sum += c.prob
		if sum >= p {
			selected := candidates[:i+1]

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
