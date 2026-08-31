package engine

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/logger"
	"github.com/23skdu/longbow-quarrel/internal/metrics"
)

// SpeculativeManager orchestrates draft-model generation logic.
type SpeculativeManager struct {
	mu           sync.Mutex
	targetEngine Engine
	draftEngine  Engine
	rng          *rand.Rand
}

func NewSpeculativeManager(target, draft Engine) *SpeculativeManager {
	return &SpeculativeManager{
		targetEngine: target,
		draftEngine:  draft,
		rng:          rand.New(rand.NewSource(time.Now().UnixNano())), // #nosec G404 -- math/rand is fine for speculative sampling
	}
}

// GenerateSpeculativeMultiPath performs continuous decoding using multiple parallel draft paths.
func (sm *SpeculativeManager) GenerateSpeculativeMultiPath(ctx context.Context, seq *Sequence) error {
	if sm.draftEngine == nil || sm.targetEngine == nil {
		return fmt.Errorf("engines not fully initialized")
	}

	numPaths := seq.NumPaths
	if numPaths < 1 { numPaths = 1 }
	draftK := seq.DraftK
	if draftK < 1 { draftK = 4 }

	currentPos := seq.Pos

	// 1. Generate N parallel candidate paths using the Draft model
	// We create N sequences in the draft engine, starting from the same prefix
	candidates := make([][]int, numPaths)
	for i := 0; i < numPaths; i++ {
		candidates[i] = make([]int, 0, draftK)
	}

	// For simplicity in this implementation, we run draft sampling in a loop
	// but using the DraftEngine's internal batching if it were exposed.
	// Since the current Engine interface is sequence-oriented for Infer, 
	// we will run N parallel inferences.
	
	var wg sync.WaitGroup
	wg.Add(numPaths)
	for p := 0; p < numPaths; p++ {
		go func(pathIdx int) {
			defer wg.Done()
			// Each path gets its own stochastic sample
			pathTokens := append([]int{}, seq.Tokens...)
			for k := 0; k < draftK; k++ {
				token, _, err := sm.draftEngine.InferWithLogits(pathTokens, 1, seq.Config)
				if err != nil {
					logger.Log.Error("Draft path inference failed", "path", pathIdx, "error", err)
					return
				}
				candidates[pathIdx] = append(candidates[pathIdx], token[0])
				pathTokens = append(pathTokens, token[0])
			}
		}(p)
	}
	wg.Wait()

	// 2. Target model evaluates all draft paths
	// In a real implementation, we would pack all candidates into a single ForwardBatch call.
	// For now, we'll implement the "Best Path" selection.
	
	bestAcceptedCount := 0
	bestPathIdx := -1

	for p := 0; p < numPaths; p++ {
		if len(candidates[p]) == 0 { continue }
		
		// Evaluate this candidate path against the target model
		candidateTokens := append([]int{}, seq.Tokens...)
		candidateTokens = append(candidateTokens, candidates[p]...)
		
		targetLogits, err := sm.targetEngine.ForwardDraft(candidateTokens)
		if err != nil {
			continue
		}

		accepted := 0
		corrected := make([]int, 0)
		for i := 0; i < len(candidates[p]); i++ {
			if i >= len(targetLogits) {
				break
			}
			draftToken := candidates[p][i]
			isAccepted, correctedToken := rejectSample(targetLogits[i], draftToken, seq.Config, sm.rng)
			if isAccepted {
				accepted++
			} else {
				corrected = append(corrected, correctedToken)
				break
			}
		}

		if len(corrected) > 0 {
			candidates[p] = append(candidates[p][:accepted], corrected[0])
		}
		
		if accepted > bestAcceptedCount {
			bestAcceptedCount = accepted
			bestPathIdx = p
		}
	}

	// 3. Finalize best path
	if bestPathIdx != -1 && bestAcceptedCount > 0 {
		seq.Tokens = append(seq.Tokens, candidates[bestPathIdx][:bestAcceptedCount]...)
		seq.Pos = currentPos + bestAcceptedCount
		
		// Re-sync KV caches if needed (Handled by targetEngine.ForwardBatch in real loop)
	} else {
		// All paths rejected, sample 1 token from target normally
		// (This logic will be moved into the runBatchLoop orchestrator)
	}

	return nil
}

// GenerateSpeculative is kept for backward compatibility with single-path logic
func (sm *SpeculativeManager) GenerateSpeculative(ctx context.Context, seq *Sequence) error {
	seq.NumPaths = 1
	seq.DraftK = 4 // Default
	return sm.GenerateSpeculativeMultiPath(ctx, seq)
}

func rejectSample(targetLogits []float32, draftToken int, cfg SamplerConfig, rng *rand.Rand) (accepted bool, correctedToken int) {
	if draftToken >= len(targetLogits) {
		return false, 0
	}

	draftProb := math.Exp(float64(targetLogits[draftToken]))
	maxLogit := targetLogits[0]
	for _, l := range targetLogits {
		if l > maxLogit {
			maxLogit = l
		}
	}

	sum := 0.0
	probs := make([]float64, len(targetLogits))
	for i, l := range targetLogits {
		probs[i] = math.Exp(float64(l - maxLogit))
		sum += probs[i]
	}
	for i := range probs {
		probs[i] /= sum
	}

	targetProb := probs[draftToken]
	if targetProb <= 0 {
		return false, 0
	}

	acceptanceRatio := targetProb / draftProb
	if acceptanceRatio > 1 {
		acceptanceRatio = 1
	}

	if rng.Float64() < acceptanceRatio {
		metrics.SpeculativeTokensAccepted.Add(1)
		return true, draftToken
	}

	metrics.SpeculativeTokensRejected.Add(1)

	diff := make([]float64, len(probs))
	for i := range diff {
		diff[i] = probs[i] - draftProb
		if diff[i] < 0 {
			diff[i] = 0
		}
	}

	residualSum := 0.0
	for _, v := range diff {
		residualSum += v
	}

	if residualSum <= 0 {
		return false, 0
	}

	r := rng.Float64() * residualSum
	acc := 0.0
	for i, v := range diff {
		acc += v
		if r < acc {
			return false, i
		}
	}

	return false, 0
}
