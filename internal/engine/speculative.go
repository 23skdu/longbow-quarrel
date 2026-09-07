package engine

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/logger"
	"github.com/23skdu/longbow-quarrel/internal/metrics"
)

// SpeculativeManager orchestrates draft-model generation logic.
type SpeculativeManager struct {
	mu            sync.Mutex
	targetEngine  Engine
	draftEngine   Engine
	rng           *rand.Rand
	currentDraftK int
	acceptanceEMA float64
}

func NewSpeculativeManager(target, draft Engine) *SpeculativeManager {
	return &SpeculativeManager{
		targetEngine:  target,
		draftEngine:   draft,
		rng:           rand.New(rand.NewSource(time.Now().UnixNano())), // #nosec G404 -- math/rand is fine for speculative sampling
		currentDraftK: 4,
		acceptanceEMA: 0.65,
	}
}

// GetDynamicDraftLength returns the current adaptive draft length.
func (sm *SpeculativeManager) GetDynamicDraftLength() int {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	if sm.currentDraftK < 1 {
		sm.currentDraftK = 4
	}
	return sm.currentDraftK
}

// GenerateSpeculativeMultiPath performs continuous decoding using multiple parallel draft paths.
func (sm *SpeculativeManager) GenerateSpeculativeMultiPath(ctx context.Context, seq *Sequence) error {
	if sm.draftEngine == nil || sm.targetEngine == nil {
		return fmt.Errorf("engines not fully initialized")
	}

	numPaths := seq.NumPaths
	if numPaths < 1 {
		numPaths = 1
	}
	sm.mu.Lock()
	draftK := seq.DraftK
	if draftK < 1 {
		draftK = sm.currentDraftK
	}
	if draftK < 1 {
		draftK = 4
	}
	sm.mu.Unlock()

	currentPos := seq.Pos

	// 1. Generate N parallel candidate paths using the Draft model
	// We create N sequences in the draft engine, starting from the same prefix
	candidates := make([][]int, numPaths)
	for i := 0; i < numPaths; i++ {
		candidates[i] = make([]int, 0, draftK)
	}

	var wg sync.WaitGroup
	wg.Add(numPaths)
	for p := 0; p < numPaths; p++ {
		go func(pathIdx int) {
			defer wg.Done()
			pathTokens := append([]int{}, seq.Tokens...)
			for k := 0; k < draftK; k++ {
				token, _, err := sm.draftEngine.InferWithLogits(pathTokens, 1, seq.Config)
				if err != nil {
					logger.Log.Error("Draft path inference failed", "path", pathIdx, "error", err)
					return
				}
				if len(token) > 0 {
					candidates[pathIdx] = append(candidates[pathIdx], token[0])
					pathTokens = append(pathTokens, token[0])
				}
			}
		}(p)
	}
	wg.Wait()

	// 2. Target model evaluates all draft paths
	bestAcceptedCount := 0
	bestPathIdx := -1

	for p := 0; p < numPaths; p++ {
		if len(candidates[p]) == 0 {
			continue
		}

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

	// Dynamic draft length adaptation
	sm.mu.Lock()
	accRate := float64(bestAcceptedCount) / float64(draftK)
	sm.acceptanceEMA = 0.8*sm.acceptanceEMA + 0.2*accRate
	if sm.acceptanceEMA > 0.8 && sm.currentDraftK < 8 {
		sm.currentDraftK++
	} else if sm.acceptanceEMA < 0.5 && sm.currentDraftK > 1 {
		sm.currentDraftK--
	}
	dynamicLen := sm.currentDraftK
	sm.mu.Unlock()

	metrics.RecordSpeculativeStep(draftK, bestAcceptedCount, dynamicLen)

	// 3. Finalize best path
	if bestPathIdx != -1 && bestAcceptedCount > 0 {
		seq.Tokens = append(seq.Tokens, candidates[bestPathIdx][:bestAcceptedCount]...)
		seq.Pos = currentPos + bestAcceptedCount
	}

	return nil
}

// GenerateSpeculative is kept for backward compatibility with single-path logic
func (sm *SpeculativeManager) GenerateSpeculative(ctx context.Context, seq *Sequence) error {
	seq.NumPaths = 1
	seq.DraftK = sm.GetDynamicDraftLength()
	return sm.GenerateSpeculativeMultiPath(ctx, seq)
}

// AsymmetricSpeculativeEngine combines an arbitrary draft engine with a primary target engine.
type AsymmetricSpeculativeEngine struct {
	targetEngine Engine
	draftEngine  Engine
	manager      *SpeculativeManager
}

func NewAsymmetricSpeculativeEngine(target, draft Engine) *AsymmetricSpeculativeEngine {
	return &AsymmetricSpeculativeEngine{
		targetEngine: target,
		draftEngine:  draft,
		manager:      NewSpeculativeManager(target, draft),
	}
}

func (e *AsymmetricSpeculativeEngine) Infer(tokens []int, count int, cfg SamplerConfig) ([]int, error) {
	return e.InferWithCallback(tokens, count, cfg, nil)
}

func (e *AsymmetricSpeculativeEngine) InferWithLogits(tokens []int, count int, cfg SamplerConfig) ([]int, []float32, error) {
	out, err := e.Infer(tokens, count, cfg)
	return out, nil, err
}

func (e *AsymmetricSpeculativeEngine) InferWithCallback(tokens []int, count int, cfg SamplerConfig, callback func(int)) ([]int, error) {
	currentTokens := append([]int{}, tokens...)
	seq := &Sequence{
		Tokens:   currentTokens,
		Pos:      len(tokens),
		DraftK:   e.manager.GetDynamicDraftLength(),
		NumPaths: 2,
		Config:   cfg,
	}

	generated := 0
	ctx := context.Background()

	for generated < count {
		seq.DraftK = e.manager.GetDynamicDraftLength()
		startLen := len(seq.Tokens)
		if err := e.manager.GenerateSpeculativeMultiPath(ctx, seq); err != nil {
			single, err2 := e.targetEngine.Infer(seq.Tokens, 1, cfg)
			if err2 != nil {
				return seq.Tokens[len(tokens):], err
			}
			if len(single) > 0 {
				seq.Tokens = append(seq.Tokens, single[0])
				if callback != nil {
					callback(single[0])
				}
				generated++
			}
			continue
		}

		newTokens := len(seq.Tokens) - startLen
		if newTokens > 0 {
			for i := startLen; i < len(seq.Tokens); i++ {
				if callback != nil {
					callback(seq.Tokens[i])
				}
				generated++
				if generated >= count {
					break
				}
			}
		} else {
			single, err := e.targetEngine.Infer(seq.Tokens, 1, cfg)
			if err != nil {
				return seq.Tokens[len(tokens):], err
			}
			if len(single) > 0 {
				seq.Tokens = append(seq.Tokens, single[0])
				if callback != nil {
					callback(single[0])
				}
				generated++
			}
		}
	}

	return seq.Tokens[len(tokens):], nil
}

func (e *AsymmetricSpeculativeEngine) InferWithCallbackLogits(tokens []int, count int, cfg SamplerConfig, tokenCallback func(int), logitsCallback func([]float32)) ([]int, error) {
	return e.InferWithCallback(tokens, count, cfg, tokenCallback)
}

func (e *AsymmetricSpeculativeEngine) Config() config.Config {
	return e.targetEngine.Config()
}

func (e *AsymmetricSpeculativeEngine) Close() {
	if e.targetEngine != nil {
		e.targetEngine.Close()
	}
	if e.draftEngine != nil {
		e.draftEngine.Close()
	}
}

func (e *AsymmetricSpeculativeEngine) SwapModel(modelPath string, cfg config.Config) error {
	return e.targetEngine.SwapModel(modelPath, cfg)
}

func (e *AsymmetricSpeculativeEngine) LoadAdapter(path, id string) error {
	return e.targetEngine.LoadAdapter(path, id)
}

func (e *AsymmetricSpeculativeEngine) GetSeqCachePos(seqID string) int {
	return e.targetEngine.GetSeqCachePos(seqID)
}

func (e *AsymmetricSpeculativeEngine) ForwardDraft(tokens []int) ([][]float32, error) {
	return e.targetEngine.ForwardDraft(tokens)
}

func (e *AsymmetricSpeculativeEngine) RollbackKV(seqID string, newPos int) error {
	_ = e.targetEngine.RollbackKV(seqID, newPos)
	if e.draftEngine != nil {
		_ = e.draftEngine.RollbackKV(seqID, newPos)
	}
	return nil
}

func (e *AsymmetricSpeculativeEngine) ForwardBatch(desc *BatchDescriptor) ([]*device.Tensor, error) {
	return e.targetEngine.ForwardBatch(desc)
}

func rejectSample(targetLogits []float32, draftToken int, _ SamplerConfig, rng *rand.Rand) (accepted bool, correctedToken int) {
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
