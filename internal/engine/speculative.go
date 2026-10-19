package engine

import (
	"context"
	"fmt"
	"math/rand"
	"sync"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/mathx"

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
	numPaths      int
	acceptanceEMA float64
}

func NewSpeculativeManager(target, draft Engine) *SpeculativeManager {
	return &SpeculativeManager{
		targetEngine:  target,
		draftEngine:   draft,
		rng:           rand.New(rand.NewSource(time.Now().UnixNano())), // #nosec G404 -- math/rand is fine for speculative sampling
		currentDraftK: 4,
		numPaths:      2,
		acceptanceEMA: 0.65,
	}
}

// NewSpeculativeManagerWithOptions builds a manager with an explicit draft
// length and number of parallel draft paths.
//
// A draft length of zero or less keeps the default, and so does a path
// count of zero or less.
func NewSpeculativeManagerWithOptions(target, draft Engine, draftLength, numPaths int) *SpeculativeManager {
	sm := NewSpeculativeManager(target, draft)
	if draftLength > 0 {
		sm.currentDraftK = draftLength
	}
	if numPaths > 0 {
		sm.numPaths = numPaths
	}
	return sm
}

// SetDraftLength sets how many tokens the draft model proposes per step.
//
// The length adapts on its own from the measured acceptance rate, so this
// is a starting point rather than a fixed value.
func (e *AsymmetricSpeculativeEngine) SetDraftLength(k int) {
	if k < 1 {
		return
	}
	e.manager.mu.Lock()
	defer e.manager.mu.Unlock()
	e.manager.currentDraftK = k
}

// GetNumPaths returns how many parallel draft paths are generated per step.
func (sm *SpeculativeManager) GetNumPaths() int {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	if sm.numPaths < 1 {
		sm.numPaths = 2
	}
	return sm.numPaths
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
		numPaths = sm.GetNumPaths()
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

	// The draft model's distribution over the vocabulary is needed to
	// verify its proposals: the acceptance test compares the target's
	// probability for a token with the draft's probability for the same
	// token, and only the draft engine knows the latter.
	draftLogits := make([][][]float32, numPaths)

	var wg sync.WaitGroup
	wg.Add(numPaths)
	for p := 0; p < numPaths; p++ {
		go func(pathIdx int) {
			defer wg.Done()
			pathTokens := append([]int{}, seq.Tokens...)
			for k := 0; k < draftK; k++ {
				token, logits, err := sm.draftEngine.InferWithLogits(pathTokens, 1, seq.Config)
				if err != nil {
					logger.Log.Error("Draft path inference failed", "path", pathIdx, "error", err)
					return
				}
				if len(token) > 0 {
					candidates[pathIdx] = append(candidates[pathIdx], token[0])
					draftLogits[pathIdx] = append(draftLogits[pathIdx], logits)
					pathTokens = append(pathTokens, token[0])
				}
			}
		}(p)
	}
	wg.Wait()

	// 2. Target model evaluates all draft paths
	bestAcceptedCount := 0
	bestPathIdx := -1
	bestCorrected := 0
	bestHasCorrection := false

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
		corrected := 0
		hasCorrection := false
		for i := 0; i < len(candidates[p]); i++ {
			if i >= len(targetLogits) {
				break
			}
			draftToken := candidates[p][i]
			var draftVec []float32
			if i < len(draftLogits[p]) {
				draftVec = draftLogits[p][i]
			}
			isAccepted, correctedToken := rejectSample(targetLogits[i], draftVec, draftToken, sm.rng)
			if isAccepted {
				accepted++
				continue
			}
			// The draft token was rejected. The correction is the token
			// that replaces it, and it has to be emitted: it is the
			// target's own sample for this position.
			corrected = correctedToken
			hasCorrection = true
			break
		}

		// A path that produced a correction is worth more than one that
		// merely accepted the same number of tokens, because it also
		// advances the sequence by the corrected token.
		if accepted > bestAcceptedCount ||
			(accepted == bestAcceptedCount && hasCorrection && !bestHasCorrection) {
			bestAcceptedCount = accepted
			bestPathIdx = p
			bestCorrected = corrected
			bestHasCorrection = hasCorrection
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
	if bestPathIdx != -1 && (bestAcceptedCount > 0 || bestHasCorrection) {
		seq.Tokens = append(seq.Tokens, candidates[bestPathIdx][:bestAcceptedCount]...)
		if bestHasCorrection {
			// The token that replaces the rejected draft token.
			seq.Tokens = append(seq.Tokens, bestCorrected)
		}
		seq.Pos = currentPos + bestAcceptedCount
		if bestHasCorrection {
			seq.Pos++
		}
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

// NewAsymmetricSpeculativeEngineWithOptions builds the engine with an
// explicit draft length and number of parallel draft paths.
func NewAsymmetricSpeculativeEngineWithOptions(target, draft Engine, draftLength, numPaths int) *AsymmetricSpeculativeEngine {
	return &AsymmetricSpeculativeEngine{
		targetEngine: target,
		draftEngine:  draft,
		manager:      NewSpeculativeManagerWithOptions(target, draft, draftLength, numPaths),
	}
}

// NewSpeculativeEngine loads the draft model and wraps the target engine so
// that the draft model's proposals are verified by the target model.
//
// The draft model is loaded with the same configuration as the target, minus
// the GPU layer offload: a draft model is small by definition, and putting
// it on the GPU would take the memory the target model needs. A caller that
// wants the draft model on the GPU can build the draft engine itself and
// use NewAsymmetricSpeculativeEngine directly.
func NewSpeculativeEngine(target Engine, draftModelPath string, cfg config.Config) (Engine, error) {
	if target == nil {
		return nil, fmt.Errorf("no target engine to wrap")
	}
	if draftModelPath == "" {
		return nil, fmt.Errorf("no draft model given")
	}

	draftCfg := cfg
	draftCfg.NumGPULayers = 0
	draftCfg.SpeculativeDraftModel = ""

	draft, err := NewEngine(draftModelPath, draftCfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create the draft engine: %w", err)
	}

	logger.Log.Info("Speculative decoding enabled",
		"draft_model", draftModelPath,
		"draft_length", cfg.SpeculativeDraftLength,
		"paths", cfg.SpeculativeNumPaths)

	return NewAsymmetricSpeculativeEngineWithOptions(target, draft,
		cfg.SpeculativeDraftLength, cfg.SpeculativeNumPaths), nil
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
		NumPaths: e.manager.GetNumPaths(),
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

	// A speculative step can produce more tokens than were asked for. Only
	// the tokens that were actually emitted count: returning the surplus
	// would hand the caller tokens it never saw, and would leave the
	// sequence ahead of the caller's view of it.
	all := seq.Tokens[len(tokens):]
	if len(all) > generated {
		seq.Tokens = seq.Tokens[:len(tokens)+generated]
		all = all[:generated]
	}
	return all, nil
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

// softmax converts a logit vector into a probability distribution.
//
// The maximum is subtracted first so that the exponentials cannot
// overflow, which matters because model logits routinely reach into the
// tens.
func softmax(logits []float32) []float64 {
	if len(logits) == 0 {
		return nil
	}
	probs := make([]float64, len(logits))
	maxLogit := float64(logits[0])
	for _, l := range logits {
		if float64(l) > maxLogit {
			maxLogit = float64(l)
		}
	}
	sum := 0.0
	for i, l := range logits {
		probs[i] = mathx.Exp(float64(l) - maxLogit)
		sum += probs[i]
	}
	if sum <= 0 {
		return probs
	}
	for i := range probs {
		probs[i] /= sum
	}
	return probs
}

// sampleFrom draws an index from a probability distribution.
func sampleFrom(probs []float64, rng *rand.Rand) int {
	total := 0.0
	for _, p := range probs {
		total += p
	}
	if total <= 0 {
		return 0
	}
	r := rng.Float64() * total
	acc := 0.0
	for i, p := range probs {
		acc += p
		if r < acc {
			return i
		}
	}
	return len(probs) - 1
}

// rejectSample verifies one token proposed by the draft model against the
// target model's distribution.
//
// This is the acceptance step of speculative sampling: the draft token is
// accepted with probability min(1, p_target(x) / p_draft(x)), and a
// rejected token is replaced by a draw from the residual distribution
// norm(max(0, p_target - p_draft)). The result is a sample from the
// target's distribution, which is what makes the whole scheme correct.
//
// The draft's probability for the token is essential, and the previous
// implementation did not have it: it compared the target's *normalized*
// probability for the token with the target's *unnormalized* exp(logit) for
// the same token. With real model logits the unnormalized value dwarfs the
// normalized one, so the acceptance ratio collapsed towards zero and
// virtually every draft token was rejected -- the scheme could not have
// produced a speedup.
//
// When the draft and the target disagree on the vocabulary size the two
// distributions cannot be compared, and the draft token is accepted when
// the target ranks it first and replaced by a draw from the target's own
// distribution otherwise.
func rejectSample(targetLogits []float32, draftLogits []float32, draftToken int, rng *rand.Rand) (accepted bool, correctedToken int) {
	if draftToken < 0 || draftToken >= len(targetLogits) {
		return false, 0
	}

	p := softmax(targetLogits)
	targetProb := p[draftToken]
	if targetProb <= 0 {
		return false, 0
	}

	q := softmax(draftLogits)
	if len(q) != len(p) {
		// The distributions are not comparable. Fall back to accepting
		// the draft token when the target prefers it.
		best := 0
		for i := range p {
			if p[i] > p[best] {
				best = i
			}
		}
		if best == draftToken {
			metrics.SpeculativeTokensAccepted.Add(1)
			return true, draftToken
		}
		metrics.SpeculativeTokensRejected.Add(1)
		return false, sampleFrom(p, rng)
	}

	draftProb := q[draftToken]
	if draftProb <= 0 {
		metrics.SpeculativeTokensRejected.Add(1)
		return false, sampleFrom(p, rng)
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

	// The residual distribution: norm(max(0, p - q)).
	residual := make([]float64, len(p))
	residualSum := 0.0
	for i := range p {
		d := p[i] - q[i]
		if d < 0 {
			d = 0
		}
		residual[i] = d
		residualSum += d
	}

	if residualSum <= 0 {
		// The two distributions are identical, so the rejection was a
		// coin flip. The target's own distribution is the correction.
		return false, sampleFrom(p, rng)
	}

	return false, sampleFrom(residual, rng)
}
