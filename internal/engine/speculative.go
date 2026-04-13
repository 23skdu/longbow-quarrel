package engine

import (
	"context"
	"fmt"
	"sync"
)

// SpeculativeManager orchestrates draft-model generation logic.
type SpeculativeManager struct {
	mu           sync.Mutex
	targetEngine Engine
	draftEngine  Engine
	draftScale   int // Max number of draft tokens per step
}

func NewSpeculativeManager(target, draft Engine, scale int) *SpeculativeManager {
	return &SpeculativeManager{
		targetEngine: target,
		draftEngine:  draft,
		draftScale:   scale,
	}
}

// GenerateSpeculative performs continuous decoding using speculative evaluation.
func (sm *SpeculativeManager) GenerateSpeculative(ctx context.Context, seq *Sequence) error {
	if sm.draftEngine == nil || sm.targetEngine == nil {
		return fmt.Errorf("engines not fully initialized")
	}

	// 1. Draft model autoregressively suggests K tokens
	draftTokens := make([]int, 0, sm.draftScale)
	draftLogits := make([][]float32, 0, sm.draftScale)

	currentPos := seq.Pos
	seqIDStr := fmt.Sprintf("seq-%d", seq.ID)

	for i := 0; i < sm.draftScale; i++ {
		// Sample one token from Draft model
		token, logits, err := sm.draftEngine.InferWithLogits(seq.Tokens, 1, seq.Config)
		if err != nil {
			return err
		}
		draftTokens = append(draftTokens, token[0])
		draftLogits = append(draftLogits, logits)
		
		// Temporarily advance seq.Tokens for next draft step
		seq.Tokens = append(seq.Tokens, token[0])
	}

	// 2. Target model evaluates all draft tokens simultaneously
	// Standard speculative decoding: Target evaluates P(x_i | x_{<i}, drafted_{<i})
	targetLogits, err := sm.targetEngine.ForwardDraft(seq.Tokens)
	if err != nil {
		return err
	}

	// 3. Rejection Sampling
	acceptedCount := 0
	for i := 0; i < len(draftTokens); i++ {
		if i >= len(targetLogits) {
			break
		}
		pTarget := targetLogits[i] 
		qDraft := draftLogits[i]
		draftedToken := draftTokens[i]

		// Acceptance probability: min(1, P(x)/Q(x))
		if draftedToken >= len(pTarget) || draftedToken >= len(qDraft) {
			break
		}
		p_x := pTarget[draftedToken]
		q_x := qDraft[draftedToken]
		
		accepted := false
		if q_x > 0 && p_x/q_x >= 1.0 {
			accepted = true
		} else if q_x > 0 {
			// Random acceptance (mock for now, should use rand.Float32())
			accepted = true 
		}

		if accepted {
			acceptedCount++
		} else {
			break
		}
	}

	// 4. Handle Rollback
	if acceptedCount < len(draftTokens) {
		rollbackAmount := len(draftTokens) - acceptedCount
		seq.Tokens = seq.Tokens[:len(seq.Tokens)-rollbackAmount]
		
		// Release KV blocks
		_ = sm.targetEngine.RollbackKV(seqIDStr, currentPos+acceptedCount)
		_ = sm.draftEngine.RollbackKV(seqIDStr, currentPos+acceptedCount)
	}

	seq.Pos = currentPos + acceptedCount
	return nil
}
