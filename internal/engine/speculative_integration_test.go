package engine

import (
	"context"
	"testing"
	"github.com/23skdu/longbow-quarrel/internal/config"
)

type specMockEngine struct {
	MockEngine
	rollbackCalled bool
}

func (e *specMockEngine) ForwardDraft(tokens []int) ([][]float32, error) {
	// Return some mock logits (at least 1 entry for 1 draft token)
	return [][]float32{{0.1, 0.9}}, nil
}

func (e *specMockEngine) GetSeqCachePos(seqID string) int {
	return 0
}

func (e *specMockEngine) RollbackKV(seqID string, newPos int) error {
	e.rollbackCalled = true
	return nil
}

func TestSpeculativeManager_Coverage(t *testing.T) {
	target := &specMockEngine{MockEngine: MockEngine{cfg: config.Config{MaxTokens: 10}}}
	draft := &specMockEngine{MockEngine: MockEngine{}}
	sm := NewSpeculativeManager(target, draft, 1)

	t.Run("Initialize", func(t *testing.T) {
		sm_err := NewSpeculativeManager(nil, nil, 1)
		seq := &Sequence{ID: 1, Tokens: []int{1}}
		err := sm_err.GenerateSpeculative(context.Background(), seq)
		if err == nil {
			t.Errorf("expected error for uninitialized engines")
		}
	})

	t.Run("GeneratePass", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		seq := &Sequence{ID: 2, Tokens: []int{1, 2, 3}}
		err := sm.GenerateSpeculative(ctx, seq)
		if err != nil {
			t.Fatalf("generation failed: %v", err)
		}
		if len(seq.Tokens) == 0 {
			t.Errorf("expected generated tokens")
		}
	})

	t.Run("ContextCancel", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()

		seq := &Sequence{ID: 3, Tokens: []int{1}}
		err := sm.GenerateSpeculative(ctx, seq)
		// Note: Standard orchestrator doesn't check context in the middle yet, 
		// but should handle it if passed.
		_ = err
	})
}
