package engine

import (
	"context"
	"testing"
	"github.com/23skdu/longbow-quarrel/internal/config"
)

type specMockEngine struct {
	mockEngine
	rollbackCalled bool
}

func (e *specMockEngine) ForwardDraft(tokens []int) ([][]float32, error) {
	// Return some mock logits
	return [][]float32{{0.1, 0.9}}, nil
}

func (e *specMockEngine) RollbackKV(seqID int, stepCount int) {
	e.rollbackCalled = true
}

func TestSpeculativeManager_Coverage(t *testing.T) {
	target := &specMockEngine{mockEngine: mockEngine{cfg: config.Config{MaxTokens: 10}}}
	draft := &specMockEngine{mockEngine: mockEngine{}}
	sm := NewSpeculativeManager(target, draft, 1)

	t.Run("Initialize", func(t *testing.T) {
		sm_err := NewSpeculativeManager(nil, nil, 1)
		_, err := sm_err.GenerateSpeculative(context.Background(), []int{1})
		if err == nil {
			t.Errorf("expected error for uninitialized engines")
		}
	})

	t.Run("GeneratePass", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		tokens, err := sm.GenerateSpeculative(ctx, []int{1, 2, 3})
		if err != nil {
			t.Fatalf("generation failed: %v", err)
		}
		if len(tokens) == 0 {
			t.Errorf("expected generated tokens")
		}
	})

	t.Run("ContextCancel", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()

		_, err := sm.GenerateSpeculative(ctx, []int{1})
		if err == nil {
			t.Errorf("expected context error")
		}
	})
}
