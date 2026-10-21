//go:build !cuda && !metal

package engine

import (
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
)

func TestCPUEngine_Extra_Coverage(t *testing.T) {
	// We already have some CPUEngine tests, adding specific method coverage
	e := &CPUEngine{
		config: config.Config{VocabSize: 100},
	}

	// The sequence does not exist, so the position is zero.
	if got := e.GetSeqCachePos("seq_1"); got != 0 {
		t.Errorf("the cache position of an unknown sequence is %d, want 0", got)
	}

	// The draft forward needs a model: without one there is nothing to
	// draft, and the caller sees an empty result.
	draft, err := e.ForwardDraft([]int{1})
	if err != nil {
		t.Errorf("the draft forward of an engine without a model failed: %v", err)
	}
	if len(draft) != 0 {
		t.Errorf("the draft forward of an engine without a model returned %d drafts", len(draft))
	}

	// The rollback of an unknown sequence is a no-op: there is nothing to
	// roll back and nothing to report.
	if err := e.RollbackKV("seq_1", 0); err != nil {
		t.Errorf("the rollback of an unknown sequence failed: %v", err)
	}
}
