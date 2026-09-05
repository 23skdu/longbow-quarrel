//go:build !cuda && !metal && !tpu

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

	e.GetSeqCachePos("seq_1")
	// Test rollback/forward draft stubs
	e.ForwardDraft([]int{1})
	e.RollbackKV("seq_1", 0)
}
