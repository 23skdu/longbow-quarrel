package engine

import (
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
)

// TestPromptWrapperConditions covers the chat template's condition
// evaluation, which decides whether a block is rendered.
func TestPromptWrapperConditions(t *testing.T) {
	cases := []struct {
		name string
		expr string
		want bool
		vars map[string]string
	}{
		{"a non-empty string is true", "name", true, map[string]string{"name": "llama"}},
		{"an empty string is false", "name", false, map[string]string{"name": ""}},
		{"a missing variable is false", "missing", false, map[string]string{}},
		{"not inverts the result", "not name", false, map[string]string{"name": "llama"}},
		{"and needs both", "a and b", true, map[string]string{"a": "1", "b": "2"}},
		{"and needs both to be true", "a and b", false, map[string]string{"a": "1"}},
		{"or needs one", "a or b", true, map[string]string{"b": "2"}},
		{"equality compares", `name == "llama"`, true, map[string]string{"name": "llama"}},
		{"equality compares the other way", `name == "llama"`, false, map[string]string{"name": "qwen"}},
		{"inequality compares", `name != "llama"`, true, map[string]string{"name": "qwen"}},
		{"is defined checks the map", "name is defined", true, map[string]string{"name": "llama"}},
		{"is defined checks a missing name", "name is defined", false, map[string]string{}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := evalCondition(tc.expr, tc.vars)
			if got != tc.want {
				t.Errorf("the condition %q evaluated to %v, want %v", tc.expr, got, tc.want)
			}
		})
	}
}

// TestCPUEngineCloseIsIdempotent checks that closing the engine twice is
// safe: the stop channel is closed once, and the sequence caches are
// released.
func TestCPUEngineCloseIsIdempotent(t *testing.T) {
	e := &CPUEngine{
		weights:     &CPUWeights{},
		stopChan:    make(chan struct{}),
		doneChan:    make(chan struct{}),
		seqKVCaches: map[string]*CPUKVCache{"seq-1": NewCPUKVCache(1)},
	}
	// The batch loop closes the done channel when it exits. There is no
	// loop here, so it stands in for one.
	close(e.doneChan)

	e.Close()

	select {
	case <-e.stopChan:
	default:
		t.Error("the stop channel was not closed")
	}
	if e.seqKVCaches != nil {
		t.Error("the sequence caches were not released")
	}

	// Closing again must not panic on the closed channel.
	e.Close()
}

// TestEngineConfigIsUsed checks that the engine reports its configuration.
func TestEngineConfigIsUsed(t *testing.T) {
	cfg := config.Default()
	cfg.Layers = 11
	e := &CPUEngine{config: cfg}
	if got := e.Config().Layers; got != 11 {
		t.Errorf("the engine reports %d layers, want 11", got)
	}
}

// TestCPUEngineGetSeqCachePos checks the cache position lookup.
func TestCPUEngineGetSeqCachePos(t *testing.T) {
	e := &CPUEngine{weights: &CPUWeights{}}
	if got := e.GetSeqCachePos("seq-1"); got != 0 {
		t.Errorf("the cache position is %d, want 0", got)
	}
}
