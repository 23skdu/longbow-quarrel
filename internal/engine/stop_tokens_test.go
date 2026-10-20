package engine

import (
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
)

// TestStopTokenSet checks the lookup-set helper.
func TestStopTokenSet(t *testing.T) {
	if set := stopTokenSet(nil); set != nil {
		t.Errorf("expected a nil set for no stop tokens, got %v", set)
	}
	if set := stopTokenSet([]int{}); set != nil {
		t.Errorf("expected a nil set for an empty stop token list, got %v", set)
	}

	set := stopTokenSet([]int{7, 11, 13})
	if len(set) != 3 {
		t.Fatalf("expected 3 entries, got %d", len(set))
	}
	for _, tok := range []int{7, 11, 13} {
		if !isStopToken(set, tok) {
			t.Errorf("expected %d to be a stop token", tok)
		}
	}
	for _, tok := range []int{0, 6, 8, 12, 14} {
		if isStopToken(set, tok) {
			t.Errorf("expected %d not to be a stop token", tok)
		}
	}
	// An empty set never matches.
	if isStopToken(nil, 7) {
		t.Error("a nil stop-token set must not match")
	}
}

// TestStopTokensAreReadFromConfig guards against config.Config.StopTokens
// regressing to a declared-but-ignored field.
//
// StopTokens was declared in config.Config and never read anywhere in the
// tree: a user who configured stop tokens got them silently ignored, and
// generation only stopped at the EOS token or the token limit.
func TestStopTokensAreReadFromConfig(t *testing.T) {
	cfg := config.Default()
	cfg.StopTokens = []int{99, 100}
	if len(cfg.StopTokens) == 0 {
		t.Fatal("config.Default() should carry the configured stop tokens")
	}

	// The engines must build their lookup set from the configuration.
	set := stopTokenSet(cfg.StopTokens)
	if !isStopToken(set, 99) || !isStopToken(set, 100) {
		t.Error("the configured stop tokens are not reflected in the lookup set")
	}
}

// TestStopTokensTerminateSequence verifies that a sequence terminates early
// when a token matches seq.Config.StopTokens or engine stopTokens.
func TestStopTokensTerminateSequence(t *testing.T) {
	seq := &Sequence{
		Config: SamplerConfig{
			StopTokens: []int{42},
		},
	}
	engineStopTokens := stopTokenSet([]int{99})

	// Non-matching token
	tok := 10
	if isStopToken(engineStopTokens, tok) || isStopToken(stopTokenSet(seq.Config.StopTokens), tok) {
		t.Fatalf("token %d should not be a stop token", tok)
	}

	// Engine stop token
	tok = 99
	if !(isStopToken(engineStopTokens, tok) || isStopToken(stopTokenSet(seq.Config.StopTokens), tok)) {
		t.Fatalf("token %d should be recognised as engine stop token", tok)
	}

	// Per-sequence stop token from SamplerConfig
	tok = 42
	if !(isStopToken(engineStopTokens, tok) || isStopToken(stopTokenSet(seq.Config.StopTokens), tok)) {
		t.Fatalf("token %d should be recognised as per-sequence stop token", tok)
	}
}
