package engine

import (
	"strings"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
)

// TestMockEngineIsNotRegistered guards against the mock engine re-entering
// the production fallback chain.
//
// The mock used to be registered as "mock" and was the last entry in the
// NewRegisteredEngine fallback order. Because MockEngine.Infer returns
// `count` copies of token 42 with a nil error, an engine-construction failure
// on a host without CUDA or Metal produced confident, plausible-looking
// garbage instead of an error -- the worst possible failure mode for a
// serving binary.
func TestMockEngineIsNotRegistered(t *testing.T) {
	if _, registered := engineCreators["mock"]; registered {
		t.Error("the mock engine is registered in the production registry; " +
			"it must not be selectable by a shipped binary")
	}
}

// TestNewRegisteredEngineDoesNotFallBackToMock checks the fallback chain
// directly: with every real backend failing, the call must return an error
// rather than a mock engine.
func TestNewRegisteredEngineDoesNotFallBackToMock(t *testing.T) {
	// Temporarily replace every registered creator with one that fails, so
	// the fallback chain is exercised end to end.
	original := engineCreators
	engineCreators = map[string]EngineCreator{
		"metal": func(string, config.Config) (Engine, error) { return nil, errFakeBackend },
		"cuda":  func(string, config.Config) (Engine, error) { return nil, errFakeBackend },
		"cpu":   func(string, config.Config) (Engine, error) { return nil, errFakeBackend },
	}
	defer func() { engineCreators = original }()

	e, err := NewRegisteredEngine("/nonexistent/model.gguf", config.Default())
	if err == nil {
		t.Fatalf("expected an error when every backend fails, got engine %T", e)
	}
	if e != nil {
		t.Errorf("expected a nil engine on failure, got %T", e)
	}
	if !strings.Contains(err.Error(), "failed to create engine") {
		t.Errorf("expected the error to mention the failing backends, got: %v", err)
	}
}

var errFakeBackend = errFake("simulated backend failure")

type errFake string

func (e errFake) Error() string { return string(e) }
