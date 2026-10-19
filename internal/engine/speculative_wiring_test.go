package engine

import (
	"math/rand"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
)

// specStubEngine is a deterministic engine used to drive the speculative
// decoding path.
//
// The draft side reports a fixed token and a fixed distribution, and the
// target side reports a fixed set of distributions through ForwardDraft.
type specStubEngine struct {
	MockEngine
	token        int
	logits       []float32
	forward      [][]float32
	forwardCalls int
}

func (e *specStubEngine) InferWithLogits(tokens []int, count int, cfg SamplerConfig) ([]int, []float32, error) {
	return []int{e.token}, e.logits, nil
}

func (e *specStubEngine) ForwardDraft(tokens []int) ([][]float32, error) {
	e.forwardCalls++
	return e.forward, nil
}

// peakedLogits returns a distribution that puts almost all of its mass on
// the given token, with a realistic logit scale.
func peakedLogits(token, vocab int) []float32 {
	logits := make([]float32, vocab)
	for i := range logits {
		logits[i] = -20
	}
	if token >= 0 && token < vocab {
		logits[token] = 20
	}
	return logits
}

// TestRejectSample_AcceptsWhenTheDistributionsAgree is the regression test
// for the acceptance rule.
//
// The previous implementation compared the target's *normalized* probability
// for the token with the target's *unnormalized* exp(logit) for the same
// token. With the logit scale real models use, the unnormalized value
// dwarfs the normalized one, so the acceptance ratio collapsed towards zero
// and virtually every draft token was rejected: the scheme could not have
// produced a speedup.
//
// When the draft and the target agree, the acceptance ratio is
// min(1, p/q) = 1, so every token has to be accepted.
func TestRejectSample_AcceptsWhenTheDistributionsAgree(t *testing.T) {
	const vocab = 32
	logits := peakedLogits(7, vocab)
	rng := rand.New(rand.NewSource(1)) // #nosec G404 -- deterministic test seed

	accepted := 0
	const trials = 200
	for i := 0; i < trials; i++ {
		ok, token := rejectSample(logits, logits, 7, rng)
		if !ok {
			t.Fatalf("trial %d: the token was rejected even though the distributions agree", i)
		}
		if token != 7 {
			t.Fatalf("trial %d: got token %d, want 7", i, token)
		}
		accepted++
	}
	if accepted != trials {
		t.Errorf("accepted %d of %d tokens, want all of them", accepted, trials)
	}
}

// TestRejectSample_RejectsAndCorrects checks the rejection path: a token
// the target does not want is replaced by a draw from the residual
// distribution, which is dominated by the target's own preferred token.
func TestRejectSample_RejectsAndCorrects(t *testing.T) {
	const vocab = 32
	draft := peakedLogits(7, vocab)
	target := peakedLogits(3, vocab)
	rng := rand.New(rand.NewSource(1)) // #nosec G404 -- deterministic test seed

	ok, corrected := rejectSample(target, draft, 7, rng)
	if ok {
		t.Fatal("the token was accepted even though the target prefers another one")
	}
	if corrected != 3 {
		t.Errorf("got the corrected token %d, want 3 (the target's preferred token)", corrected)
	}
}

// TestRejectSample_MismatchedVocabularies checks the degraded mode: the two
// distributions cannot be compared, so the draft token is only accepted
// when the target ranks it first.
func TestRejectSample_MismatchedVocabularies(t *testing.T) {
	target := peakedLogits(3, 32)
	draft := peakedLogits(7, 16)       // a different vocabulary size
	rng := rand.New(rand.NewSource(1)) // #nosec G404 -- deterministic test seed

	ok, corrected := rejectSample(target, draft, 7, rng)
	if ok {
		t.Error("the token was accepted even though the target prefers another one")
	}
	if corrected != 3 {
		t.Errorf("got the corrected token %d, want 3", corrected)
	}

	// Now the target agrees with the draft token.
	target2 := peakedLogits(7, 32)
	ok, token := rejectSample(target2, draft, 7, rng)
	if !ok || token != 7 {
		t.Errorf("got (%v, %d), want (true, 7)", ok, token)
	}
}

// TestSpeculativeEngine_EmitsAcceptedDraftTokens drives the whole path: the
// draft proposes tokens, the target verifies them, and the accepted tokens
// are emitted to the caller.
func TestSpeculativeEngine_EmitsAcceptedDraftTokens(t *testing.T) {
	const vocab = 32

	draft := &specStubEngine{
		token:  7,
		logits: peakedLogits(7, vocab),
	}
	target := &specStubEngine{
		forward: [][]float32{
			peakedLogits(7, vocab),
			peakedLogits(7, vocab),
			peakedLogits(7, vocab),
			peakedLogits(7, vocab),
		},
	}

	spec := NewAsymmetricSpeculativeEngineWithOptions(target, draft, 4, 1)

	var emitted []int
	got, err := spec.InferWithCallback([]int{1, 2}, 3, SamplerConfig{Temperature: 0.7}, func(token int) {
		emitted = append(emitted, token)
	})
	if err != nil {
		t.Fatalf("the speculative inference failed: %v", err)
	}
	if len(got) != 3 {
		t.Errorf("got %d tokens, want 3", len(got))
	}
	if len(emitted) != 3 {
		t.Fatalf("the callback fired %d times, want 3", len(emitted))
	}
	for i, token := range emitted {
		if token != 7 {
			t.Errorf("emitted token %d is %d, want the accepted draft token 7", i, token)
		}
	}
	if target.forwardCalls == 0 {
		t.Error("the target engine was never asked to verify the draft tokens")
	}
}

// TestSpeculativeEngine_EmitsTheCorrectionOnRejection checks that a
// rejected draft token is replaced by the target's own token, and that the
// replacement is emitted.
//
// The previous implementation computed the correction and then dropped it:
// only the accepted prefix was appended to the sequence, so a rejection
// produced no token at all.
func TestSpeculativeEngine_EmitsTheCorrectionOnRejection(t *testing.T) {
	const vocab = 32

	draft := &specStubEngine{
		token:  7,
		logits: peakedLogits(7, vocab),
	}
	target := &specStubEngine{
		forward: [][]float32{
			peakedLogits(3, vocab),
			peakedLogits(3, vocab),
			peakedLogits(3, vocab),
			peakedLogits(3, vocab),
		},
	}

	spec := NewAsymmetricSpeculativeEngineWithOptions(target, draft, 4, 1)

	var emitted []int
	got, err := spec.InferWithCallback([]int{1, 2}, 1, SamplerConfig{Temperature: 0.7}, func(token int) {
		emitted = append(emitted, token)
	})
	if err != nil {
		t.Fatalf("the speculative inference failed: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("got %d tokens, want 1", len(got))
	}
	if len(emitted) != 1 {
		t.Fatalf("the callback fired %d times, want 1", len(emitted))
	}
	if emitted[0] != 3 {
		t.Errorf("emitted token %d, want 3 (the target's correction)", emitted[0])
	}
}

// TestNewSpeculativeEngine_Errors covers the guard rails of the wiring.
func TestNewSpeculativeEngine_Errors(t *testing.T) {
	if _, err := NewSpeculativeEngine(nil, "draft.gguf", config.Default()); err == nil {
		t.Error("expected an error when there is no target engine")
	}

	target := &specStubEngine{}
	if _, err := NewSpeculativeEngine(target, "", config.Default()); err == nil {
		t.Error("expected an error when no draft model is given")
	}

	if _, err := NewSpeculativeEngine(target, "/nonexistent/draft.gguf", config.Default()); err == nil {
		t.Error("expected an error when the draft model cannot be loaded")
	}
}

// TestSpeculativeManager_Options checks the configurable draft length and
// path count.
func TestSpeculativeManager_Options(t *testing.T) {
	sm := NewSpeculativeManagerWithOptions(nil, nil, 6, 3)
	if got := sm.GetDynamicDraftLength(); got != 6 {
		t.Errorf("the draft length is %d, want 6", got)
	}
	if got := sm.GetNumPaths(); got != 3 {
		t.Errorf("the path count is %d, want 3", got)
	}

	// A non-positive setting keeps the default.
	sm2 := NewSpeculativeManagerWithOptions(nil, nil, 0, 0)
	if got := sm2.GetDynamicDraftLength(); got != 4 {
		t.Errorf("the draft length is %d, want the default of 4", got)
	}
	if got := sm2.GetNumPaths(); got != 2 {
		t.Errorf("the path count is %d, want the default of 2", got)
	}
}
