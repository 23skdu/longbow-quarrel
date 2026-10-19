package engine

import (
	"math"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
)

// TestCPUWeightsMatVec covers the matrix-vector product that every
// projection in the CPU engine goes through.
//
// It was at 22% coverage: only the float32 weight path was exercised, and
// only through the inference path.
func TestCPUWeightsMatVec(t *testing.T) {
	w := &CPUWeights{}

	// A 2x3 matrix multiplied by a 3-vector.
	weight := []float32{
		1, 2, 3,
		4, 5, 6,
	}
	vector := []float32{1, 1, 1}

	got := w.MatVec(weight, nil, vector)
	if len(got) != 2 {
		t.Fatalf("expected 2 results, got %d", len(got))
	}
	if math.Abs(float64(got[0]-6)) > 1e-5 {
		t.Errorf("got[0] = %v, want 6", got[0])
	}
	if math.Abs(float64(got[1]-15)) > 1e-5 {
		t.Errorf("got[1] = %v, want 15", got[1])
	}

	// An empty vector yields no result.
	if got := w.MatVec(weight, nil, nil); got != nil {
		t.Errorf("expected no result for an empty vector, got %v", got)
	}

	// No weight and no raw tensor yields no result.
	if got := w.MatVec(nil, nil, vector); got != nil {
		t.Errorf("expected no result without a weight, got %v", got)
	}
}

// TestCPUWeightsMatVec_LargeMatrix exercises the blocked path, which is used
// once the matrix has at least 32 rows.
func TestCPUWeightsMatVec_LargeMatrix(t *testing.T) {
	const rows, cols = 40, 8

	weight := make([]float32, rows*cols)
	vector := make([]float32, cols)
	for i := range weight {
		weight[i] = float32(i%3) - 1
	}
	for i := range vector {
		vector[i] = float32(i) + 1
	}

	w := &CPUWeights{}
	got := w.MatVec(weight, nil, vector)
	if len(got) != rows {
		t.Fatalf("expected %d results, got %d", rows, len(got))
	}

	// Compare against a straightforward reference.
	for r := 0; r < rows; r++ {
		var want float32
		for c := 0; c < cols; c++ {
			want += weight[r*cols+c] * vector[c]
		}
		if math.Abs(float64(got[r]-want)) > 1e-4 {
			t.Errorf("result %d = %v, want %v", r, got[r], want)
		}
	}
}

// TestNewEngine_BadPath covers the public entry point, which every binary
// calls.
//
// It was at 0% coverage.
func TestNewEngine_BadPath(t *testing.T) {
	// A model path that does not exist must produce an error rather than a
	// nil engine: a nil engine would be dereferenced by the caller.
	e, err := NewEngine("/nonexistent/model.gguf", defaultTestConfig())
	if err == nil {
		t.Fatalf("expected an error for a missing model, got engine %T", e)
	}
	if e != nil {
		t.Errorf("expected a nil engine on failure, got %T", e)
	}
	_ = e
}

// TestNewEngine_RegistryHasNoMockFallback checks that the engine registry
// does not end at the mock engine.
//
// The mock used to be the last entry in the fallback chain and returns token
// 42 with a nil error, so an engine-construction failure produced
// confident-looking garbage instead of an error.
func TestNewEngine_RegistryHasNoMockFallback(t *testing.T) {
	if _, registered := engineCreators["mock"]; registered {
		t.Error("the mock engine is registered; it must not be selectable " +
			"implicitly by a shipped binary")
	}

	// Every registered creator must be a real backend.
	for name := range engineCreators {
		switch name {
		case "cpu", "cuda", "metal", "tpu":
		default:
			t.Errorf("unexpected engine registered: %q", name)
		}
	}
}

// defaultTestConfig returns a small, valid configuration for the tests.
func defaultTestConfig() config.Config {
	return config.Config{
		Layers:      2,
		Dim:         64,
		Heads:       4,
		KVHeads:     2,
		HeadDim:     16,
		HiddenDim:   128,
		VocabSize:   256,
		KVCacheSize: 256,
	}
}
