package engine

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/metrics"
	"github.com/23skdu/longbow-quarrel/internal/promtest"
)

// TestActivationLoggerProducesALog checks that the activation log is
// actually written and contains what it claims to.
//
// The logger existed, was instantiated by the Metal engine, and had its
// methods called by nothing: the log was never produced.
func TestActivationLoggerProducesALog(t *testing.T) {
	al := NewActivationLogger()
	if al.IsEnabled() {
		t.Fatal("a fresh logger must be disabled")
	}

	al.Enable("hello", []int{1, 2, 3})
	if !al.IsEnabled() {
		t.Fatal("the logger must be enabled after Enable")
	}

	al.LogEmbedding([]float32{0.5, -0.25, 0.125})

	collector := &LayerActivationCollector{
		LayerIdx: 3,
		Q:        []float32{1, 2, 3},
		K:        []float32{-1, 0.5},
		V:        []float32{0.25, 0.75},
		AttnOut:  []float32{2, 2},
		FFNOut:   []float32{float32(math.NaN()), 1},
	}
	logLayerActivations(al, collector)

	al.LogLogits([]float32{0.1, 0.9, 0.5}, topTokenIDs([]float32{0.1, 0.9, 0.5}, 2))

	path := filepath.Join(t.TempDir(), "activations.json")
	if err := al.SaveToFile(path); err != nil {
		t.Fatalf("failed to save the activation log: %v", err)
	}

	raw, err := os.ReadFile(path) // #nosec G304 -- the path is a temp dir created by the test
	if err != nil {
		t.Fatalf("failed to read the activation log: %v", err)
	}

	var log ActivationLog
	if err := json.Unmarshal(raw, &log); err != nil {
		t.Fatalf("the activation log is not valid JSON: %v", err)
	}

	if len(log.Embedding) != 3 {
		t.Errorf("the log has %d embedding values, want 3", len(log.Embedding))
	}
	if len(log.Layers) != 1 {
		t.Fatalf("the log has %d layers, want 1", len(log.Layers))
	}

	layer := log.Layers[0]
	if layer.Idx != 3 {
		t.Errorf("the layer index is %d, want 3", layer.Idx)
	}
	if layer.QMax != 3 {
		t.Errorf("the Q maximum is %v, want 3", layer.QMax)
	}
	if layer.KMax != 1 {
		t.Errorf("the K maximum is %v, want 1", layer.KMax)
	}
	if layer.AttnOutMax != 2 {
		t.Errorf("the attention output maximum is %v, want 2", layer.AttnOutMax)
	}
	if layer.FFNOutMax != 1 {
		t.Errorf("the FFN output maximum is %v, want 1", layer.FFNOutMax)
	}
	// The NaN in the FFN output has to be counted, and it must not
	// become the reported maximum.
	if layer.FFNNaNCount != 1 {
		t.Errorf("the FFN NaN count is %d, want 1", layer.FFNNaNCount)
	}
	if len(layer.QSample) != 3 {
		t.Errorf("the Q sample has %d values, want 3", len(layer.QSample))
	}

	// The final logits are recorded for the highest-scoring tokens.
	if len(log.FinalLogits) != 2 {
		t.Errorf("the log has %d final logits, want 2", len(log.FinalLogits))
	}
	if v, ok := log.FinalLogits["1"]; !ok || v != 0.9 {
		t.Errorf("the final logit for token 1 is %v (present: %v), want 0.9", v, ok)
	}
}

// TestActivationLoggerDisabledWritesNothing checks that a disabled logger
// refuses to write a file.
func TestActivationLoggerDisabledWritesNothing(t *testing.T) {
	al := NewActivationLogger()
	path := filepath.Join(t.TempDir(), "activations.json")
	if err := al.SaveToFile(path); err == nil {
		t.Error("expected an error when there is nothing to save")
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Error("a disabled logger must not create a file")
	}
}

// TestAuditLogitsRecordsTheMetrics checks that the logit audit reaches the
// Prometheus metrics.
//
// The audit functions and the "logit range audit" metrics both existed,
// and nothing ever connected them: the metrics were permanently zero.
func TestAuditLogitsRecordsTheMetrics(t *testing.T) {
	maxBefore := promtest.Value(t, "logit_max_value", metrics.LogitMaxValue)
	nanBefore := promtest.Value(t, "logit_nan_count_total", metrics.LogitNaNCount)
	flatBefore := promtest.Value(t, "logit_flat_distribution_total", metrics.LogitFlatDistribution)

	// A healthy distribution.
	audit := auditLogits([]float32{1, 2, 3, 4})
	if audit.HasNaN || audit.HasInf {
		t.Error("a healthy distribution was reported as broken")
	}
	promtest.RequireDelta(t, "logit_max_value", metrics.LogitMaxValue, maxBefore, 1)

	// A broken one: all the same value is a flat distribution.
	flat := auditLogits([]float32{0.001, 0.001, 0.001, 0.001})
	if !flat.IsFlat {
		t.Error("a flat distribution was not detected")
	}
	promtest.RequireDelta(t, "logit_flat_distribution_total", metrics.LogitFlatDistribution, flatBefore, 1)

	// A broken one: a NaN.
	broken := auditLogits([]float32{1, 2, float32(math.NaN()), 4})
	if !broken.HasNaN || broken.NumNaNs != 1 {
		t.Errorf("the NaN was not detected: %+v", broken)
	}
	promtest.RequireDelta(t, "logit_nan_count_total", metrics.LogitNaNCount, nanBefore, 1)
}
