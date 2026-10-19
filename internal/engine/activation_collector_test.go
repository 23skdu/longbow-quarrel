package engine

import (
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
)

// TestLayerCollectorCapturesActivations checks that the layer function
// fills the collector when one is attached.
//
// The collector is how the activation log gets the per-layer values. It
// hangs off the scratchpad, so the hot path pays one nil check per layer
// and nothing at all when the log is off.
func TestLayerCollectorCapturesActivations(t *testing.T) {
	const (
		hidden   = 4
		heads    = 2
		kvHeads  = 2
		headDim  = 2
		ffnDim   = 6
		attnDim  = heads * headDim
		kvDim    = kvHeads * headDim
		embedDim = hidden
	)

	ones := func(n int) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = 0.5
		}
		return out
	}

	w := &CPUWeights{
		AttnQ:    [][]float32{ones(hidden * attnDim)},
		AttnK:    [][]float32{ones(hidden * kvDim)},
		AttnV:    [][]float32{ones(hidden * kvDim)},
		AttnO:    [][]float32{ones(attnDim * hidden)},
		AttnNorm: [][]float32{ones(hidden)},
		FfnGate:  [][]float32{ones(hidden * ffnDim)},
		FfnUp:    [][]float32{ones(hidden * ffnDim)},
		FfnDown:  [][]float32{ones(ffnDim * hidden)},
		FfnNorm:  [][]float32{ones(hidden)},
	}

	cfg := config.Config{
		Dim:       hidden,
		Heads:     heads,
		KVHeads:   kvHeads,
		HeadDim:   headDim,
		Eps:       1e-5,
		RopeTheta: 10000,
	}

	input := make([]float32, hidden)
	for i := range input {
		input[i] = 1
	}

	// Without a collector the layer still runs.
	scratch := NewCPULayerScratch(hidden, heads, headDim)
	out := ApplyLayerCPUKVWithScratch(w, input, 0, 0, nil, cfg, scratch)
	if len(out) != hidden {
		t.Fatalf("the layer returned %d values, want %d", len(out), hidden)
	}
	if scratch.Collector != nil {
		t.Fatal("the scratch must not have a collector unless one is attached")
	}

	// With one, the activations are captured.
	collector := &LayerActivationCollector{}
	scratch.Collector = collector
	ApplyLayerCPUKVWithScratch(w, input, 0, 0, nil, cfg, scratch)

	if collector.LayerIdx != 0 {
		t.Errorf("the collector recorded layer %d, want 0", collector.LayerIdx)
	}
	if len(collector.Q) == 0 {
		t.Error("the collector did not capture the query")
	}
	if len(collector.K) == 0 {
		t.Error("the collector did not capture the key")
	}
	if len(collector.V) == 0 {
		t.Error("the collector did not capture the value")
	}
	if len(collector.AttnOut) == 0 {
		t.Error("the collector did not capture the attention output")
	}
	if len(collector.FFNOut) == 0 {
		t.Error("the collector did not capture the FFN output")
	}
}
