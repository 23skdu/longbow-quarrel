package engine

import (
	"math"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
)

// syntheticGemma4Weights builds a CPUWeights with the tensors the Gemma 4
// layer needs.
//
// Gemma 4 is one of the two architectures the README leads with, and its
// CPU layer had 0.0% coverage.
func syntheticGemma4Weights(layers, dim, heads, kvHeads, headDim, hiddenDim int) *CPUWeights {
	w := &CPUWeights{
		AttnNorm:  make([][]float32, layers),
		AttnQ:     make([][]float32, layers),
		AttnK:     make([][]float32, layers),
		AttnV:     make([][]float32, layers),
		AttnO:     make([][]float32, layers),
		AttnQNorm: make([][]float32, layers),
		AttnKNorm: make([][]float32, layers),
		FfnGate:   make([][]float32, layers),
		FfnUp:     make([][]float32, layers),
		FfnDown:   make([][]float32, layers),
	}

	for i := 0; i < layers; i++ {
		w.AttnNorm[i] = ones(dim)
		w.AttnQNorm[i] = ones(headDim)
		w.AttnKNorm[i] = ones(headDim)

		// The projections are applied with MatVec against the normalised
		// input, so they are [outFeatures, dim] matrices. The Q and K
		// projections additionally determine headDim at runtime
		// (headDim = len(q)/heads and len(k)/kvHeads), so their shapes have
		// to agree with the configured head count.
		w.AttnQ[i] = smallMatrix(heads*headDim, dim)
		w.AttnK[i] = smallMatrix(kvHeads*headDim, dim)
		w.AttnV[i] = smallMatrix(kvHeads*headDim, dim)
		w.AttnO[i] = smallMatrix(dim, heads*headDim)

		w.FfnGate[i] = smallMatrix(hiddenDim, dim)
		w.FfnUp[i] = smallMatrix(hiddenDim, dim)
		w.FfnDown[i] = smallMatrix(dim, hiddenDim)
	}
	return w
}

func ones(n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = 1.0
	}
	return v
}

func smallMatrix(n, m int) []float32 {
	v := make([]float32, n*m)
	for i := range v {
		v[i] = float32((i%7)-3) / 16.0
	}
	return v
}

func gemma4Config(layers, dim, heads, kvHeads, headDim, hiddenDim int) config.Config {
	cfg := config.Default()
	cfg.Layers = layers
	cfg.Dim = dim
	cfg.Heads = heads
	cfg.KVHeads = kvHeads
	cfg.HeadDim = headDim
	cfg.HiddenDim = hiddenDim
	cfg.Gemma4FullHeadDim = headDim
	cfg.Gemma4SlidingHeadDim = headDim
	cfg.Gemma4SlidingRoPETheta = 10000.0
	cfg.Gemma4FullRoPETheta = 10000.0
	cfg.Gemma4SharedKVLayers = 2
	return cfg
}

// TestIsGemma4SlidingLayer covers the sliding/full layer pattern helper.
func TestIsGemma4SlidingLayer(t *testing.T) {
	// The default Gemma 4 pattern is 5 sliding layers then 1 full layer.
	for i := 0; i < 12; i++ {
		want := (i % 6) != 5
		if got := isGemma4SlidingLayer(i, nil); got != want {
			t.Errorf("isGemma4SlidingLayer(%d, nil) = %v, want %v", i, got, want)
		}
	}

	// An explicit pattern wins.
	pattern := []bool{true, false, true}
	for i := 0; i < 6; i++ {
		want := pattern[i%len(pattern)]
		if got := isGemma4SlidingLayer(i, pattern); got != want {
			t.Errorf("isGemma4SlidingLayer(%d, pattern) = %v, want %v", i, got, want)
		}
	}
}

// TestNewGemma4LayerBuf_SizesBuffers covers the scratch allocator.
func TestNewGemma4LayerBuf_SizesBuffers(t *testing.T) {
	const dim, headDim, heads, hiddenDim, ctxLen = 64, 16, 4, 128, 32

	buf := NewGemma4LayerBuf(dim, headDim, heads, hiddenDim, ctxLen)
	if buf == nil {
		t.Fatal("NewGemma4LayerBuf returned nil")
	}
	checks := []struct {
		name string
		got  int
		want int
	}{
		{"Normed", len(buf.Normed), dim},
		{"NormedProj", len(buf.NormedProj), dim},
		{"NormedFFN", len(buf.NormedFFN), dim},
		{"NormedDown", len(buf.NormedDown), dim},
		{"QNormed", len(buf.QNormed), heads * headDim},
		{"AttnOut", len(buf.AttnOut), heads * headDim},
		{"Act", len(buf.Act), hiddenDim},
		{"Scores", len(buf.Scores), ctxLen},
		{"PLEGated", len(buf.PLEGated), 256},
		{"PLEProj", len(buf.PLEProj), dim},
	}
	for _, c := range checks {
		if c.got != c.want {
			t.Errorf("%s has %d elements, want %d", c.name, c.got, c.want)
		}
	}
}

// TestApplyGemma4LayerCPU_ProducesFiniteOutput exercises the Gemma 4 layer on
// the live forward path.
func TestApplyGemma4LayerCPU_ProducesFiniteOutput(t *testing.T) {
	const layers, dim, heads, kvHeads, headDim, hiddenDim = 2, 64, 4, 2, 16, 128

	w := syntheticGemma4Weights(layers, dim, heads, kvHeads, headDim, hiddenDim)
	cfg := gemma4Config(layers, dim, heads, kvHeads, headDim, hiddenDim)
	buf := NewGemma4LayerBuf(dim, headDim, heads, hiddenDim, 64)
	kv := NewCPUKVCache(layers)

	x := syntheticInput(dim, 3)
	ple := make([]float32, 256)

	out := ApplyGemma4LayerCPU(w, x, 0, 0, kv, ple, cfg, buf)

	if len(out) != dim {
		t.Fatalf("expected %d output values, got %d", dim, len(out))
	}
	for i, f := range out {
		if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
			t.Fatalf("output[%d] is %v", i, f)
		}
	}
}

// TestApplyGemma4LayerCPU_SharedKVLayer checks the shared-KV path, where a
// layer beyond Gemma4SharedKVLayers reuses an earlier layer's KV.
func TestApplyGemma4LayerCPU_SharedKVLayer(t *testing.T) {
	const layers, dim, heads, kvHeads, headDim, hiddenDim = 4, 64, 4, 2, 16, 128

	w := syntheticGemma4Weights(layers, dim, heads, kvHeads, headDim, hiddenDim)
	cfg := gemma4Config(layers, dim, heads, kvHeads, headDim, hiddenDim)
	buf := NewGemma4LayerBuf(dim, headDim, heads, hiddenDim, 64)
	kv := NewCPUKVCache(layers)

	x := syntheticInput(dim, 5)
	ple := make([]float32, 256)

	// Layer 3 is beyond the shared-KV boundary (2), so it must reuse the KV
	// of an earlier layer rather than computing its own.
	out := ApplyGemma4LayerCPU(w, x, 3, 0, kv, ple, cfg, buf)
	if len(out) != dim {
		t.Fatalf("expected %d output values, got %d", dim, len(out))
	}
	for i, f := range out {
		if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
			t.Fatalf("output[%d] is %v", i, f)
		}
	}
}

// TestApplyGemma4LayerCPU_DegenerateInputs covers the guard clauses.
func TestApplyGemma4LayerCPU_DegenerateInputs(t *testing.T) {
	const dim = 32

	w := syntheticGemma4Weights(1, dim, 2, 1, 8, 64)
	cfg := gemma4Config(1, dim, 2, 1, 8, 64)
	buf := NewGemma4LayerBuf(dim, 8, 2, 64, 16)

	if out := ApplyGemma4LayerCPU(nil, []float32{1, 2, 3}, 0, 0, nil, nil, cfg, buf); len(out) != 3 {
		t.Errorf("expected the input to be returned unchanged for a nil weights struct, got %d values", len(out))
	}
	if out := ApplyGemma4LayerCPU(w, nil, 0, 0, nil, nil, cfg, buf); len(out) != 0 {
		t.Errorf("expected an empty result for a nil input, got %d values", len(out))
	}
	if out := ApplyGemma4LayerCPU(w, []float32{1}, -1, 0, nil, nil, cfg, buf); len(out) != 1 {
		t.Errorf("expected the input to be returned unchanged for a negative layer index, got %d values", len(out))
	}
}

// TestComputeGemma4PLE covers the per-layer embedding computation.
func TestComputeGemma4PLE(t *testing.T) {
	const layers, dim = 3, 64

	w := &CPUWeights{}
	hidden := syntheticInput(dim, 9)

	ple := w.ComputeGemma4PLE(5, hidden, layers)
	if len(ple) != layers {
		t.Fatalf("expected %d PLE slices, got %d", layers, len(ple))
	}
	for i, slice := range ple {
		if len(slice) != 256 {
			t.Errorf("PLE slice %d has %d elements, want 256", i, len(slice))
		}
		for j, f := range slice {
			if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
				t.Fatalf("PLE[%d][%d] is %v", i, j, f)
			}
		}
	}
}
