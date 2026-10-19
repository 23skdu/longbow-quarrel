package engine

import (
	"math"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
)

// syntheticSSMWeights builds a CPUWeights with just enough SSM tensors to
// drive the GatedDeltaNet and Mamba-2 forward passes.
//
// The Qwen 3.5 hybrid SSM path and the Gemma 4 layer are two of the
// architectures the README leads with, and both were completely untested:
// every function on their live path reported 0.0% coverage.
func syntheticSSMWeights(layers, numVHeads, headDim int) *CPUWeights {
	w := &CPUWeights{
		SSMConv1d:     make([][]float32, layers),
		SSMConv1dBias: make([][]float32, layers),
		SSMA:          make([][]float32, layers),
		SSMD:          make([][]float32, layers),
		SSMAlpha:      make([][]float32, layers),
		SSMBeta:       make([][]float32, layers),
		SSMDtWeight:   make([][]float32, layers),
		SSMDtBias:     make([][]float32, layers),
		SSMNorm:       make([][]float32, layers),
		SSMOut:        make([][]float32, layers),
		AttnGate:      make([][]float32, layers),
	}

	for i := 0; i < layers; i++ {
		// Conv1d weight: [dConv * convDim] with dConv = 4.
		convDim := numVHeads * headDim
		conv := make([]float32, 4*convDim)
		for j := range conv {
			conv[j] = float32((j%7)-3) / 8.0
		}
		w.SSMConv1d[i] = conv
		w.SSMConv1dBias[i] = make([]float32, 4*convDim)

		// A matrix: one entry per head, negative so the decay gate is stable.
		a := make([]float32, numVHeads)
		for j := range a {
			a[j] = float32(-(j + 1)) / 16.0
		}
		w.SSMA[i] = a
		w.SSMD[i] = make([]float32, numVHeads)
		w.SSMDtWeight[i] = make([]float32, numVHeads)
		w.SSMDtBias[i] = make([]float32, numVHeads)

		// The alpha and beta projections are matrices of shape
		// [numVHeads, totalProj]: they are applied with MatVec against the
		// normalised input, so their length must be numVHeads*totalProj.
		// totalProj = qSize + kSize + vSize with one K head and one Q head.
		totalProj := 2*headDim + numVHeads*headDim
		alpha := make([]float32, numVHeads*totalProj)
		beta := make([]float32, numVHeads*totalProj)
		for i := range alpha {
			alpha[i] = float32((i%9)-4) / 16.0
			beta[i] = float32((i%5)-2) / 32.0
		}
		w.SSMAlpha[i] = alpha
		w.SSMBeta[i] = beta

		// The norm weight determines headDim.
		w.SSMNorm[i] = make([]float32, headDim)
		for j := range w.SSMNorm[i] {
			w.SSMNorm[i][j] = 1.0
		}
		// The output projection maps dInner -> the model hidden size, so it
		// is a [outDim, dInner] matrix. It is applied with MatVec against
		// the SSM output, so its length must be outDim*dInner.
		dInner := numVHeads * headDim
		outDim := headDim
		w.SSMOut[i] = make([]float32, outDim*dInner)
		for j := range w.SSMOut[i] {
			w.SSMOut[i][j] = float32((j%5)-2) / 16.0
		}

		// The z gate projects the normalised input to dInner and is applied
		// multiplicatively after the RMSNorm. Without it the layer output is
		// identically zero, so it has to be populated for the layer to be
		// exercised at all.
		w.AttnGate[i] = make([]float32, dInner*totalProj)
		for j := range w.AttnGate[i] {
			w.AttnGate[i][j] = float32((j%11)-5) / 32.0
		}
	}
	return w
}

func syntheticInput(n int, seed int64) []float32 {
	x := make([]float32, n)
	// A deterministic, non-trivial pattern.
	v := seed
	for i := range x {
		v = v*6364136223846793005 + 1442695040888963407
		x[i] = float32(int64(v>>40)%2000-1000) / 250.0
	}
	return x
}

func assertFinite(t *testing.T, label string, v []float32) {
	t.Helper()
	if len(v) == 0 {
		t.Fatalf("%s: produced an empty slice", label)
	}
	for i, f := range v {
		if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
			t.Fatalf("%s[%d] is %v", label, i, f)
		}
	}
}

// TestGatedDeltaNetForward_ProducesFiniteOutput covers the Qwen 3.5
// GatedDeltaNet SSM layer.
//
// This is on the live forward path (cpu_weights.go dispatches to it via
// mambaForward) and had 0.0% coverage.
func TestGatedDeltaNetForward_ProducesFiniteOutput(t *testing.T) {
	const layers = 2
	const numVHeads = 4
	const headDim = 16

	w := syntheticSSMWeights(layers, numVHeads, headDim)
	cfg := config.Default()

	// totalProj = qSize + kSize + vSize with one K head and one Q head.
	totalProj := 2*headDim + numVHeads*headDim
	normed := syntheticInput(totalProj, 7)
	qkv := syntheticInput(totalProj, 8)

	out := w.gatedDeltaNetForward(qkv, normed, 0, nil, cfg)

	assertFinite(t, "gatedDeltaNetForward", out)
	if len(out) == 0 {
		t.Fatal("gatedDeltaNetForward produced no output")
	}
}

// TestGatedDeltaNetForward_IsStateful checks that the layer keeps SSM state
// across calls, which is the entire point of a recurrent layer: the second
// call with the same input must not produce the same output as the first.
func TestGatedDeltaNetForward_IsStateful(t *testing.T) {
	const numVHeads = 4
	const headDim = 16

	w := syntheticSSMWeights(1, numVHeads, headDim)
	cfg := config.Default()

	totalProj := 2*headDim + numVHeads*headDim
	qkv := syntheticInput(totalProj, 11)
	normed := syntheticInput(totalProj, 12)

	kv := NewCPUKVCache(1)

	first := w.gatedDeltaNetForward(qkv, normed, 0, kv, cfg)
	second := w.gatedDeltaNetForward(qkv, normed, 0, kv, cfg)

	assertFinite(t, "first", first)
	assertFinite(t, "second", second)

	same := true
	for i := range first {
		if math.Abs(float64(first[i]-second[i])) > 1e-6 {
			same = false
			break
		}
	}
	if same {
		t.Error("gatedDeltaNetForward produced identical output for two calls; " +
			"the recurrent state is not being carried over")
	}
}

// TestMambaForward_DispatchesToGatedDeltaNet checks the dispatcher picks the
// GatedDeltaNet path when alpha and beta are present.
func TestMambaForward_DispatchesToGatedDeltaNet(t *testing.T) {
	const numVHeads = 4
	const headDim = 16

	w := syntheticSSMWeights(1, numVHeads, headDim)
	cfg := config.Default()

	totalProj := 2*headDim + numVHeads*headDim
	qkv := syntheticInput(totalProj, 21)
	normed := syntheticInput(totalProj, 22)

	viaDispatcher := w.mambaForward(qkv, normed, 0, nil, cfg)
	direct := w.gatedDeltaNetForward(qkv, normed, 0, nil, cfg)

	assertFinite(t, "mambaForward", viaDispatcher)
	if len(viaDispatcher) != len(direct) {
		t.Fatalf("mambaForward returned %d values, gatedDeltaNetForward returned %d",
			len(viaDispatcher), len(direct))
	}
	for i := range direct {
		if math.Abs(float64(viaDispatcher[i]-direct[i])) > 1e-6 {
			t.Fatalf("mambaForward[%d] = %v, gatedDeltaNetForward[%d] = %v",
				i, viaDispatcher[i], i, direct[i])
		}
	}
}

// TestMamba2Forward_ProducesFiniteOutput covers the Mamba-2 SSM path, taken
// when a layer has no alpha/beta projections.
func TestMamba2Forward_ProducesFiniteOutput(t *testing.T) {
	const numVHeads = 4
	const headDim = 16

	w := syntheticSSMWeights(1, numVHeads, headDim)
	// Drop alpha and beta so the dispatcher takes the Mamba-2 path.
	w.SSMAlpha = nil
	w.SSMBeta = nil
	cfg := config.Default()

	dInner := numVHeads
	dState := 8
	totalProj := 2*dInner + 2*dState
	qkv := syntheticInput(totalProj, 31)

	out := w.mamba2Forward(qkv, 0, nil, cfg)
	assertFinite(t, "mamba2Forward", out)
}

// TestMambaForward_DispatchesToMamba2 checks the dispatcher falls back to
// Mamba-2 when alpha and beta are absent.
func TestMambaForward_DispatchesToMamba2(t *testing.T) {
	const numVHeads = 4
	const headDim = 16

	w := syntheticSSMWeights(1, numVHeads, headDim)
	w.SSMAlpha = nil
	w.SSMBeta = nil
	cfg := config.Default()

	dInner := numVHeads
	dState := 8
	totalProj := 2*dInner + 2*dState
	qkv := syntheticInput(totalProj, 41)
	normed := syntheticInput(totalProj, 42)

	viaDispatcher := w.mambaForward(qkv, normed, 0, nil, cfg)
	direct := w.mamba2Forward(qkv, 0, nil, cfg)

	assertFinite(t, "mambaForward", viaDispatcher)
	if len(viaDispatcher) != len(direct) {
		t.Fatalf("mambaForward returned %d values, mamba2Forward returned %d",
			len(viaDispatcher), len(direct))
	}
}

// TestMambaForward_EmptyInputIsSafe covers the degenerate inputs.
func TestMambaForward_EmptyInputIsSafe(t *testing.T) {
	w := syntheticSSMWeights(1, 4, 16)
	cfg := config.Default()

	if out := w.mambaForward(nil, nil, 0, nil, cfg); len(out) != 0 {
		t.Errorf("expected an empty result for empty input, got %d values", len(out))
	}
	if out := w.gatedDeltaNetForward(nil, nil, 0, nil, cfg); len(out) != 0 {
		t.Errorf("expected an empty result for empty input, got %d values", len(out))
	}
}

// TestGeluCPU covers the GELU activation, which had 0.0% coverage.
func TestGeluCPU(t *testing.T) {
	in := []float32{-3, -1, -0.5, 0, 0.5, 1, 3}
	out := make([]float32, len(in))
	for i, v := range in {
		out[i] = geluCPU(v)
		if math.IsNaN(float64(out[i])) {
			t.Fatalf("geluCPU(%v) is NaN", v)
		}
	}
	// GELU(0) == 0.
	if math.Abs(float64(out[3])) > 1e-6 {
		t.Errorf("geluCPU(0) = %v, want 0", out[3])
	}
	// GELU keeps the sign of its input.
	for i, v := range in {
		if v > 0 && out[i] <= 0 {
			t.Errorf("geluCPU(%v) = %v, expected a positive value", v, out[i])
		}
		if v < 0 && out[i] >= 0 {
			t.Errorf("geluCPU(%v) = %v, expected a negative value", v, out[i])
		}
	}
	// GELU is increasing for x >= 0. (It is *not* increasing over the whole
	// real line: it has a minimum near -0.75, so a global monotonicity check
	// would be wrong.)
	for i := 4; i < len(out); i++ {
		if out[i] < out[i-1] {
			t.Errorf("geluCPU is not increasing on [0, inf) at %d: %v < %v", i, out[i], out[i-1])
		}
	}
	// GELU is bounded above by its input for positive x (GELU(x) < x).
	for i, v := range in {
		if v > 0 && out[i] >= v {
			t.Errorf("geluCPU(%v) = %v, expected less than the input", v, out[i])
		}
	}
}
