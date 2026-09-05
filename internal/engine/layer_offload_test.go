package engine

import (
	"math"
	"math/rand"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/metrics"
)

func TestApplyLayerCPU_Basic(t *testing.T) {
	dim := 64
	numHeads := 4
	kvHeads := 4
	headDim := dim / numHeads
	hiddenDim := 128

	cfg := config.Config{
		Dim:       dim,
		HiddenDim: hiddenDim,
		Layers:    2,
		Heads:     numHeads,
		KVHeads:   kvHeads,
		HeadDim:   headDim,
		Eps:       1e-5,
	}

	w := &CPUWeights{
		AttnNorm: make([][]float32, 2),
		AttnQ:    make([][]float32, 2),
		AttnK:    make([][]float32, 2),
		AttnV:    make([][]float32, 2),
		AttnO:    make([][]float32, 2),
		FfnNorm:  make([][]float32, 2),
		FfnGate:  make([][]float32, 2),
		FfnUp:    make([][]float32, 2),
		FfnDown:  make([][]float32, 2),
	}

	rng := rand.New(rand.NewSource(42))
	for l := 0; l < 2; l++ {
		w.AttnNorm[l] = make([]float32, dim)
		w.FfnNorm[l] = make([]float32, dim)
		for i := 0; i < dim; i++ {
			w.AttnNorm[l][i] = 1.0
			w.FfnNorm[l][i] = 1.0
		}

		w.AttnQ[l] = make([]float32, dim*dim)
		w.AttnK[l] = make([]float32, dim*dim)
		w.AttnV[l] = make([]float32, dim*dim)
		w.AttnO[l] = make([]float32, dim*dim)
		for i := range w.AttnQ[l] {
			w.AttnQ[l][i] = rng.Float32() * 0.02
			w.AttnK[l][i] = rng.Float32() * 0.02
			w.AttnV[l][i] = rng.Float32() * 0.02
			w.AttnO[l][i] = rng.Float32() * 0.02
		}

		w.FfnGate[l] = make([]float32, hiddenDim*dim)
		w.FfnUp[l] = make([]float32, hiddenDim*dim)
		w.FfnDown[l] = make([]float32, dim*hiddenDim)
		for i := range w.FfnGate[l] {
			w.FfnGate[l][i] = rng.Float32() * 0.02
			w.FfnUp[l][i] = rng.Float32() * 0.02
		}
		for i := range w.FfnDown[l] {
			w.FfnDown[l][i] = rng.Float32() * 0.02
		}
	}

	input := make([]float32, dim)
	for i := range input {
		input[i] = rng.Float32() * 0.5
	}

	outLayer0 := ApplyLayerCPU(w, input, 0, cfg)
	if len(outLayer0) != dim {
		t.Fatalf("expected output length %d, got %d", dim, len(outLayer0))
	}
	for i, v := range outLayer0 {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("outLayer0[%d] is NaN/Inf: %v", i, v)
		}
	}

	outLayer1 := ApplyLayerCPU(w, outLayer0, 1, cfg)
	if len(outLayer1) != dim {
		t.Fatalf("expected output length %d, got %d", dim, len(outLayer1))
	}
	for i, v := range outLayer1 {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("outLayer1[%d] is NaN/Inf: %v", i, v)
		}
	}
}

func TestLayerOffloadMetrics(t *testing.T) {
	modelName := "test-model-offload"
	totalLayers := 32
	gpuLayers := 12
	cpuLayers := totalLayers - gpuLayers

	metrics.RecordLayerOffload(modelName, gpuLayers, cpuLayers)
	// Verify metrics call executes safely without panic
}

func FuzzApplyLayerCPU(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, int(0))

	f.Fuzz(func(t *testing.T, data []byte, layerIdx int) {
		dim := 16
		if len(data) < dim {
			return
		}
		cfg := config.Config{
			Dim:       dim,
			HiddenDim: 32,
			Layers:    2,
			Heads:     2,
			KVHeads:   2,
			HeadDim:   8,
			Eps:       1e-5,
		}
		w := &CPUWeights{}
		input := make([]float32, dim)
		for i := 0; i < dim; i++ {
			input[i] = float32(int8(data[i])) * 0.1
		}

		out := ApplyLayerCPU(w, input, layerIdx%2, cfg)
		if len(out) != dim {
			t.Fatalf("out length %d != %d", len(out), dim)
		}
	})
}
