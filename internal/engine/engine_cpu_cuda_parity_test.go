//go:build linux && amd64 && cuda && cgo

package engine

import (
	"math"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
)

// TestCPUAndCUDAAgree runs the same model on the CPU and on the GPU and
// compares the logits.
//
// The two backends use different kernels for the attention, the
// normalisation and the feed-forward, so the results are not bit
// identical. They do have to agree, and a weight laid out differently, or
// a kernel that reads the wrong tensor, shows up as a large difference.
//
// The Q4_K codec was once decoded differently by the CPU and the GPU, and
// nothing compared them: the reference implementation of the format was
// correct and the production decoders were not.
func TestCPUAndCUDAAgree(t *testing.T) {
	dir := t.TempDir()
	path := buildTestGGUF(t, dir)

	cfg := config.Default()
	cfg.KVCacheSize = 128
	cfg.SeqLen = 128
	// Every layer on the GPU, so that the GPU's own kernels are
	// exercised rather than its CPU fallback.
	cfg.NumGPULayers = -1

	cpuCreated, err := NewCPUEngine(path, cfg)
	if err != nil {
		t.Fatalf("failed to create the CPU engine: %v", err)
	}
	defer cpuCreated.Close()
	cpuEngine := cpuCreated.(*CPUEngine)

	cudaCreated, err := NewcudaEngine(path, cfg)
	if err != nil {
		t.Fatalf("failed to create the CUDA engine: %v", err)
	}
	defer cudaCreated.Close()
	cudaEngine := cudaCreated.(*cudaEngine)

	// The batched forward is the path the inference loop uses.
	batch := &BatchDescriptor{
		Sequences:   []*Sequence{{ID: 1, Tokens: []int{1, 2}, PromptLen: 2}},
		Tokens:      []int{1, 2},
		Offsets:     []int{0},
		ContextLens: []int{0},
		TokenToSeq:  []int{0, 0},
	}
	cpuBatch, err := cpuEngine.ForwardBatch(batch)
	if err != nil {
		t.Fatalf("the CPU batched forward failed: %v", err)
	}
	gpuBatch, err := cudaEngine.ForwardBatch(batch)
	if err != nil {
		t.Fatalf("the GPU batched forward failed: %v", err)
	}
	defer gpuBatch[0].Free()

	cpuLogits := cpuBatch[0].ToHostF32()
	gpuLogits := gpuBatch[0].ToHostF32()
	if len(cpuLogits) != len(gpuLogits) {
		t.Fatalf("the CPU produced %d logits and the GPU %d", len(cpuLogits), len(gpuLogits))
	}

	// The scale of the logits sets the tolerance.
	scale := 0.0
	for _, v := range cpuLogits {
		if a := math.Abs(float64(v)); a > scale {
			scale = a
		}
	}
	if scale == 0 {
		scale = 1
	}
	tolerance := scale * 0.05

	for i := range cpuLogits {
		cpuValue := float64(cpuLogits[i])
		gpuValue := float64(gpuLogits[i])
		// A NaN never compares greater than anything, so the difference
		// has to be checked for it explicitly. The test passed with a
		// NaN on the GPU before, because NaN > tolerance is false.
		if math.IsNaN(cpuValue) || math.IsNaN(gpuValue) {
			t.Fatalf("the logit %d is %v on the CPU and %v on the GPU", i, cpuLogits[i], gpuLogits[i])
		}
		diff := math.Abs(cpuValue - gpuValue)
		if diff > tolerance {
			t.Errorf("the logit %d is %v on the CPU and %v on the GPU (difference %v, tolerance %v)",
				i, cpuLogits[i], gpuLogits[i], diff, tolerance)
		}
	}
}

// TestCPUAndCUDAAgreeOnTheGreedyToken checks that the two backends pick
// the same token, which is the property a user cares about.
func TestCPUAndCUDAAgreeOnTheGreedyToken(t *testing.T) {
	dir := t.TempDir()
	path := buildTestGGUF(t, dir)

	cfg := config.Default()
	cfg.KVCacheSize = 128
	cfg.SeqLen = 128
	cfg.NumGPULayers = -1

	cpuCreated, err := NewCPUEngine(path, cfg)
	if err != nil {
		t.Fatalf("failed to create the CPU engine: %v", err)
	}
	defer cpuCreated.Close()
	cpuEngine := cpuCreated.(*CPUEngine)

	cudaCreated, err := NewcudaEngine(path, cfg)
	if err != nil {
		t.Fatalf("failed to create the CUDA engine: %v", err)
	}
	defer cudaCreated.Close()
	cudaEngine := cudaCreated.(*cudaEngine)

	batch := &BatchDescriptor{
		Sequences:   []*Sequence{{ID: 1, Tokens: []int{1, 2}, PromptLen: 2}},
		Tokens:      []int{1, 2},
		Offsets:     []int{0},
		ContextLens: []int{0},
		TokenToSeq:  []int{0, 0},
	}
	cpuBatch, err := cpuEngine.ForwardBatch(batch)
	if err != nil {
		t.Fatalf("the CPU batched forward failed: %v", err)
	}
	gpuBatch, err := cudaEngine.ForwardBatch(batch)
	if err != nil {
		t.Fatalf("the GPU batched forward failed: %v", err)
	}
	defer gpuBatch[0].Free()

	cpuLogits := cpuBatch[0].ToHostF32()
	gpuLogits := gpuBatch[0].ToHostF32()

	cpuBest, gpuBest := 0, 0
	for i := range cpuLogits {
		if cpuLogits[i] > cpuLogits[cpuBest] {
			cpuBest = i
		}
		if gpuLogits[i] > gpuLogits[gpuBest] {
			gpuBest = i
		}
	}
	if cpuBest != gpuBest {
		t.Errorf("the CPU picked the token %d and the GPU the token %d", cpuBest, gpuBest)
	}
}
