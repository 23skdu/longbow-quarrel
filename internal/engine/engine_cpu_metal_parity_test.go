//go:build darwin && metal && cgo

package engine

import (
	"math"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
)

// TestCPUAndMetalAgree runs the same model on the CPU and on Apple Silicon
// via Metal and compares the logits element by element.
//
// The two backends use different kernels for attention, normalisation and
// feed-forward, so the outputs are not bit-identical, but must agree within
// numerical tolerance.
func TestCPUAndMetalAgree(t *testing.T) {
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

	metalCreated, err := NewMetalEngine(path, cfg)
	if err != nil {
		t.Fatalf("failed to create the Metal engine: %v", err)
	}
	defer metalCreated.Close()
	metalEngine := metalCreated.(*metalEngine)

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
	metalBatch, err := metalEngine.ForwardBatch(batch)
	if err != nil {
		t.Fatalf("the Metal batched forward failed: %v", err)
	}
	defer metalBatch[0].Free()

	cpuLogits := cpuBatch[0].ToHostF32()
	metalLogits := metalBatch[0].ToHostF32()
	if len(cpuLogits) != len(metalLogits) {
		t.Fatalf("the CPU produced %d logits and Metal %d", len(cpuLogits), len(metalLogits))
	}

	scale := 0.0
	for _, v := range cpuLogits {
		if a := math.Abs(float64(v)); a > scale {
			scale = a
		}
	}
	if scale == 0 {
		scale = 1
	}
	tolerance := scale * 0.01

	for i := range cpuLogits {
		cpuValue := float64(cpuLogits[i])
		metalValue := float64(metalLogits[i])
		if math.IsNaN(cpuValue) || math.IsNaN(metalValue) {
			t.Fatalf("the logit %d is %v on the CPU and %v on Metal", i, cpuLogits[i], metalLogits[i])
		}
		diff := math.Abs(cpuValue - metalValue)
		if diff > tolerance {
			t.Errorf("the logit %d is %v on the CPU and %v on Metal (difference %v, tolerance %v)",
				i, cpuLogits[i], metalLogits[i], diff, tolerance)
		}
	}
}

// TestCPUAndMetalAgreeOnTheGreedyToken checks that CPU and Metal pick the same token.
func TestCPUAndMetalAgreeOnTheGreedyToken(t *testing.T) {
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

	metalCreated, err := NewMetalEngine(path, cfg)
	if err != nil {
		t.Fatalf("failed to create the Metal engine: %v", err)
	}
	defer metalCreated.Close()
	metalEngine := metalCreated.(*metalEngine)

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
	metalBatch, err := metalEngine.ForwardBatch(batch)
	if err != nil {
		t.Fatalf("the Metal batched forward failed: %v", err)
	}
	defer metalBatch[0].Free()

	cpuLogits := cpuBatch[0].ToHostF32()
	metalLogits := metalBatch[0].ToHostF32()

	cpuBest, metalBest := 0, 0
	for i := range cpuLogits {
		if cpuLogits[i] > cpuLogits[cpuBest] {
			cpuBest = i
		}
		if metalLogits[i] > metalLogits[metalBest] {
			metalBest = i
		}
	}
	if cpuBest != metalBest {
		t.Errorf("the CPU picked the token %d and Metal the token %d", cpuBest, metalBest)
	}
}
