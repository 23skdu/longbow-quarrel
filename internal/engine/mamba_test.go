//go:build darwin && metal

package engine

import (
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/device"
)

func TestMambaWeightsStructure(t *testing.T) {
	// Verify MambaWeights struct exists and fields are accessible
	mw := &MambaWeights{
		A:            &device.Tensor{},
		D:            &device.Tensor{},
		Conv1dWeight: &device.Tensor{},
	}

	if mw.A == nil {
		t.Error("MambaWeights.A should not be nil")
	}
}

func TestIsMambaLayer(t *testing.T) {
	// Mock Engine with populated Mamba weights
	e := &Engine{
		Weights: &LlamaWeights{
			Mamba: make([]*MambaWeights, 10),
		},
	}

	// Case 1: Nil weight (standard Transformer layer in hybrid model)
	if e.IsMambaLayer(0) {
		t.Error("Layer 0 should not be Mamba (nil weights)")
	}

	// Case 2: Populated weight (Mamba layer)
	e.Weights.Mamba[1] = &MambaWeights{}
	if !e.IsMambaLayer(1) {
		t.Error("Layer 1 should be detected as Mamba")
	}

	// Case 3: Out of bounds
	if e.IsMambaLayer(99) {
		t.Error("Layer 99 should not be Mamba (out of bounds)")
	}
}
