//go:build metal

package engine

import (
	"testing"
)

func TestMultiLoRA_MetadataPropagation(t *testing.T) {
	mgr := NewContinuousBatchManager()

	// 1. Submit two requests with different adapters
	mgr.Submit(&InferenceRequest{
		ID:        1,
		Prompt:    []int{1, 2, 3},
		AdapterID: "adapter-a",
	})
	mgr.Submit(&InferenceRequest{
		ID:        2,
		Prompt:    []int{4, 5},
		AdapterID: "adapter-b",
	})

	// 2. Step to admit them
	desc, err := mgr.Step(4, nil, nil)
	if err != nil {
		t.Fatalf("Step failed: %v", err)
	}

	if len(desc.Sequences) != 2 {
		t.Fatalf("Expected 2 sequences, got %d", len(desc.Sequences))
	}

	// 3. Verify AdapterIDs in BatchDescriptor
	if len(desc.AdapterIDs) != 2 {
		t.Fatalf("Expected 2 AdapterIDs, got %d", len(desc.AdapterIDs))
	}

	if desc.AdapterIDs[0] != "adapter-a" {
		t.Errorf("Expected adapter-a for seq 0, got %s", desc.AdapterIDs[0])
	}
	if desc.AdapterIDs[1] != "adapter-b" {
		t.Errorf("Expected adapter-b for seq 1, got %s", desc.AdapterIDs[1])
	}

	// 4. Verify propagation to decoding stage
	// Move them to running
	for _, seq := range desc.Sequences {
		seq.Pos = seq.PromptLen
		seq.PrefillCompleted = true
	}

	// Clear prefill for next step simulation
	mgr.prefill = make(map[uint64]*Sequence)
	for _, seq := range desc.Sequences {
		mgr.running[seq.ID] = seq
	}

	desc2, _ := mgr.Step(4, nil, nil)
	if len(desc2.AdapterIDs) != 2 {
		t.Fatalf("Expected 2 AdapterIDs in decoding step, got %d", len(desc2.AdapterIDs))
	}
	
	// IDs might be reordered if they are in a map, but here we expect consistency
	foundA := false
	foundB := false
	for _, id := range desc2.AdapterIDs {
		if id == "adapter-a" { foundA = true }
		if id == "adapter-b" { foundB = true }
	}
	if !foundA || !foundB {
		t.Errorf("One or more adapters lost in decoding step: foundA=%v, foundB=%v", foundA, foundB)
	}
}
