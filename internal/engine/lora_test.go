package engine

import (
	"math"
	"path/filepath"
	"testing"
)

// TestCPUEngineMergeLoRADelta covers the LoRA merge, which is how a LoRA
// adapter is applied on the CPU backend.
//
// The delta is A x B scaled by alpha/rank and added to the base weight.
func TestCPUEngineMergeLoRADelta(t *testing.T) {
	e := &CPUEngine{
		weights: &CPUWeights{
			AttnQ: [][]float32{{1, 2, 3, 4}},
			AttnK: [][]float32{{5, 6, 7, 8}},
		},
	}

	// A delta of the same length is added to the target.
	e.mergeLoRADelta(0, "attn_q", []float32{10, 20, 30, 40})
	got := e.weights.AttnQ[0]
	want := []float32{11, 22, 33, 44}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Errorf("AttnQ[0][%d] = %v, want %v", i, got[i], want[i])
		}
	}

	// The other projection must be untouched.
	if e.weights.AttnK[0][0] != 5 {
		t.Errorf("AttnK was modified: %v", e.weights.AttnK[0])
	}

	// A delta of the wrong length is ignored.
	e.mergeLoRADelta(0, "attn_q", []float32{1, 2})
	if e.weights.AttnQ[0][0] != 11 {
		t.Errorf("a mismatched delta was applied: %v", e.weights.AttnQ[0])
	}

	// An out-of-range layer or an unknown key is ignored.
	e.mergeLoRADelta(99, "attn_q", []float32{1, 1, 1, 1})
	e.mergeLoRADelta(0, "not_a_key", []float32{1, 1, 1, 1})
	if e.weights.AttnQ[0][0] != 11 {
		t.Errorf("an out-of-range merge was applied: %v", e.weights.AttnQ[0])
	}
}

// TestCPUEngineMergeLoRADelta_NilWeights checks the nil guard.
func TestCPUEngineMergeLoRADelta_NilWeights(t *testing.T) {
	e := &CPUEngine{}

	// Must not panic with a nil weights struct.
	e.mergeLoRADelta(0, "attn_q", []float32{1, 2, 3, 4})

	// The engine must be left in a consistent state: no weights were
	// created, and the adapter registry is still usable.
	if e.weights != nil {
		t.Errorf("expected the weights to stay nil, got %T", e.weights)
	}
	if got := e.ListAdapters(); len(got) != 0 {
		t.Errorf("expected no adapters, got %v", got)
	}
}

// TestCPUEngineListAdapters covers the adapter registry that backs the
// /v1/adapters/list endpoint.
func TestCPUEngineListAdapters(t *testing.T) {
	e := &CPUEngine{}

	if got := e.ListAdapters(); len(got) != 0 {
		t.Errorf("expected no adapters, got %v", got)
	}

	e.adapters = []AdapterInfo{
		{ID: "zeta", Path: "/z.gguf", NumWeights: 2},
		{ID: "alpha", Path: "/a.gguf", NumWeights: 1},
	}

	got := e.ListAdapters()
	if len(got) != 2 {
		t.Fatalf("expected 2 adapters, got %d", len(got))
	}
	// The list must be sorted by ID.
	if got[0].ID != "alpha" || got[1].ID != "zeta" {
		t.Errorf("expected the adapters sorted by ID, got %v", got)
	}
	// The returned slice must be a copy: mutating it must not affect the
	// engine's registry.
	got[0].ID = "mutated"
	if e.adapters[1].ID != "alpha" {
		t.Error("ListAdapters returned the engine's own slice instead of a copy")
	}
}

// TestLoRAManagerListAdapters covers the LoRA manager's registry.
func TestLoRAManagerListAdapters(t *testing.T) {
	lm := NewLoRAManager()
	if got := lm.ListAdapters(); len(got) != 0 {
		t.Errorf("expected no adapters, got %v", got)
	}

	lm.adapters["b"] = &LoRAAdapter{ID: "b", Path: "/b.gguf", Weights: map[string]*LoRAWeight{}}
	lm.adapters["a"] = &LoRAAdapter{ID: "a", Path: "/a.gguf", Weights: map[string]*LoRAWeight{}}

	got := lm.ListAdapters()
	if len(got) != 2 {
		t.Fatalf("expected 2 adapters, got %d", len(got))
	}
	if got[0].ID != "a" || got[1].ID != "b" {
		t.Errorf("expected the adapters sorted by ID, got %v", got)
	}
	if got[0].Path != "/a.gguf" {
		t.Errorf("expected the path to be reported, got %q", got[0].Path)
	}
}

// TestLoadGGUFChatTemplate covers the chat-template loader.
func TestLoadGGUFChatTemplate(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.gguf")

	if err := writeGGUFWithChatTemplate(path, "{% for message in messages %}{{ message['content'] }}{% endfor %}"); err != nil {
		t.Fatalf("failed to write the model file: %v", err)
	}

	tmpl, err := LoadGGUFChatTemplate(path)
	if err != nil {
		t.Fatalf("LoadGGUFChatTemplate failed: %v", err)
	}
	if tmpl == "" {
		t.Error("expected the chat template to be read from the model")
	}

	// A model without a chat template yields an empty string, not an error.
	plain := filepath.Join(dir, "plain.gguf")
	if err := writeGGUFWithChatTemplate(plain, ""); err != nil {
		t.Fatalf("failed to write the model file: %v", err)
	}
	tmpl, err = LoadGGUFChatTemplate(plain)
	if err != nil {
		t.Fatalf("LoadGGUFChatTemplate failed: %v", err)
	}
	if tmpl != "" {
		t.Errorf("expected no chat template, got %q", tmpl)
	}

	// A missing file is an error.
	if _, err := LoadGGUFChatTemplate(filepath.Join(dir, "missing.gguf")); err == nil {
		t.Error("expected an error for a missing file")
	}
}
