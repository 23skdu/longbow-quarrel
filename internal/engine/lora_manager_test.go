package engine

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

// TestLoRAManagerLoadAdapter covers the LoRA manager's loading path.
//
// It is the Metal and CUDA path: the manager reads the adapter from a GGUF
// file and puts the A and B matrices on the device.
func TestLoRAManagerLoadAdapter(t *testing.T) {
	const (
		dimIn  = 4
		rank   = 2
		dimOut = 3
	)

	dir := t.TempDir()
	path := buildLoRAFile(t, dir, "adapter.gguf", dimIn, rank, dimOut, 8.0)

	ctx := device.NewContext()
	defer ctx.Free()

	lm := NewLoRAManager()
	if err := lm.LoadAdapter(ctx, path, "adapter-1"); err != nil {
		t.Fatalf("LoadAdapter failed: %v", err)
	}

	weights, ok := lm.GetWeights("adapter-1", "blk.0.attn_q")
	if !ok {
		t.Fatal("the adapter was not recorded")
	}
	if weights.Rank != rank {
		t.Errorf("the rank is %d, want %d", weights.Rank, rank)
	}
	if weights.Alpha != 8.0 {
		t.Errorf("the alpha is %v, want 8", weights.Alpha)
	}
	if weights.A == nil || weights.B == nil {
		t.Fatal("the adapter has no weights")
	}
	if weights.A.Cols() != dimIn || weights.A.Rows() != rank {
		t.Errorf("the A matrix is %dx%d, want %dx%d", weights.A.Rows(), weights.A.Cols(), rank, dimIn)
	}
	if weights.B.Cols() != rank || weights.B.Rows() != dimOut {
		t.Errorf("the B matrix is %dx%d, want %dx%d", weights.B.Rows(), weights.B.Cols(), dimOut, rank)
	}

	// The weights are the ones from the file.
	aData := weights.A.ToHostF32()
	if len(aData) != dimIn*rank {
		t.Fatalf("the A matrix has %d values, want %d", len(aData), dimIn*rank)
	}
	for i := range aData {
		want := float32(i%5) / 10.0
		if math.Abs(float64(aData[i]-want)) > 0.01 {
			t.Errorf("the A value %d is %v, want %v", i, aData[i], want)
		}
	}

	// A second adapter with the same id is refused.
	if err := lm.LoadAdapter(ctx, path, "adapter-1"); err == nil {
		t.Error("expected an error when the adapter is already loaded")
	}
}

// TestLoRAManagerGetWeights covers the lookup of an adapter that is not
// there.
func TestLoRAManagerGetWeights(t *testing.T) {
	lm := NewLoRAManager()

	if _, ok := lm.GetWeights("", "blk.0.attn_q"); ok {
		t.Error("an empty adapter id must not resolve")
	}
	if _, ok := lm.GetWeights("missing", "blk.0.attn_q"); ok {
		t.Error("a missing adapter must not resolve")
	}
}

// TestIsNeededTensor covers the tensor filter that decides which weights
// are loaded from a model file.
func TestIsNeededTensor(t *testing.T) {
	needed := []string{
		"token_embd.weight",
		"output_norm.weight",
		"output.weight",
		"model.embed_tokens.weight",
		"blk.0.attn_q.weight",
		"blk.0.attn_k.weight",
		"blk.0.attn_v.weight",
		"blk.0.attn_output.weight",
		"blk.0.ffn_gate.weight",
		"blk.0.ffn_up.weight",
		"blk.0.ffn_down.weight",
		"blk.0.ffn_norm.weight",
	}
	for _, name := range needed {
		if !isNeededTensor(name) {
			t.Errorf("the tensor %q should be loaded", name)
		}
	}

	notNeeded := []string{
		"blk.0.attn_q.bias",
		"blk.0.some_other.weight",
		"random_tensor",
	}
	for _, name := range notNeeded {
		if isNeededTensor(name) {
			t.Errorf("the tensor %q should not be loaded", name)
		}
	}
}

// TestMergeLoRADeltaKeys covers the mapping from a LoRA tensor name to the
// weight it merges into.
func TestMergeLoRADeltaKeys(t *testing.T) {
	const (
		dimIn  = 2
		dimOut = 2
	)

	// The keys are the ones the adapter loader maps the tensor names to.
	keys := []string{
		"attn_q", "attn_k", "attn_v", "attn_o",
		"ffn_gate", "ffn_up", "ffn_down",
	}

	for _, key := range keys {
		w := &CPUWeights{
			AttnQ:   [][]float32{make([]float32, dimOut*dimIn)},
			AttnK:   [][]float32{make([]float32, dimOut*dimIn)},
			AttnV:   [][]float32{make([]float32, dimOut*dimIn)},
			AttnO:   [][]float32{make([]float32, dimOut*dimIn)},
			FfnGate: [][]float32{make([]float32, dimOut*dimIn)},
			FfnUp:   [][]float32{make([]float32, dimOut*dimIn)},
			FfnDown: [][]float32{make([]float32, dimOut*dimIn)},
		}
		e := &CPUEngine{weights: w}

		delta := make([]float32, dimOut*dimIn)
		for i := range delta {
			delta[i] = 1
		}
		e.mergeLoRADelta(0, key, delta)

		// One of the weights must have changed.
		changed := false
		for _, slice := range [][]float32{w.AttnQ[0], w.AttnK[0], w.AttnV[0], w.AttnO[0], w.FfnGate[0], w.FfnUp[0], w.FfnDown[0]} {
			if slice[0] != 0 {
				changed = true
			}
		}
		if !changed {
			t.Errorf("the key %q did not merge into any weight", key)
		}
	}

	// An unknown key merges into nothing.
	w := &CPUWeights{AttnQ: [][]float32{make([]float32, 4)}}
	e := &CPUEngine{weights: w}
	e.mergeLoRADelta(0, "not_a_key", []float32{1, 1, 1, 1})
	if w.AttnQ[0][0] != 0 {
		t.Error("an unknown key merged into a weight")
	}

	// An engine with no weights does not crash.
	empty := &CPUEngine{}
	empty.mergeLoRADelta(0, "attn_q", []float32{1})
}

// TestGGUFWriterWritesTheDimensionsTheLoaderExpects checks that the
// writer and the loader agree on the dimension order, since the LoRA
// loaders depend on it.
func TestGGUFWriterWritesTheDimensionsTheLoaderExpects(t *testing.T) {
	dir := t.TempDir()
	path := buildLoRAFile(t, dir, "adapter.gguf", 4, 2, 3, 8.0)

	f, err := gguf.LoadFile(path)
	if err != nil {
		t.Fatalf("failed to load the adapter: %v", err)
	}
	defer func() { _ = f.Close() }()

	for _, t2 := range f.Tensors {
		switch t2.Name {
		case "blk.0.attn_q.lora_A.weight":
			// The GGUF dimensions are the row length first: [rank, dimIn].
			if len(t2.Dimensions) != 2 || t2.Dimensions[0] != 2 || t2.Dimensions[1] != 4 {
				t.Errorf("the A matrix has the dimensions %v, want [2 4]", t2.Dimensions)
			}
		case "blk.0.attn_q.lora_B.weight":
			if len(t2.Dimensions) != 2 || t2.Dimensions[0] != 3 || t2.Dimensions[1] != 2 {
				t.Errorf("the B matrix has the dimensions %v, want [3 2]", t2.Dimensions)
			}
		}
	}

	_ = filepath.Join(dir, "unused")
}
