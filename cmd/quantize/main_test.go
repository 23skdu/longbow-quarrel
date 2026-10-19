package main

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

// buildTestModel writes a small GGUF model with two float32 tensors.
func buildTestModel(t *testing.T) (string, []float32) {
	t.Helper()

	// 256 values, so that the K-quant block sizes divide it.
	const n = 256
	values := make([]float32, n)
	for i := range values {
		values[i] = float32(i%17) - 8
	}

	data := make([]byte, n*4)
	for i, v := range values {
		bits := math.Float32bits(v)
		data[i*4] = byte(bits)
		data[i*4+1] = byte(bits >> 8)
		data[i*4+2] = byte(bits >> 16)
		data[i*4+3] = byte(bits >> 24)
	}

	kv := map[string]interface{}{
		"general.architecture":   "llama",
		"general.alignment":      uint32(32),
		"llama.block_count":      uint32(2),
		"llama.embedding_length": uint64(n),
	}
	tensors := []gguf.TensorData{
		{Name: "blk.0.attn_q.weight", Dimensions: []uint64{n, 1}, Type: gguf.GGMLTypeF32, Data: data},
		{Name: "blk.1.attn_q.weight", Dimensions: []uint64{n, 1}, Type: gguf.GGMLTypeF32, Data: data},
	}

	path := filepath.Join(t.TempDir(), "model.gguf")
	if err := gguf.WriteFile(path, kv, tensors, 32); err != nil {
		t.Fatalf("failed to write the test model: %v", err)
	}
	return path, values
}

// TestQuantizeQ4K checks the whole path: a float32 model is converted to
// Q4_K and the result can be read back and dequantized.
func TestQuantizeQ4K(t *testing.T) {
	input, values := buildTestModel(t)
	output := filepath.Join(t.TempDir(), "out.gguf")

	if err := run(input, output, "q4_k", true, false); err != nil {
		t.Fatalf("the conversion failed: %v", err)
	}

	model, err := gguf.LoadFile(output)
	if err != nil {
		t.Fatalf("failed to read the output: %v", err)
	}
	defer func() { _ = model.Close() }()

	if len(model.Tensors) != 2 {
		t.Fatalf("the output has %d tensors, want 2", len(model.Tensors))
	}
	if model.Tensors[0].Type != gguf.GGMLTypeQ4_K {
		t.Errorf("the first tensor has type %v, want Q4_K", model.Tensors[0].Type)
	}

	// The metadata survives.
	if got, ok := model.KV["general.architecture"].(string); !ok || got != "llama" {
		t.Errorf("the architecture is %v, want llama", model.KV["general.architecture"])
	}

	// The values survive, up to the precision of the format.
	back, err := gguf.DequantizeTensor(model.Tensors[0])
	if err != nil {
		t.Fatalf("failed to dequantize the output: %v", err)
	}
	if len(back) != len(values) {
		t.Fatalf("the output has %d values, want %d", len(back), len(values))
	}
	for i := range values {
		diff := math.Abs(float64(values[i] - back[i]))
		if diff > 1.0 {
			t.Fatalf("the value %d changed from %v to %v", i, values[i], back[i])
		}
	}
}

// TestQuantizeQ6KAndQ8_0 checks the other two block formats.
func TestQuantizeQ6KAndQ8_0(t *testing.T) {
	for _, tc := range []struct {
		name string
		typ  gguf.GGMLType
		tol  float64
	}{
		{"q6_k", gguf.GGMLTypeQ6_K, 0.6},
		{"q8_0", gguf.GGMLTypeQ8_0, 0.2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			input, values := buildTestModel(t)
			output := filepath.Join(t.TempDir(), "out.gguf")

			if err := run(input, output, tc.name, true, false); err != nil {
				t.Fatalf("the conversion failed: %v", err)
			}

			model, err := gguf.LoadFile(output)
			if err != nil {
				t.Fatalf("failed to read the output: %v", err)
			}
			defer func() { _ = model.Close() }()

			if model.Tensors[0].Type != tc.typ {
				t.Errorf("the first tensor has type %v, want %v", model.Tensors[0].Type, tc.typ)
			}

			back, err := gguf.DequantizeTensor(model.Tensors[0])
			if err != nil {
				t.Fatalf("failed to dequantize the output: %v", err)
			}
			for i := range values {
				diff := math.Abs(float64(values[i] - back[i]))
				if diff > tc.tol {
					t.Fatalf("the value %d changed from %v to %v", i, values[i], back[i])
				}
			}
		})
	}
}

// TestQuantizeF16 checks the half-precision target.
func TestQuantizeF16(t *testing.T) {
	input, values := buildTestModel(t)
	output := filepath.Join(t.TempDir(), "out.gguf")

	if err := run(input, output, "f16", true, false); err != nil {
		t.Fatalf("the conversion failed: %v", err)
	}

	model, err := gguf.LoadFile(output)
	if err != nil {
		t.Fatalf("failed to read the output: %v", err)
	}
	defer func() { _ = model.Close() }()

	if model.Tensors[0].Type != gguf.GGMLTypeF16 {
		t.Errorf("the first tensor has type %v, want F16", model.Tensors[0].Type)
	}

	back, err := gguf.DequantizeTensor(model.Tensors[0])
	if err != nil {
		t.Fatalf("failed to dequantize the output: %v", err)
	}
	for i := range values {
		diff := math.Abs(float64(values[i] - back[i]))
		if diff > 0.01 {
			t.Fatalf("the value %d changed from %v to %v", i, values[i], back[i])
		}
	}
}

// TestQuantizeKeepsTheEmbedding checks that the token embedding is kept at
// half precision when the flag says so.
func TestQuantizeKeepsTheEmbedding(t *testing.T) {
	const n = 256
	data := make([]byte, n*4)
	kv := map[string]interface{}{"general.architecture": "llama"}
	tensors := []gguf.TensorData{
		{Name: "token_embd.weight", Dimensions: []uint64{n, 1}, Type: gguf.GGMLTypeF32, Data: data},
	}
	input := filepath.Join(t.TempDir(), "model.gguf")
	if err := gguf.WriteFile(input, kv, tensors, 32); err != nil {
		t.Fatalf("failed to write the test model: %v", err)
	}

	output := filepath.Join(t.TempDir(), "out.gguf")
	if err := run(input, output, "q4_k", true, false); err != nil {
		t.Fatalf("the conversion failed: %v", err)
	}

	model, err := gguf.LoadFile(output)
	if err != nil {
		t.Fatalf("failed to read the output: %v", err)
	}
	defer func() { _ = model.Close() }()

	if model.Tensors[0].Type != gguf.GGMLTypeF16 {
		t.Errorf("the embedding has type %v, want F16", model.Tensors[0].Type)
	}
}

// TestQuantizeDryRun checks that the dry run writes nothing.
func TestQuantizeDryRun(t *testing.T) {
	input, _ := buildTestModel(t)
	output := filepath.Join(t.TempDir(), "out.gguf")

	if err := run(input, output, "q4_k", true, true); err != nil {
		t.Fatalf("the dry run failed: %v", err)
	}
	if _, err := gguf.LoadFile(output); err == nil {
		t.Error("the dry run wrote a file")
	}
}

// TestQuantizeUnknownTarget checks the guard.
func TestQuantizeUnknownTarget(t *testing.T) {
	input, _ := buildTestModel(t)
	output := filepath.Join(t.TempDir(), "out.gguf")
	if err := run(input, output, "nonsense", true, false); err == nil {
		t.Error("expected an error for an unknown target type")
	}
}
