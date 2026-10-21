package gguf

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

// The repository could read GGUF files but not write them, so the
// quantization functions had nowhere to put their output.

// TestWriteFileRoundTrip checks that a file written by the writer can be
// read back by the reader, with the same metadata and the same tensor
// data.
func TestWriteFileRoundTrip(t *testing.T) {
	kv := map[string]interface{}{
		"general.architecture":       "llama",
		"general.name":               "test-model",
		"llama.block_count":          uint32(2),
		"llama.embedding_length":     uint64(64),
		"llama.context_length":       uint32(128),
		"llama.rope.freq_base":       float32(10000),
		"llama.attention.head_count": uint32(4),
		"tokenizer.ggml.tokens":      []string{"<unk>", "hello", "world"},
		"tokenizer.ggml.scores":      []float32{0, -1, -2},
		"tokenizer.ggml.token_type":  []int32{2, 1, 1},
		"general.alignment":          uint32(32),
	}

	// A float32 tensor and a Q8_0 tensor.
	f32Data := make([]byte, 4*8*4)
	for i := range f32Data {
		f32Data[i] = byte(i)
	}
	// A Q8_0 tensor holds whole blocks of 32 values.
	q8Data := make([]byte, 34)

	tensors := []TensorData{
		{Name: "blk.0.weight", Dimensions: []uint64{8, 4}, Type: GGMLTypeF32, Data: f32Data},
		{Name: "blk.1.weight", Dimensions: []uint64{32}, Type: GGMLTypeQ8_0, Data: q8Data},
	}

	path := filepath.Join(t.TempDir(), "model.gguf")
	if err := WriteFile(path, kv, tensors, 0); err != nil {
		t.Fatalf("failed to write the file: %v", err)
	}

	f, err := LoadFile(path)
	if err != nil {
		t.Fatalf("failed to read the file back: %v", err)
	}
	defer func() { _ = f.Close() }()

	if f.Header.TensorCount != 2 {
		t.Errorf("the file has %d tensors, want 2", f.Header.TensorCount)
	}
	if len(f.KV) != len(kv) {
		t.Errorf("the file has %d metadata entries, want %d", len(f.KV), len(kv))
	}

	if got, ok := f.KV["general.architecture"].(string); !ok || got != "llama" {
		t.Errorf("the architecture is %v, want llama", f.KV["general.architecture"])
	}
	if got, ok := f.KV["llama.block_count"].(uint32); !ok || got != 2 {
		t.Errorf("the block count is %v, want 2", f.KV["llama.block_count"])
	}
	if got, ok := f.KV["llama.embedding_length"].(uint64); !ok || got != 64 {
		t.Errorf("the embedding length is %v, want 64", f.KV["llama.embedding_length"])
	}
	if got, ok := f.KV["llama.rope.freq_base"].(float32); !ok || got != 10000 {
		t.Errorf("the rope base is %v, want 10000", f.KV["llama.rope.freq_base"])
	}
	if got, ok := f.KV["tokenizer.ggml.tokens"].([]interface{}); !ok || len(got) != 3 {
		t.Errorf("the tokens are %v, want three of them", f.KV["tokenizer.ggml.tokens"])
	} else if got[0].(string) != "<unk>" {
		t.Errorf("the first token is %v, want <unk>", got[0])
	}
	if got, ok := f.KV["tokenizer.ggml.scores"].([]interface{}); !ok || len(got) != 3 {
		t.Errorf("the scores are %v, want three of them", f.KV["tokenizer.ggml.scores"])
	}

	if len(f.Tensors) != 2 {
		t.Fatalf("the file has %d tensors, want 2", len(f.Tensors))
	}

	first := f.Tensors[0]
	if first.Name != "blk.0.weight" {
		t.Errorf("the first tensor is %q, want blk.0.weight", first.Name)
	}
	if first.Type != GGMLTypeF32 {
		t.Errorf("the first tensor has type %v, want F32", first.Type)
	}
	if len(first.Dimensions) != 2 || first.Dimensions[0] != 8 || first.Dimensions[1] != 4 {
		t.Errorf("the first tensor has dimensions %v, want [8 4]", first.Dimensions)
	}
	if string(first.Data[:len(f32Data)]) != string(f32Data) {
		t.Error("the first tensor's data did not survive the round trip")
	}

	second := f.Tensors[1]
	if second.Name != "blk.1.weight" {
		t.Errorf("the second tensor is %q, want blk.1.weight", second.Name)
	}
	if second.Type != GGMLTypeQ8_0 {
		t.Errorf("the second tensor has type %v, want Q8_0", second.Type)
	}
	if string(second.Data[:len(q8Data)]) != string(q8Data) {
		t.Error("the second tensor's data did not survive the round trip")
	}
}

// TestWriteFileIsReproducible checks that the same input produces the same
// bytes: the metadata is written in sorted key order.
func TestWriteFileIsReproducible(t *testing.T) {
	kv := map[string]interface{}{
		"b.key": uint32(2),
		"a.key": uint32(1),
		"c.key": "three",
	}
	tensors := []TensorData{
		{Name: "t", Dimensions: []uint64{4}, Type: GGMLTypeF32, Data: make([]byte, 16)},
	}

	dir := t.TempDir()
	first := filepath.Join(dir, "a.gguf")
	second := filepath.Join(dir, "b.gguf")
	if err := WriteFile(first, kv, tensors, 0); err != nil {
		t.Fatalf("failed to write the first file: %v", err)
	}
	if err := WriteFile(second, kv, tensors, 0); err != nil {
		t.Fatalf("failed to write the second file: %v", err)
	}

	a, err := os.ReadFile(first) // #nosec G304 -- the path is a temp dir created by the test
	if err != nil {
		t.Fatalf("failed to read the first file: %v", err)
	}
	b, err := os.ReadFile(second) // #nosec G304 -- the path is a temp dir created by the test
	if err != nil {
		t.Fatalf("failed to read the second file: %v", err)
	}
	if string(a) != string(b) {
		t.Error("two writes of the same input produced different files")
	}
}

// TestWriteFileRejectsAWrongSizedTensor checks the size validation: a
// tensor whose data does not match its type and dimensions is a mistake
// that would produce an unreadable file.
func TestWriteFileRejectsAWrongSizedTensor(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.gguf")
	err := WriteFile(path, nil, []TensorData{
		{Name: "t", Dimensions: []uint64{4}, Type: GGMLTypeF32, Data: make([]byte, 8)},
	}, 0)
	if err == nil {
		t.Error("expected an error for a tensor with the wrong data size")
	}
}

// TestWriteFileUnsupportedValueType checks that an unsupported metadata
// value is reported rather than written as garbage.
func TestWriteFileUnsupportedValueType(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.gguf")
	err := WriteFile(path, map[string]interface{}{"key": struct{ A int }{1}}, nil, 0)
	if err == nil {
		t.Error("expected an error for an unsupported metadata value type")
	}
}

// TestWriteFileHeterogeneousArray checks that an array with mixed element
// types is rejected: the GGUF array type is per-array.
func TestWriteFileHeterogeneousArray(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.gguf")
	err := WriteFile(path, map[string]interface{}{
		"key": []interface{}{uint32(1), "two"},
	}, nil, 0)
	if err == nil {
		t.Error("expected an error for an array with mixed element types")
	}
}

// TestQuantizeRoundTrip checks that the quantization functions and their
// dequantizers agree, which is what makes the quantizer usable: a model
// that is quantized and then loaded has to produce the same numbers, up to
// the precision of the format.
func TestQuantizeRoundTrip(t *testing.T) {
	const n = 256

	// Q6_K.
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(i%32) - 16
	}
	q6 := QuantizeQ6K(src)
	if len(q6) != (n/256)*210 {
		t.Fatalf("the Q6_K encoding is %d bytes, want %d", len(q6), (n/256)*210)
	}
	back := DequantizeQ6K(q6, n)
	if len(back) != n {
		t.Fatalf("the Q6_K decoding produced %d values, want %d", len(back), n)
	}
	for i := range src {
		diff := float64(src[i]) - float64(back[i])
		if diff < 0 {
			diff = -diff
		}
		// Q6_K keeps six bits per value, so the step within a group of
		// sixteen is the group's range over 31, and the error is up to
		// half a step.
		if diff > 0.6 {
			t.Fatalf("the Q6_K round trip lost value %d: %v became %v", i, src[i], back[i])
		}
	}

	// FP8 E4M3, through the live scalar codec in fp8.go.
	fp8 := make([]byte, n)
	fp8Back := make([]float32, n)
	for i, v := range src {
		fp8[i] = Float32ToFP8E4M3(v)
		fp8Back[i] = FP8E4M3ToFloat32(fp8[i])
	}
	if len(fp8) != n {
		t.Fatalf("the FP8 encoding is %d bytes, want %d", len(fp8), n)
	}
	if len(fp8Back) != n {
		t.Fatalf("the FP8 decoding produced %d values, want %d", len(fp8Back), n)
	}
	for i := range src {
		diff := float64(src[i]) - float64(fp8Back[i])
		if diff < 0 {
			diff = -diff
		}
		// E4M3 has three mantissa bits, so the step near 16 is 1.
		if diff > 1.1 {
			t.Fatalf("the FP8 round trip lost value %d: %v became %v", i, src[i], fp8Back[i])
		}
	}
}

// TestFloat16RoundTrip checks the half-precision conversion the writer
// relies on.
func TestFloat16RoundTrip(t *testing.T) {
	values := []float32{0, 1, -1, 0.5, 1024, -0.0009765625, 65504}
	for _, v := range values {
		back := Float16ToFloat32(Float32ToFloat16(v))
		if math.Abs(float64(back-v)) > math.Abs(float64(v))*0.001 {
			t.Errorf("the half-precision round trip of %v produced %v", v, back)
		}
	}
}
