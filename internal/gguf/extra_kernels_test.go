package gguf

import (
	"testing"
)

func TestDequantizeQ3K_Kernel(t *testing.T) {
	// Block size 256. Q3_K is 110 bytes.
	data := make([]byte, 110)
	res := DequantizeQ3K(data, 256)
	if len(res) != 256 {
		t.Fatalf("expected 256 elements, got %d", len(res))
	}
}

func TestDequantizeQ5_0_Kernel(t *testing.T) {
	// Block size 32. Q5_0 is 22 bytes (2-byte f16 delta + 4-byte high bits uint32 + 16-byte low nibbles).
	data := make([]byte, 22)
	res := DequantizeQ5_0(data, 32)
	if len(res) != 32 {
		t.Fatalf("expected 32 elements, got %d", len(res))
	}
}

func TestDequantizeQ4KBranchless_Kernel(t *testing.T) {
	data := make([]byte, 144)
	res := DequantizeQ4KBranchless(data, 256)
	if len(res) != 256 {
		t.Fatalf("expected 256 elements, got %d", len(res))
	}
}

func TestTurboQuant_Matrices(t *testing.T) {
	f := &GGUFFile{
		KV:      make(map[string]interface{}),
		Tensors: []*TensorInfo{},
	}
	// Missing matrices
	_, _, err := f.GetTurboQuantMatrices()
	if err == nil {
		t.Error("expected error for missing matrices")
	}

	// Mock some data
	rotData := make([]byte, 4096*4) // Float32
	qjlData := make([]byte, 4096*4)
	f.Tensors = append(f.Tensors, &TensorInfo{
		Name:       "turboquant.rotation_matrix",
		Type:       GGMLTypeF32,
		Data:       rotData,
		Dimensions: []uint64{4096},
	})
	f.Tensors = append(f.Tensors, &TensorInfo{
		Name:       "turboquant.qjl_matrix",
		Type:       GGMLTypeF32,
		Data:       qjlData,
		Dimensions: []uint64{4096},
	})

	rot, qjl, err := f.GetTurboQuantMatrices()
	if err != nil {
		t.Errorf("failed to get matrices: %v", err)
	}
	if len(rot) != 4096 || len(qjl) != 4096 {
		t.Errorf("unexpected lengths: rot=%d, qjl=%d", len(rot), len(qjl))
	}
}
