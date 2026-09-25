package gguf

import (
	"encoding/binary"
	"math"
	"testing"
)

func TestDequantizeBlock(t *testing.T) {
	tests := []struct {
		name     string
		dataType GGMLType
		size     int
		input    []byte
	}{
		{
			name:     "F32",
			dataType: GGMLTypeF32,
			size:     4,
			input:    []byte{0, 0, 0, 0}, // 0.0
		},
		{
			name:     "F16",
			dataType: GGMLTypeF16,
			size:     2,
			input:    []byte{0, 0}, // 0.0 in F16
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, 1)
			DequantizeBlock(tt.input, dst, tt.dataType)
			if math.IsNaN(float64(dst[0])) {
				t.Errorf("DequantizeBlock(%s) produced NaN", tt.name)
			}
		})
	}
}

func TestDequantizeQ4_K(t *testing.T) {
	// 256 weights in Q4_K
	// Super-block: 8 blocks of 32
	// Each block has 16 bytes (32 weights @ 4bit)
	// Plus scales/mins
	input := make([]byte, 176) // Correct size for Q4_K block
	dst := make([]float32, 256)

	// Fill with some pattern
	for i := range input {
		input[i] = byte(i)
	}

	DequantizeBlock(input, dst, GGMLTypeQ4_K)

	// Just verify no panic and non-zero-ish values (since input is non-zero)
	foundNonZero := false
	for _, v := range dst {
		if v != 0 {
			foundNonZero = true
			break
		}
	}
	if !foundNonZero {
		t.Error("DequantizeBlock(Q4_K) produced all zeros for non-zero input")
	}
}

func TestDequantizeQ6_K(t *testing.T) {
	dst := make([]float32, 256)

	// A zeroed block decodes to all zeros (d == 0).
	DequantizeBlock(make([]byte, 210), dst, GGMLTypeQ6_K)
	for i, v := range dst {
		if v != 0 {
			t.Fatalf("zeroed Q6_K block: dst[%d] = %v, want 0", i, v)
		}
	}

	// A populated block must decode exactly like DequantizeQ6K, which is the
	// path CPU inference takes.
	block := make([]byte, 210)
	for i := 0; i < 208; i++ {
		block[i] = byte((i*53 + 7) & 0xFF) // #nosec G115 -- masked to byte range
	}
	binary.LittleEndian.PutUint16(block[208:210], Float32ToFloat16(-2.5))

	DequantizeBlock(block, dst, GGMLTypeQ6_K)
	want := DequantizeQ6K(block, 256)
	for i := range want {
		if dst[i] != want[i] {
			t.Fatalf("populated Q6_K block: dst[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
	// d = -2.5 and every scale byte is 0x07+... so the block is not all zero.
	nonZero := false
	for _, v := range dst {
		if v != 0 {
			nonZero = true
			break
		}
	}
	if !nonZero {
		t.Fatal("populated Q6_K block decoded to all zeros; input data is not exercising the decoder")
	}
}

func TestDequantizeQ8_0(t *testing.T) {
	dst := make([]float32, 32)

	// Zeroed block: f16 scale is 0, so every element is 0.
	DequantizeBlock(make([]byte, 34), dst, GGMLTypeQ8_0)
	for i, v := range dst {
		if v != 0 {
			t.Fatalf("zeroed Q8_0 block: dst[%d] = %v, want 0", i, v)
		}
	}

	// Known block: f16 scale (2 bytes) followed by 32 int8 quants.
	// value[i] = scale * int8(quants[i]).
	block := make([]byte, 34)
	binary.LittleEndian.PutUint16(block[0:2], Float32ToFloat16(0.5))
	for i := 0; i < 32; i++ {
		block[2+i] = byte(i - 16) // #nosec G115 -- int8 range [-16,15]
	}
	DequantizeBlock(block, dst, GGMLTypeQ8_0)
	for i := 0; i < 32; i++ {
		want := float32(0.5) * float32(int8(i-16))
		if dst[i] != want {
			t.Fatalf("Q8_0 dst[%d] = %v, want %v", i, dst[i], want)
		}
	}
}

func TestMatVecMulQ8_0(t *testing.T) {
	// Create 2 rows, 64 cols (2 blocks per row, 68 bytes per row = 136 bytes)
	rows := 2
	cols := 64
	data := make([]byte, rows*(cols/32)*34)

	// Set scale = 1.0 (float16: 0x3C00) for all blocks
	for b := 0; b < rows*(cols/32); b++ {
		data[b*34] = 0x00
		data[b*34+1] = 0x3C
		// Set values to 1, 2, 3...
		for j := 0; j < 32; j++ {
			data[b*34+2+j] = byte(j + 1)
		}
	}

	vector := make([]float32, cols)
	for i := range vector {
		vector[i] = 1.0
	}

	res := MatVecMulQ8_0(data, vector, rows, cols)
	if len(res) != rows {
		t.Fatalf("expected %d rows, got %d", rows, len(res))
	}

	dequant := DequantizeQ8_0(data, rows*cols)
	for r := 0; r < rows; r++ {
		var expected float32
		for c := 0; c < cols; c++ {
			expected += dequant[r*cols+c] * vector[c]
		}
		diff := math.Abs(float64(res[r] - expected))
		if diff > 1e-4 {
			t.Errorf("row %d: got %f, want %f (diff %f)", r, res[r], expected, diff)
		}
	}
}

func TestDequantizeQ5K(t *testing.T) {
	data := make([]byte, 176)
	// d = 1.0 (0x3C00)
	data[0] = 0x00
	data[1] = 0x3C
	for i := 2; i < 176; i++ {
		data[i] = byte(i)
	}
	out := DequantizeQ5K(data, 256)
	if len(out) != 256 {
		t.Fatalf("expected 256 elements, got %d", len(out))
	}

	// Test panic when numElements not multiple of 256
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expected panic on non-multiple of 256")
		}
	}()
	DequantizeQ5K(data, 100)
}

func TestDequantizeQ2K(t *testing.T) {
	data := make([]byte, 84)
	// d = 1.0, dmin = 0.5
	data[80] = 0x00
	data[81] = 0x3C
	data[82] = 0x00
	data[83] = 0x38
	for i := 0; i < 80; i++ {
		data[i] = byte(i)
	}
	out := DequantizeQ2K(data, 256)
	if len(out) != 256 {
		t.Fatalf("expected 256 elements, got %d", len(out))
	}
}

func TestDequantizeIQ4XS(t *testing.T) {
	data := make([]byte, 138)
	// d = 1.0 at [136:138]
	data[136] = 0x00
	data[137] = 0x3C
	for i := 0; i < 136; i++ {
		data[i] = byte(i)
	}
	out := DequantizeIQ4XS(data, 256)
	if len(out) != 256 {
		t.Fatalf("expected 256 elements, got %d", len(out))
	}
}

func TestDequantizeBlock_AllTypes(t *testing.T) {
	types := []GGMLType{
		GGMLTypeF32, GGMLTypeF16, GGMLTypeQ4_K, GGMLTypeQ6_K,
		GGMLTypeQ8_0, GGMLTypeQ5_0, GGMLTypeQ2_K, GGMLTypeQ5_K,
		GGMLTypeIQ4_XS,
	}
	data := make([]byte, 512)
	for i := range data {
		data[i] = byte(i + 1)
	}
	dst := make([]float32, 256)
	for _, dt := range types {
		DequantizeBlock(data, dst, dt)
	}

	// Test fallback zero-fill on unsupported type
	DequantizeBlock(data, dst, GGMLType(9999))
	for _, v := range dst {
		if v != 0 {
			t.Errorf("expected zero-fill on unsupported type, got %v", v)
			break
		}
	}
}

func TestFP8_TensorHelpers(t *testing.T) {
	data := []float32{1.0, -0.5, 2.0, 0.0}
	shape := []int{2, 2}
	tensor, err := NewFP8Tensor(data, shape, FP8E4M3)
	if err != nil {
		t.Fatalf("NewFP8Tensor failed: %v", err)
	}
	if tensor.ByteSize() != 4 {
		t.Errorf("expected byte size 4, got %d", tensor.ByteSize())
	}
	f32, err := tensor.ToFloat32()
	if err != nil {
		t.Fatalf("ToFloat32 failed: %v", err)
	}
	if len(f32) != 4 {
		t.Errorf("expected 4 floats, got %d", len(f32))
	}

	if FP8E4M3SizeBytes(100) != 100 {
		t.Errorf("unexpected FP8E4M3SizeBytes")
	}
	if FP8E5M2SizeBytes(100) != 100 {
		t.Errorf("unexpected FP8E5M2SizeBytes")
	}
}

func TestGGUF_GetGapTensors(t *testing.T) {
	f := &GGUFFile{
		Data:       make([]byte, 5*1024*1024),
		DataOffset: 0,
		Tensors: []*TensorInfo{
			{
				Name:       "t1",
				Offset:     0,
				Dimensions: []uint64{64},
				Type:       GGMLTypeF32, // 256 bytes
			},
			{
				Name:       "t2",
				Offset:     2 * 1024 * 1024, // 2MB offset, gap > 1MB
				Dimensions: []uint64{64},
				Type:       GGMLTypeF32,
			},
		},
	}

	gaps := f.GetGapTensors()
	if len(gaps) != 1 {
		t.Fatalf("expected 1 gap, got %d", len(gaps))
	}
	if gaps[0].Offset != f.Tensors[0].Offset+f.Tensors[0].SizeBytes() {
		t.Errorf("unexpected gap offset: %d", gaps[0].Offset)
	}
}

func TestDequantizeIQ4_NL(t *testing.T) {
	// Block size: 18 bytes for 32 elements.
	// scale = 1.0 (float16 0x3c00)
	input := make([]byte, 18)
	input[0] = 0x00
	input[1] = 0x3c // Float16 for 1.0

	// low nibble = 8 -> kvaluesIQ4NL[8] = 1
	// high nibble = 9 -> kvaluesIQ4NL[9] = 13
	input[2] = 0x98

	out := DequantizeIQ4NL(input, 32)
	if len(out) != 32 {
		t.Fatalf("expected 32 elements, got %d", len(out))
	}
	if out[0] != 1.0 {
		t.Errorf("expected out[0] == 1.0, got %f", out[0])
	}
	if out[16] != 13.0 {
		t.Errorf("expected out[16] == 13.0, got %f", out[16])
	}
}
