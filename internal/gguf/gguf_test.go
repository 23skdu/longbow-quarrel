package gguf

import (
	"encoding/binary"
	"testing"
)

func TestGGUFMagic(t *testing.T) {
	if GGUFMagic != 0x46554747 {
		t.Errorf("expected GGUFMagic 0x46554747, got 0x%x", GGUFMagic)
	}
}

func TestGGUFVersion(t *testing.T) {
	if GGUFVersion != 3 {
		t.Errorf("expected GGUFVersion 3, got %d", GGUFVersion)
	}
}

func TestGGMLTypeConstants(t *testing.T) {
	tests := []struct {
		got  GGMLType
		want uint32
		name string
	}{
		{GGMLTypeF32, 0, "GGMLTypeF32"},
		{GGMLTypeF16, 1, "GGMLTypeF16"},
		{GGMLTypeQ4_0, 2, "GGMLTypeQ4_0"},
		{GGMLTypeQ4_1, 3, "GGMLTypeQ4_1"},
		{GGMLTypeQ5_0, 6, "GGMLTypeQ5_0"},
		{GGMLTypeQ8_0, 8, "GGMLTypeQ8_0"},
		{GGMLTypeQ2_K, 10, "GGMLTypeQ2_K"},
		{GGMLTypeQ3_K, 11, "GGMLTypeQ3_K"},
		{GGMLTypeQ4_K, 12, "GGMLTypeQ4_K"},
		{GGMLTypeQ5_K, 13, "GGMLTypeQ5_K"},
		{GGMLTypeQ6_K, 14, "GGMLTypeQ6_K"},
		{GGMLTypeQ8_K, 15, "GGMLTypeQ8_K"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if uint32(tt.got) != tt.want {
				t.Errorf("%s = %d, want %d", tt.name, tt.got, tt.want)
			}
		})
	}
}

func TestGGMLTypeString(t *testing.T) {
	tests := []struct {
		ggmlType GGMLType
		expected string
	}{
		{GGMLTypeF32, "F32"},
		{GGMLTypeF16, "F16"},
		{GGMLTypeQ4_0, "Q4_0"},
		{GGMLTypeQ4_1, "Q4_1"},
		{GGMLTypeQ5_0, "Q5_0"},
		{GGMLTypeQ8_0, "Q8_0"},
		{GGMLTypeQ2_K, "Q2_K"},
		{GGMLTypeQ3_K, "Q3_K"},
		{GGMLTypeQ4_K, "Q4_K"},
		{GGMLTypeQ5_K, "Q5_K"},
		{GGMLTypeQ6_K, "Q6_K"},
		{GGMLTypeQ8_K, "Q8_K"},
		{GGMLTypeIQ2_XXS, "IQ2_XXS"},
		{GGMLTypeIQ2_XS, "IQ2_XS"},
		{GGMLTypeIQ3_XXS, "IQ3_XXS"},
		{GGMLTypeIQ1_S, "IQ1_S"},
		{GGMLTypeIQ4_NL, "IQ4_NL"},
		{GGMLTypeIQ3_S, "IQ3_S"},
		{GGMLTypeIQ2_S, "IQ2_S"},
		{GGMLTypeIQ4_XS, "IQ4_XS"},
		{GGMLTypeIQ1_M, "IQ1_M"},
		{GGMLType(999), "UNKNOWN_TYPE_999"},
	}

	for _, tt := range tests {
		t.Run(tt.expected, func(t *testing.T) {
			if got := tt.ggmlType.String(); got != tt.expected {
				t.Errorf("GGMLType(%d).String() = %q, want %q", tt.ggmlType, got, tt.expected)
			}
		})
	}
}

func TestTensorInfoSizeBytes(t *testing.T) {
	tests := []struct {
		name       string
		dimensions []uint64
		ggmlType   GGMLType
		expected   uint64
	}{
		{"F32 1D", []uint64{100}, GGMLTypeF32, 400},
		{"F16 1D", []uint64{100}, GGMLTypeF16, 200},
		{"F32 2D", []uint64{10, 20}, GGMLTypeF32, 800},
		{"F16 2D", []uint64{10, 20}, GGMLTypeF16, 400},
		{"Q4_0 1D", []uint64{256}, GGMLTypeQ4_0, 144},
		{"Q5_0 1D", []uint64{256}, GGMLTypeQ5_0, 176},
		{"Q8_0 1D", []uint64{256}, GGMLTypeQ8_0, 272},
		{"Q4_K 1D", []uint64{256}, GGMLTypeQ4_K, 144},
		{"Q5_0 1D", []uint64{256}, GGMLTypeQ5_0, 176},
		{"Q6_K 1D", []uint64{256}, GGMLTypeQ6_K, 210},
		{"Q3_K 1D", []uint64{256}, GGMLTypeQ3_K, 110},
		{"IQ1_S 1D", []uint64{256}, GGMLTypeIQ1_S, 48},
		{"IQ2_XXS 1D", []uint64{256}, GGMLTypeIQ2_XXS, 66},
		{"IQ2_XS 1D", []uint64{256}, GGMLTypeIQ2_XS, 74},
		{"IQ2_S 1D", []uint64{256}, GGMLTypeIQ2_S, 82},
		{"IQ3_XXS 1D", []uint64{256}, GGMLTypeIQ3_XXS, 98},
		{"IQ3_S 1D", []uint64{256}, GGMLTypeIQ3_S, 110},
		{"IQ4_XS 1D", []uint64{256}, GGMLTypeIQ4_XS, 138},
		{"IQ4_NL 1D", []uint64{256}, GGMLTypeIQ4_NL, 144},
		{"IQ4_NL_2 1D", []uint64{256}, GGMLTypeIQ4_NL_2, 144},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			info := &TensorInfo{
				Name:       "test",
				Dimensions: tt.dimensions,
				Type:       tt.ggmlType,
			}
			if got := info.SizeBytes(); got != tt.expected {
				t.Errorf("SizeBytes() = %d, want %d", got, tt.expected)
			}
		})
	}
}

func TestTensorInfoSizeBytesIQTypes(t *testing.T) {
	// Test IQ4_NL with alternate type constant
	info := &TensorInfo{
		Name:       "test",
		Dimensions: []uint64{256},
		Type:       GGMLTypeIQ4_NL_2,
	}
	if got := info.SizeBytes(); got != 144 {
		t.Errorf("IQ4_NL_2 SizeBytes() = %d, want 144", got)
	}
}

func TestTensorInfoSizeBytesIQ1M(t *testing.T) {
	info := &TensorInfo{
		Name:       "test",
		Dimensions: []uint64{256},
		Type:       GGMLTypeIQ1_M,
	}
	if got := info.SizeBytes(); got != 0 {
		t.Errorf("IQ1_M SizeBytes() = %d, want 0 (not implemented)", got)
	}
}

func TestTensorInfoSizeBytesMXFP4(t *testing.T) {
	info := &TensorInfo{
		Name:       "test",
		Dimensions: []uint64{256},
		Type:       GGMLTypeMXFP4,
	}
	if got := info.SizeBytes(); got != 144 {
		t.Errorf("MXFP4 SizeBytes() = %d, want 144", got)
	}
}

func TestTensorInfoSizeBytesUnknown(t *testing.T) {
	info := &TensorInfo{
		Name:       "test",
		Dimensions: []uint64{256},
		Type:       GGMLType(100),
	}
	if got := info.SizeBytes(); got != 0 {
		t.Errorf("Unknown type SizeBytes() = %d, want 0", got)
	}
}

func TestErrInvalidMagic(t *testing.T) {
	err := ErrInvalidMagic{Magic: 0xDEADBEEF}
	expected := "invalid GGUF magic: deadbeef"
	if got := err.Error(); got != expected {
		t.Errorf("ErrInvalidMagic.Error() = %q, want %q", got, expected)
	}
}

func TestErrUnsupportedVersion(t *testing.T) {
	err := ErrUnsupportedVersion{Version: 42}
	expected := "unsupported GGUF version: 42"
	if got := err.Error(); got != expected {
		t.Errorf("ErrUnsupportedVersion.Error() = %q, want %q", got, expected)
	}
}

func TestGGUFHeader(t *testing.T) {
	header := GGUFHeader{
		Magic:       GGUFMagic,
		Version:     GGUFVersion,
		TensorCount: 10,
		KVCount:     5,
	}

	if header.Magic != 0x46554747 {
		t.Errorf("Header Magic = 0x%x, want 0x46554747", header.Magic)
	}
	if header.Version != 3 {
		t.Errorf("Header Version = %d, want 3", header.Version)
	}
	if header.TensorCount != 10 {
		t.Errorf("Header TensorCount = %d, want 10", header.TensorCount)
	}
	if header.KVCount != 5 {
		t.Errorf("Header KVCount = %d, want 5", header.KVCount)
	}
}

func TestGGUFFile(t *testing.T) {
	file := &GGUFFile{
		Header: GGUFHeader{
			Magic:       GGUFMagic,
			Version:     GGUFVersion,
			TensorCount: 0,
			KVCount:     0,
		},
		KV:      make(map[string]interface{}),
		Tensors: []*TensorInfo{},
	}

	if file.Header.Magic != GGUFMagic {
		t.Errorf("file.Header.Magic = 0x%x, want 0x%x", file.Header.Magic, GGUFMagic)
	}
	if len(file.KV) != 0 {
		t.Errorf("len(file.KV) = %d, want 0", len(file.KV))
	}
	if len(file.Tensors) != 0 {
		t.Errorf("len(file.Tensors) = %d, want 0", len(file.Tensors))
	}
}

func TestGGUFMetadataValueType(t *testing.T) {
	tests := []struct {
		got  GGUFMetadataValueType
		want uint32
		name string
	}{
		{GGUFMetadataValueTypeUint8, 0, "Uint8"},
		{GGUFMetadataValueTypeInt8, 1, "Int8"},
		{GGUFMetadataValueTypeUint16, 2, "Uint16"},
		{GGUFMetadataValueTypeInt16, 3, "Int16"},
		{GGUFMetadataValueTypeUint32, 4, "Uint32"},
		{GGUFMetadataValueTypeInt32, 5, "Int32"},
		{GGUFMetadataValueTypeFloat32, 6, "Float32"},
		{GGUFMetadataValueTypeBool, 7, "Bool"},
		{GGUFMetadataValueTypeString, 8, "String"},
		{GGUFMetadataValueTypeArray, 9, "Array"},
		{GGUFMetadataValueTypeUint64, 10, "Uint64"},
		{GGUFMetadataValueTypeInt64, 11, "Int64"},
		{GGUFMetadataValueTypeFloat64, 12, "Float64"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if uint32(tt.got) != tt.want {
				t.Errorf("%s = %d, want %d", tt.name, tt.got, tt.want)
			}
		})
	}
}

func TestTensorInfoMultiDimensional(t *testing.T) {
	// Test 3D tensor
	info := &TensorInfo{
		Name:       "test_3d",
		Dimensions: []uint64{2, 3, 4},
		Type:       GGMLTypeF32,
	}

	numElements := uint64(1)
	for _, d := range info.Dimensions {
		numElements *= d
	}
	if numElements != 24 {
		t.Errorf("numElements = %d, want 24", numElements)
	}

	expectedSize := numElements * 4 // F32 = 4 bytes
	if got := info.SizeBytes(); got != expectedSize {
		t.Errorf("SizeBytes() = %d, want %d", got, expectedSize)
	}
}

func TestTensorInfoDataField(t *testing.T) {
	info := &TensorInfo{
		Name:       "test_with_data",
		Dimensions: []uint64{10},
		Type:       GGMLTypeF32,
		Offset:     100,
		Data:       []byte{1, 2, 3, 4, 5},
	}

	if len(info.Data) != 5 {
		t.Errorf("len(info.Data) = %d, want 5", len(info.Data))
	}
	if info.Offset != 100 {
		t.Errorf("info.Offset = %d, want 100", info.Offset)
	}
}

func TestGGMLTypeIQTypes(t *testing.T) {
	// Test all IQ types are defined
	iqTypes := []GGMLType{
		GGMLTypeIQ2_XXS,
		GGMLTypeIQ2_XS,
		GGMLTypeIQ3_XXS,
		GGMLTypeIQ1_S,
		GGMLTypeIQ4_NL,
		GGMLTypeIQ3_S,
		GGMLTypeIQ2_S,
		GGMLTypeIQ4_XS,
		GGMLTypeIQ1_M,
	}

	for _, typ := range iqTypes {
		if uint32(typ) == 0 {
			t.Errorf("IQ type has value 0: %v", typ)
		}
	}
}

func TestQuantizeWeightsToQ4KNotImplemented(t *testing.T) {
	_, err := QuantizeWeightsToQ4K([]float32{1.0, 2.0}, 2)
	if err == nil {
		t.Error("QuantizeWeightsToQ4K should return error")
	}
	if err.Error() != "not implemented" {
		t.Errorf("QuantizeWeightsToQ4K error = %q, want %q", err.Error(), "not implemented")
	}
}

func TestDequantizeWeightsFromQ4K(t *testing.T) {
	// Create minimal valid Q4K block
	blockSize := 144
	data := make([]byte, blockSize)
	// Set d and dmin (2 bytes each)
	binary.LittleEndian.PutUint16(data[0:2], uint16(0x3C00)) // 1.0 in FP16
	binary.LittleEndian.PutUint16(data[2:4], uint16(0x3C00)) // 1.0 in FP16

	rows, cols := 1, 256
	result, err := DequantizeWeightsFromQ4K(data, rows, cols)
	if err != nil {
		t.Errorf("DequantizeWeightsFromQ4K returned error: %v", err)
	}
	if len(result) != rows*cols {
		t.Errorf("len(result) = %d, want %d", len(result), rows*cols)
	}
}
