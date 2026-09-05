package gguf

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"testing"
)

func TestReader_MetadataArrays(t *testing.T) {
	modelPath := "test_reader_arrays.gguf"
	f, err := os.Create(modelPath)
	if err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	// GGUF Header
	binary.Write(f, binary.LittleEndian, uint32(GGUFMagic))
	binary.Write(f, binary.LittleEndian, uint32(3)) // Version
	binary.Write(f, binary.LittleEndian, uint64(0)) // No tensors
	binary.Write(f, binary.LittleEndian, uint64(2)) // 2 KV pairs

	// KV 1: llama.rope.freq_base_array (Array of F32)
	writeStringInTest(f, "llama.rope.freq_base_array")
	binary.Write(f, binary.LittleEndian, uint32(GGUFMetadataValueTypeArray))
	binary.Write(f, binary.LittleEndian, uint32(GGUFMetadataValueTypeFloat32))
	binary.Write(f, binary.LittleEndian, uint64(2)) // 2 elements
	binary.Write(f, binary.LittleEndian, float32(10000.0))
	binary.Write(f, binary.LittleEndian, float32(500000.0))

	// KV 2: general.tags (Array of String)
	writeStringInTest(f, "general.tags")
	binary.Write(f, binary.LittleEndian, uint32(GGUFMetadataValueTypeArray))
	binary.Write(f, binary.LittleEndian, uint32(GGUFMetadataValueTypeString))
	binary.Write(f, binary.LittleEndian, uint64(1)) // 1 element
	writeStringInTest(f, "test-tag")

	f.Close()
	defer os.Remove(modelPath)

	reader, err := LoadFile(modelPath)
	if err != nil {
		t.Fatalf("Failed to create reader: %v", err)
	}
	defer reader.Close()

	meta := reader.KV
	
	// Verify Array Float32
	if arr, ok := meta["llama.rope.freq_base_array"].([]interface{}); ok {
		if len(arr) != 2 {
			t.Errorf("Unexpected array length: %d", len(arr))
		}
		if val, ok := arr[0].(float32); !ok || val != 10000.0 {
			t.Errorf("Unexpected float value: %v", arr[0])
		}
	} else {
		t.Errorf("Metadata 'llama.rope.freq_base_array' is not []interface{}: %T", meta["llama.rope.freq_base_array"])
	}

	// Verify Array String
	if arr, ok := meta["general.tags"].([]interface{}); ok {
		if len(arr) != 1 || arr[0].(string) != "test-tag" {
			t.Errorf("Unexpected string array value: %v", arr)
		}
	} else {
		t.Errorf("Metadata 'general.tags' is not []interface{}: %T", meta["general.tags"])
	}
}

func writeStringInTest(f *os.File, s string) {
	binary.Write(f, binary.LittleEndian, uint64(len(s)))
	f.WriteString(s)
}

func TestReader_InvalidHeader(t *testing.T) {
	tmpFile := "invalid.gguf"
	os.WriteFile(tmpFile, []byte("NOT_GGUF"), 0644)
	defer os.Remove(tmpFile)

	_, err := LoadFile(tmpFile)
	if err == nil {
		t.Error("Expected error for invalid header, got nil")
	}
}

func TestReader_IncompleteFile(t *testing.T) {
	tmpFile := "short.gguf"
	var buf bytes.Buffer
	binary.Write(&buf, binary.LittleEndian, uint32(GGUFMagic))
	binary.Write(&buf, binary.LittleEndian, uint32(3))
	binary.Write(&buf, binary.LittleEndian, uint64(0))
	binary.Write(&buf, binary.LittleEndian, uint64(1)) // 1 KV expected
	// Stop here
	os.WriteFile(tmpFile, buf.Bytes(), 0644)
	defer os.Remove(tmpFile)

	_, err := LoadFile(tmpFile)
	if err == nil {
		t.Error("Expected error for truncated KV section, got nil")
	}
}

func TestReadValue_AllTypes(t *testing.T) {
	// Test primitive types: Uint8, Int8, Uint16, Int16, Uint32, Int32, Float32, Bool, Uint64, Int64, Float64
	buf := make([]byte, 64)
	buf[0] = 42
	buf[1] = 0xFE // -2 as int8

	var neg16 int16 = -1234
	var neg32 int32 = -987654
	var neg64 int64 = -123456789

	binary.LittleEndian.PutUint16(buf[2:], 1234)
	binary.LittleEndian.PutUint16(buf[4:], uint16(neg16))
	binary.LittleEndian.PutUint32(buf[6:], 987654)
	binary.LittleEndian.PutUint32(buf[10:], uint32(neg32))
	binary.LittleEndian.PutUint32(buf[14:], math.Float32bits(3.14))
	buf[18] = 1 // bool true
	binary.LittleEndian.PutUint64(buf[19:], 123456789)
	binary.LittleEndian.PutUint64(buf[27:], uint64(neg64))
	binary.LittleEndian.PutUint64(buf[35:], math.Float64bits(2.718281828))

	v, n, err := readValue(buf, 0, GGUFMetadataValueTypeUint8)
	if err != nil || v.(uint8) != 42 || n != 1 {
		t.Errorf("Uint8 failed: %v, %v, %v", v, n, err)
	}

	v, n, err = readValue(buf, 1, GGUFMetadataValueTypeInt8)
	if err != nil || v.(int8) != -2 || n != 1 {
		t.Errorf("Int8 failed: %v, %v, %v", v, n, err)
	}

	v, n, err = readValue(buf, 2, GGUFMetadataValueTypeUint16)
	if err != nil || v.(uint16) != 1234 || n != 2 {
		t.Errorf("Uint16 failed: %v, %v, %v", v, n, err)
	}

	v, n, err = readValue(buf, 4, GGUFMetadataValueTypeInt16)
	if err != nil || v.(int16) != -1234 || n != 2 {
		t.Errorf("Int16 failed: %v, %v, %v", v, n, err)
	}

	v, n, err = readValue(buf, 6, GGUFMetadataValueTypeUint32)
	if err != nil || v.(uint32) != 987654 || n != 4 {
		t.Errorf("Uint32 failed: %v, %v, %v", v, n, err)
	}

	v, n, err = readValue(buf, 10, GGUFMetadataValueTypeInt32)
	if err != nil || v.(int32) != -987654 || n != 4 {
		t.Errorf("Int32 failed: %v, %v, %v", v, n, err)
	}

	v, n, err = readValue(buf, 14, GGUFMetadataValueTypeFloat32)
	if err != nil || math.Abs(float64(v.(float32)-3.14)) > 1e-5 || n != 4 {
		t.Errorf("Float32 failed: %v, %v, %v", v, n, err)
	}

	v, n, err = readValue(buf, 18, GGUFMetadataValueTypeBool)
	if err != nil || v.(bool) != true || n != 1 {
		t.Errorf("Bool failed: %v, %v, %v", v, n, err)
	}

	v, n, err = readValue(buf, 19, GGUFMetadataValueTypeUint64)
	if err != nil || v.(uint64) != 123456789 || n != 8 {
		t.Errorf("Uint64 failed: %v, %v, %v", v, n, err)
	}

	v, n, err = readValue(buf, 27, GGUFMetadataValueTypeInt64)
	if err != nil || v.(int64) != -123456789 || n != 8 {
		t.Errorf("Int64 failed: %v, %v, %v", v, n, err)
	}

	v, n, err = readValue(buf, 35, GGUFMetadataValueTypeFloat64)
	if err != nil || math.Abs(v.(float64)-2.718281828) > 1e-6 || n != 8 {
		t.Errorf("Float64 failed: %v, %v, %v", v, n, err)
	}

	// Unsupported type
	_, _, err = readValue(buf, 0, GGUFMetadataValueType(999))
	if err == nil {
		t.Errorf("expected error for unsupported metadata type")
	}
}

func TestReader_LoadFile_EdgeCases(t *testing.T) {
	// 1. Non-existent file
	_, err := LoadFile("nonexistent_file_xyz.gguf")
	if err == nil {
		t.Errorf("expected error on nonexistent file")
	}

	// 2. File too small (<24 bytes)
	tmpTooSmall := "too_small.gguf"
	os.WriteFile(tmpTooSmall, []byte("short"), 0644)
	defer os.Remove(tmpTooSmall)
	_, err = LoadFile(tmpTooSmall)
	if err == nil {
		t.Errorf("expected error for small file")
	}

	// 3. Unsupported version (version 1)
	tmpV1 := "v1.gguf"
	var buf bytes.Buffer
	binary.Write(&buf, binary.LittleEndian, uint32(GGUFMagic))
	binary.Write(&buf, binary.LittleEndian, uint32(1)) // Version 1
	binary.Write(&buf, binary.LittleEndian, uint64(0)) // 0 tensors
	binary.Write(&buf, binary.LittleEndian, uint64(0)) // 0 KV
	os.WriteFile(tmpV1, buf.Bytes(), 0644)
	defer os.Remove(tmpV1)
	_, err = LoadFile(tmpV1)
	if err == nil {
		t.Errorf("expected error for unsupported version 1")
	}

	// 4. Tensor offset out of bounds
	tmpOutOfBounds := "out_of_bounds.gguf"
	buf.Reset()
	binary.Write(&buf, binary.LittleEndian, uint32(GGUFMagic))
	binary.Write(&buf, binary.LittleEndian, uint32(3)) // Version 3
	binary.Write(&buf, binary.LittleEndian, uint64(1)) // 1 tensor
	binary.Write(&buf, binary.LittleEndian, uint64(0)) // 0 KV
	// Tensor info:
	// name
	writeStringInTestBytes(&buf, "weight")
	// dims
	binary.Write(&buf, binary.LittleEndian, uint32(1))
	binary.Write(&buf, binary.LittleEndian, uint64(10))
	// type
	binary.Write(&buf, binary.LittleEndian, uint32(GGMLTypeF32))
	// offset: huge offset beyond file
	binary.Write(&buf, binary.LittleEndian, uint64(1000000))
	os.WriteFile(tmpOutOfBounds, buf.Bytes(), 0644)
	defer os.Remove(tmpOutOfBounds)
	_, err = LoadFile(tmpOutOfBounds)
	if err == nil {
		t.Errorf("expected error for tensor offset out of bounds")
	}

	// 5. Valid file with a tensor and alignment as uint64
	tmpValid := "valid_tensor.gguf"
	buf.Reset()
	binary.Write(&buf, binary.LittleEndian, uint32(GGUFMagic))
	binary.Write(&buf, binary.LittleEndian, uint32(3))
	binary.Write(&buf, binary.LittleEndian, uint64(1)) // 1 tensor
	binary.Write(&buf, binary.LittleEndian, uint64(1)) // 1 KV
	// KV: general.alignment uint64 = 32
	writeStringInTestBytes(&buf, "general.alignment")
	binary.Write(&buf, binary.LittleEndian, uint32(GGUFMetadataValueTypeUint64))
	binary.Write(&buf, binary.LittleEndian, uint64(32))
	// Tensor info:
	writeStringInTestBytes(&buf, "layer.0.weight")
	binary.Write(&buf, binary.LittleEndian, uint32(1))
	binary.Write(&buf, binary.LittleEndian, uint64(4)) // 4 floats = 16 bytes
	binary.Write(&buf, binary.LittleEndian, uint32(GGMLTypeF32))
	binary.Write(&buf, binary.LittleEndian, uint64(0)) // offset 0 from data start
	// Pad buffer to alignment + tensor data (16 bytes)
	curLen := buf.Len()
	pad := 32 - (curLen % 32)
	if pad != 32 {
		buf.Write(make([]byte, pad))
	}
	// Write tensor data (4 floats: 16 bytes)
	buf.Write(make([]byte, 16))
	os.WriteFile(tmpValid, buf.Bytes(), 0644)
	defer os.Remove(tmpValid)

	gf, err := LoadFile(tmpValid)
	if err != nil {
		t.Fatalf("failed to load valid file: %v", err)
	}
	defer gf.Close()
	if len(gf.Tensors) != 1 {
		t.Errorf("expected 1 tensor, got %d", len(gf.Tensors))
	}
}

func writeStringInTestBytes(buf *bytes.Buffer, s string) {
	binary.Write(buf, binary.LittleEndian, uint64(len(s)))
	buf.WriteString(s)
}

