package gguf

import (
	"bytes"
	"encoding/binary"
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
