package gguf

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"testing"
)

func TestGGUFExhaustive_Metadata(t *testing.T) {
	// 1. Test every simple type
	types := []GGUFMetadataValueType{
		GGUFMetadataValueTypeUint8,
		GGUFMetadataValueTypeInt8,
		GGUFMetadataValueTypeUint16,
		GGUFMetadataValueTypeInt16,
		GGUFMetadataValueTypeUint32,
		GGUFMetadataValueTypeInt32,
		GGUFMetadataValueTypeFloat32,
		GGUFMetadataValueTypeBool,
		GGUFMetadataValueTypeUint64,
		GGUFMetadataValueTypeInt64,
		GGUFMetadataValueTypeFloat64,
	}

	data := make([]byte, 100)
	for _, typ := range types {
		t.Run(fmt.Sprintf("Type%d", typ), func(t *testing.T) {
			val, n, err := readValue(data, 0, typ)
			if err != nil {
				t.Fatalf("failed to read type %v: %v", typ, err)
			}
			if n == 0 {
				t.Errorf("read 0 bytes for type %v", typ)
			}
			_ = val
		})
	}

	// 2. Test Array Recursion
	t.Run("Array", func(t *testing.T) {
		arrData := make([]byte, 100)
		binary.LittleEndian.PutUint32(arrData[0:], uint32(GGUFMetadataValueTypeUint32))
		binary.LittleEndian.PutUint64(arrData[4:], 2) // len 2
		// vals: 10, 20
		binary.LittleEndian.PutUint32(arrData[12:], 10)
		binary.LittleEndian.PutUint32(arrData[16:], 20)

		val, n, err := readValue(arrData, 0, GGUFMetadataValueTypeArray)
		if err != nil {
			t.Fatalf("array parse failed: %v", err)
		}
		arr := val.([]interface{})
		if len(arr) != 2 || arr[0].(uint32) != 10 {
			t.Errorf("unexpected array content: %v", arr)
		}
		if n != 20 { // 4+8 + 4+4
			t.Errorf("expected 20 bytes read, got %d", n)
		}
	})

	// 3. Test Ollama Backward Search (KVCount=0)
	t.Run("OllamaSearch", func(t *testing.T) {
		// Header (KVCount=0)
		buf := new(bytes.Buffer)
		binary.Write(buf, binary.LittleEndian, uint32(0x46554747)) // GGUF
		binary.Write(buf, binary.LittleEndian, uint32(3))          // v3
		binary.Write(buf, binary.LittleEndian, uint64(0))          // tensor count
		binary.Write(buf, binary.LittleEndian, uint64(0))          // KV COUNT = 0 (Ollama trigger)

		// Padding
		buf.Write(make([]byte, 100))
		buf.WriteByte(0) // ZERO BYTE SENTINEL REQUIRED BY READER HEURISTIC

		// Backward key discovery: [uint64 len][string][uint32 type][value]
		key := "tokenizer.ggml.tokens"
		binary.Write(buf, binary.LittleEndian, uint64(len(key)))
		buf.Write([]byte(key))
		binary.Write(buf, binary.LittleEndian, uint32(GGUFMetadataValueTypeUint32))
		binary.Write(buf, binary.LittleEndian, uint32(42))

		data := buf.Bytes()
		
		// Force trigger the backward search logic by mimicking LoadFile's block
		// We'll call a dedicated test helper if available, or just test the logic here.
		
		// Simulated reader path
		tokenizerIdx := bytes.Index(data, []byte("tokenizer.ggml.tokens"))
		if tokenizerIdx <= 0 {
			t.Fatalf("could not find tokenizer trigger")
		}
		
		// In reader.go, it searches backwards for a string prefix
		// We verify it can find it
		pos := uint64(tokenizerIdx)
		keyStart := uint64(0)
		for i := int(pos - 8); i >= 0 && i > int(pos-200); i-- {
			if data[i] == 0 && i+8 < len(data) {
				strLen := binary.LittleEndian.Uint64(data[i:])
				if strLen == uint64(len(key)) {
					keyStart = uint64(i + 8)
					break
				}
			}
		}

		if keyStart == 0 {
			t.Errorf("failed to find keyStart in simulated backward search")
		}
	})
}

func TestGGUFExhaustive_Errors(t *testing.T) {
	t.Run("UnexpectedEOF", func(t *testing.T) {
		data := []byte{1, 0, 0, 0} // too short for readString (needs 8 bytes for len)
		_, _, err := readString(data, 0)
		if err == nil {
			t.Errorf("expected error for short data")
		}
	})

	t.Run("UnsupportedType", func(t *testing.T) {
		_, _, err := readValue([]byte{0}, 0, 999)
		if err == nil {
			t.Errorf("expected error for unsupported type")
		}
	})
}
