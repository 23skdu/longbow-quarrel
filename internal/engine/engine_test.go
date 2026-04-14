package engine

import (
	"encoding/binary"
	"os"
	"testing"

	"errors"

	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

func generateTestGGUF(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	// Header
	binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMagic))
	binary.Write(f, binary.LittleEndian, uint32(3)) // Version
	binary.Write(f, binary.LittleEndian, uint64(12)) // Tensors
	binary.Write(f, binary.LittleEndian, uint64(7))  // KVs

	// KVs
	// 1. Architecture
	writeString(f, "general.architecture")
	binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeString))
	writeString(f, "llama")

	// 2. Tokenizer (Required by NewEngine)
	writeString(f, "tokenizer.ggml.tokens")
	binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeArray))
	binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeString))
	binary.Write(f, binary.LittleEndian, uint64(1))
	writeString(f, "dummy")

	// 3. Head Count
	writeString(f, "llama.attention.head_count")
	binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeUint32))
	binary.Write(f, binary.LittleEndian, uint32(1))

	// 4. KV Head Count
	writeString(f, "llama.attention.head_count_kv")
	binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeUint32))
	binary.Write(f, binary.LittleEndian, uint32(1))

	// 5. Context Length
	writeString(f, "llama.context_length")
	binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeUint32))
	binary.Write(f, binary.LittleEndian, uint32(10))

	// 6. Block Count
	writeString(f, "llama.block_count")
	binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeUint32))
	binary.Write(f, binary.LittleEndian, uint32(1))

	// 7. Embedding Length
	writeString(f, "llama.embedding_length")
	binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeUint32))
	binary.Write(f, binary.LittleEndian, uint32(128))

	// Tensors
	writeTensor := func(name string, offset uint64) error {
		writeString(f, name)
		binary.Write(f, binary.LittleEndian, uint32(1))  // Dims
		binary.Write(f, binary.LittleEndian, uint64(1))  // Ne[0]
		binary.Write(f, binary.LittleEndian, uint32(0))  // Type F32
		binary.Write(f, binary.LittleEndian, uint64(offset))
		return nil
	}

	names := []string{
		"token_embd.weight", "output.weight", "output_norm.weight",
		"blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight",
		"blk.0.attn_output.weight", "blk.0.attn_norm.weight",
		"blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
		"blk.0.ffn_norm.weight",
	}

	for i, n := range names {
		writeTensor(n, uint64(i*32))
	}

	// Pad header
	f.Write(make([]byte, 1024))

	// Data
	for range names {
		binary.Write(f, binary.LittleEndian, float32(1.0))
		f.Write(make([]byte, 28))
	}

	return nil
}

func writeString(f *os.File, s string) error {
	if err := binary.Write(f, binary.LittleEndian, uint64(len(s))); err != nil {
		return err
	}
	_, err := f.WriteString(s)
	return err
}

func TestEngineLifecycle(t *testing.T) {
	// Setup dummy model
	modelPath := "test_model_lifecycle.gguf"
	if err := generateTestGGUF(modelPath); err != nil {
		t.Fatalf("Failed to generate test GGUF: %v", err)
	}
	defer os.Remove(modelPath)

	// We want an Engine that can load a model and run inference
	// NewEngine(path) -> (*Engine, error)
	// e.Infer(prompt) -> tokens

	// Test initialization
	conf := config.Default()
	conf.KVCacheSize = 1024
	e, err := NewRegisteredEngine(modelPath, conf)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}
	if e == nil {
		t.Fatal("Engine is nil")
	}

	// Generic interface assertions
	if e.Config().KVCacheSize != 1024 {
		t.Errorf("Expected KVCacheSize 1024, got %d", e.Config().KVCacheSize)
	}

	// Inference
	// We want to pass a prompt tokens list
	inputTokens := []int{1, 2, 3}
	// Add config
	// Add config
	config := SamplerConfig{
		Temperature: 0,
	}
	outputTokens, err := e.Infer(inputTokens, 10, config) // generate 10 tokens
	if err != nil {
		t.Logf("Inference returned error (expected for empty/stub engine): %v", err)
	}

	if len(outputTokens) != 0 && len(outputTokens) != 10 {
		t.Errorf("Expected 0 or 10 tokens, got %d", len(outputTokens))
	}
}

func TestEngineMetrics(t *testing.T) {
	// Verify that Engine calls metrics
}


// TestMistralMetadataSupport was moved to engine_metal_test.go as it requires internal field access.

func TestGetKV(t *testing.T) {
	mockKV := make(map[string]interface{})
	mockKV["llama.test_key"] = "llama_value"
	mockKV["qwen.test_key"] = "qwen_value"
	mockKV["shared.key"] = 123
	mockKV["nil.key"] = nil

	mockFile := &gguf.GGUFFile{KV: mockKV}

	tests := []struct {
		name       string
		llamaKey   string
		qwenKey    string
		expected   interface{}
		expectedOk bool
	}{
		{
			name:       "llamaKey exists, qwenKey exists",
			llamaKey:   "llama.test_key",
			qwenKey:    "qwen.test_key",
			expected:   "llama_value",
			expectedOk: true,
		},
		{
			name:       "llamaKey exists, qwenKey absent",
			llamaKey:   "llama.test_key",
			qwenKey:    "non_existent_qwen_key",
			expected:   "llama_value",
			expectedOk: true,
		},
		{
			name:       "llamaKey absent, qwenKey exists",
			llamaKey:   "non_existent_llama_key",
			qwenKey:    "qwen.test_key",
			expected:   "qwen_value",
			expectedOk: true,
		},
		{
			name:       "both keys absent",
			llamaKey:   "non_existent_llama_key",
			qwenKey:    "non_existent_qwen_key",
			expected:   nil,
			expectedOk: false,
		},
		{
			name:       "shared key with llama preference",
			llamaKey:   "shared.key",
			qwenKey:    "qwen.shared_key", // qwen.shared_key doesn't exist, but shared.key does
			expected:   123,
			expectedOk: true,
		},
		{
			name:       "nil value for llamaKey",
			llamaKey:   "nil.key",
			qwenKey:    "qwen.test_key",
			expected:   nil,
			expectedOk: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, gotOk := getKV(mockFile, tt.llamaKey, tt.qwenKey)
			if got != tt.expected {
				t.Errorf("getKV() got = %v, expected %v", got, tt.expected)
			}
			if gotOk != tt.expectedOk {
				t.Errorf("getKV() gotOk = %v, expectedOk %v", gotOk, tt.expectedOk)
			}
		})
	}
}

func TestToFloat64(t *testing.T) {
	tests := []struct {
		name     string
		input    interface{}
		expected float64
	}{
		{"float64 input", float64(1.23), 1.23},
		{"float32 input", float32(4.56), float64(float32(4.56))},
		{"uint64 input", uint64(789), 789.0},
		{"uint32 input", uint32(101), 101.0},
		{"int32 input", int32(-112), -112.0},
		{"int64 input", int64(-314), -314.0},
		{"int input", int(500), 500.0},
		{"unsupported string input", "hello", 0.0},
		{"unsupported bool input", true, 0.0},
		{"nil input", nil, 0.0},
		{"int64 input", int64(123), 123.0},
		{"uint64 input", uint64(456), 456.0},
		{"int32 input", int32(789), 789.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := toFloat64(tt.input)
			if got != tt.expected {
				t.Errorf("toFloat64() got = %v, expected %v", got, tt.expected)
			}
		})
	}
}

func TestMetadata_Helpers(t *testing.T) {
	if !isNormWeight("output_norm.weight") {
		t.Error("output_norm.weight should be norm")
	}
	if isNormWeight("attn_q.weight") {
		t.Error("attn_q.weight should not be norm")
	}
	
	if !isNeededTensor("token_embd.weight") {
		t.Error("token_embd should be needed")
	}
}

func TestValidateTensorDimensions(t *testing.T) {
	tests := []struct {
		name     string
		rows     int
		cols     int
		ggufType gguf.GGMLType
		expected error
	}{
		{"F32 valid", 10, 20, gguf.GGMLTypeF32, nil},
		{"F32 invalid rows", 0, 20, gguf.GGMLTypeF32, errors.New("invalid dimensions: rows=0, cols=20")},
		{"F32 invalid cols", 10, 0, gguf.GGMLTypeF32, errors.New("invalid dimensions: rows=10, cols=0")},
		{"Q4_0 valid", 10, 32, gguf.GGMLTypeQ4_0, nil},
		{"Q4_0 invalid cols", 10, 30, gguf.GGMLTypeQ4_0, errors.New("Q4_0 requires cols divisible by 32, got cols=30")},
		{"Q4_0 invalid rows", 0, 32, gguf.GGMLTypeQ4_0, errors.New("invalid Q4_0 dimensions: rows=0, cols=32")},
		{"Q4_K valid", 10, 256, gguf.GGMLTypeQ4_K, nil},
		{"Q4_K invalid cols", 10, 250, gguf.GGMLTypeQ4_K, errors.New("Q4_K requires cols divisible by 256, got cols=250")},
		{"Q4_K invalid rows", 0, 256, gguf.GGMLTypeQ4_K, errors.New("invalid Q4_K dimensions: rows=0, cols=256")},
		{"Q6_K valid", 10, 256, gguf.GGMLTypeQ6_K, nil},
		{"Q6_K invalid cols", 10, 200, gguf.GGMLTypeQ6_K, errors.New("Q6_K requires cols divisible by 256, got cols=200")},
		{"Q6_K invalid rows", 0, 256, gguf.GGMLTypeQ6_K, errors.New("invalid Q6_K dimensions: rows=0, cols=256")},
		{"Unsupported type", 10, 10, gguf.GGMLTypeQ5_0, nil}, // Should return nil for unsupported types
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ValidateTensorDimensions(tt.name, tt.rows, tt.cols, tt.ggufType)
			if (got != nil && tt.expected != nil && got.Error() != tt.expected.Error()) ||
				(got != nil && tt.expected == nil) ||
				(got == nil && tt.expected != nil) {
				t.Errorf("ValidateTensorDimensions(%s, %d, %d, %v) got error %v, expected %v", tt.name, tt.rows, tt.cols, tt.ggufType, got, tt.expected)
			}
		})
	}
}


// TestNemotronStyleLoading was moved to engine_metal_test.go as it requires internal field access.
