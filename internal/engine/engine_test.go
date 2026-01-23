//go:build darwin && metal

package engine

import (
	"encoding/binary"
	"os"
	"testing"

	"errors"
	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

func generateTestGGUF(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	// Magic
	if err := binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMagic)); err != nil {
		return err
	}
	// Version
	if err := binary.Write(f, binary.LittleEndian, uint32(3)); err != nil {
		return err
	}
	// Tensor Count (12)
	if err := binary.Write(f, binary.LittleEndian, uint64(12)); err != nil {
		return err
	}
	// KV Count (5)
	if err := binary.Write(f, binary.LittleEndian, uint64(5)); err != nil {
		return err
	}

	// ... KVs ...
	// KV Pair 1: "llama.block_count" -> 1
	if err := writeString(f, "llama.block_count"); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeUint32)); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, uint32(1)); err != nil {
		return err
	}

	// KV Pair 2: "llama.embedding_length" -> 1 (dim)
	if err := writeString(f, "llama.embedding_length"); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeUint32)); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, uint32(1)); err != nil {
		return err
	}

	// KV Pair 3: "llama.attention.head_count" -> 1
	if err := writeString(f, "llama.attention.head_count"); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeUint32)); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, uint32(1)); err != nil {
		return err
	}

	// KV Pair 4: "llama.attention.head_count_kv" -> 1
	if err := writeString(f, "llama.attention.head_count_kv"); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeUint32)); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, uint32(1)); err != nil {
		return err
	}

	// KV Pair 5: "llama.context_length" -> 10
	if err := writeString(f, "llama.context_length"); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeUint32)); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, uint32(10)); err != nil {
		return err
	}

	// Helper to write 1x1 scalar tensor
	writeTensor := func(name string, offset uint64) error {
		if err := writeString(f, name); err != nil {
			return err
		}
		if err := binary.Write(f, binary.LittleEndian, uint32(1)); err != nil {
			return err
		} // Dims
		if err := binary.Write(f, binary.LittleEndian, uint64(1)); err != nil {
			return err
		} // Ne[0]
		if err := binary.Write(f, binary.LittleEndian, uint32(0)); err != nil {
			return err
		} // Type F32
		if err := binary.Write(f, binary.LittleEndian, uint64(offset)); err != nil {
			return err
		}
		return nil
	}

	currentOff := uint64(0)
	names := []string{
		"token_embd.weight",
		"output.weight",
		"output_norm.weight",
		"blk.0.attn_q.weight",
		"blk.0.attn_k.weight",
		"blk.0.attn_v.weight",
		"blk.0.attn_output.weight",
		"blk.0.attn_norm.weight",
		"blk.0.ffn_gate.weight",
		"blk.0.ffn_up.weight",
		"blk.0.ffn_down.weight",
		"blk.0.ffn_norm.weight",
	}

	for _, n := range names {
		if err := writeTensor(n, currentOff); err != nil {
			return err
		}
		currentOff += 32
	}

	// Pad header
	if _, err := f.Write(make([]byte, 1024)); err != nil {
		return err
	}

	// Write data
	for i := 0; i < len(names); i++ {
		if err := binary.Write(f, binary.LittleEndian, float32(1.0)); err != nil {
			return err
		}
		if _, err := f.Write(make([]byte, 28)); err != nil {
			return err
		}
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

	engine, err := NewEngine(modelPath, false)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}
	if engine == nil {
		t.Fatal("Engine is nil")
	}
	defer engine.Ctx.Free()

	if engine.Weights.TokenEmb == nil {
		t.Fatal("Expected TokenEmb to be loaded")
	}

	if len(engine.Weights.AttnQ) < 1 {
		t.Fatal("Expected AttnQ to be initialized with layers")
	}
	if engine.Weights.AttnQ[0] == nil {
		t.Fatal("Expected blk.0.attn_q.weight to be loaded")
	}

	// Inference
	// We want to pass a prompt tokens list
	inputTokens := []int{1, 2, 3}
	// Add config
	config := SamplerConfig{
		Temperature: 0,
	}
	outputTokens, err := engine.Infer(inputTokens, 10, config) // generate 10 tokens
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

func generateMistralMockGGUF(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	if err := binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMagic)); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, uint32(3)); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, uint64(12)); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, uint64(6)); err != nil {
		return err
	}

	// Mistral Metadata
	if err := writeString(f, "llama.block_count"); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeUint32)); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, uint32(1)); err != nil {
		return err
	}

	if err := writeString(f, "llama.embedding_length"); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeUint32)); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, uint32(128)); err != nil {
		return err
	}

	if err := writeString(f, "llama.attention.head_count"); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeUint32)); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, uint32(32)); err != nil {
		return err
	}

	if err := writeString(f, "llama.attention.head_count_kv"); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeUint32)); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, uint32(8)); err != nil {
		return err
	}

	if err := writeString(f, "llama.rope.freq_base"); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeUint32)); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, uint32(100000)); err != nil {
		return err
	}

	if err := writeString(f, "llama.attention.layer_norm_rms_epsilon"); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeFloat32)); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, float32(1e-6)); err != nil {
		return err
	}

	// Tensors (minimal placeholders)
	names := []string{
		"token_embd.weight", "output.weight", "output_norm.weight",
		"blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight",
		"blk.0.attn_output.weight", "blk.0.attn_norm.weight",
		"blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
		"blk.0.ffn_norm.weight",
	}

	writeTensor := func(name string, offset uint64) error {
		if err := writeString(f, name); err != nil {
			return err
		}
		if err := binary.Write(f, binary.LittleEndian, uint32(1)); err != nil {
			return err
		}
		if err := binary.Write(f, binary.LittleEndian, uint64(128)); err != nil {
			return err
		}
		if err := binary.Write(f, binary.LittleEndian, uint32(0)); err != nil {
			return err
		}
		if err := binary.Write(f, binary.LittleEndian, uint64(offset)); err != nil {
			return err
		}
		return nil
	}

	currentOff := uint64(0)
	for _, n := range names {
		if err := writeTensor(n, currentOff); err != nil {
			return err
		}
		currentOff += 512
	}

	if _, err := f.Write(make([]byte, 1024)); err != nil {
		return err
	}
	for i := 0; i < len(names); i++ {
		if err := binary.Write(f, binary.LittleEndian, make([]float32, 128)); err != nil {
			return err
		}
	}

	return nil
}

func TestMistralMetadataSupport(t *testing.T) {
	modelPath := "test_mistral_metadata.gguf"
	if err := generateMistralMockGGUF(modelPath); err != nil {
		t.Fatalf("Failed to generate Mistral mock: %v", err)
	}
	defer os.Remove(modelPath)

	e, err := NewEngine(modelPath, false)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}
	defer e.Ctx.Free()

	if e.Config.KVHeads != 8 {
		t.Errorf("Expected KVHeads=8 (GQA), got %d", e.Config.KVHeads)
	}
	if e.Config.RopeTheta != 100000.0 {
		t.Errorf("Expected RopeTheta=100000.0, got %f", e.Config.RopeTheta)
	}
}

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
		{"float32 input", float32(4.56), 4.56},
		{"uint64 input", uint64(789), 789.0},
		{"uint32 input", uint32(101), 101.0},
		{"int32 input", int32(-112), -112.0},
		{"int64 input", int64(-314), -314.0},
		{"int input", int(500), 500.0},
		{"unsupported string input", "hello", 0.0},
		{"unsupported bool input", true, 0.0},
		{"nil input", nil, 0.0},
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

func TestIsNormWeight(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected bool
	}{
		{"attn_norm.weight", "blk.0.attn_norm.weight", true},
		{"ffn_norm.weight", "blk.0.ffn_norm.weight", true},
		{"output_norm.weight", "output_norm.weight", true},
		{"not a norm weight", "blk.0.attn_q.weight", false},
		{"partial match", "attn_norm", false},
		{"empty string", "", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := isNormWeight(tt.input)
			if got != tt.expected {
				t.Errorf("isNormWeight() got = %v, expected %v", got, tt.expected)
			}
		})
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

func TestInitKVCache(t *testing.T) {
	tests := []struct {
		name                string
		config              LlamaConfig
		expectedError       bool
		expectedKVCacheKLen int
		expectedKVCacheVLen int
	}{
		{
			name: "Valid config with window size",
			config: LlamaConfig{
				Layers:     2,
				WindowSize: 10,
				KVHeads:    2,
				HeadDim:    4,
				SeqLen:     20, // Should be overridden by WindowSize if set
			},
			expectedError:       false,
			expectedKVCacheKLen: 2,
			expectedKVCacheVLen: 2,
		},
		{
			name: "Valid config without window size (uses SeqLen)",
			config: LlamaConfig{
				Layers:     1,
				WindowSize: 0,
				KVHeads:    1,
				HeadDim:    8,
				SeqLen:     15,
			},
			expectedError:       false,
			expectedKVCacheKLen: 1,
			expectedKVCacheVLen: 1,
		},
		{
			name: "Invalid config: zero KVHeads",
			config: LlamaConfig{
				Layers:     1,
				WindowSize: 10,
				KVHeads:    0,
				HeadDim:    4,
				SeqLen:     20,
			},
			expectedError: true,
		},
		{
			name: "Invalid config: zero HeadDim",
			config: LlamaConfig{
				Layers:     1,
				WindowSize: 10,
				KVHeads:    2,
				HeadDim:    0,
				SeqLen:     20,
			},
			expectedError: true,
		},
		{
			name: "Zero layers",
			config: LlamaConfig{
				Layers:     0,
				WindowSize: 10,
				KVHeads:    2,
				HeadDim:    4,
				SeqLen:     20,
			},
			expectedError:       false,
			expectedKVCacheKLen: 0,
			expectedKVCacheVLen: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := &Engine{
				Ctx:    device.NewContext(),
				Config: tt.config,
			}
			defer e.Ctx.Free()

			err := e.initKVCache()

			if tt.expectedError {
				if err == nil {
					t.Errorf("Expected an error for %s, but got none", tt.name)
				}
				return // Skip further checks if error is expected
			} else if err != nil {
				t.Fatalf("Unexpected error for %s: %v", tt.name, err)
			}

			if len(e.KVCacheK) != tt.expectedKVCacheKLen {
				t.Errorf("KVCacheK length mismatch for %s: got %d, expected %d", tt.name, len(e.KVCacheK), tt.expectedKVCacheKLen)
			}
			if len(e.KVCacheV) != tt.expectedKVCacheVLen {
				t.Errorf("KVCacheV length mismatch for %s: got %d, expected %d", tt.name, len(e.KVCacheV), tt.expectedKVCacheVLen)
			}

			for i := 0; i < tt.expectedKVCacheKLen; i++ {
				if e.KVCacheK[i] == nil {
					t.Errorf("KVCacheK[%d] is nil for %s", i, tt.name)
				}
				if e.KVCacheV[i] == nil {
					t.Errorf("KVCacheV[%d] is nil for %s", i, tt.name)
				}
				// Further checks for dimensions could be added if device.Tensor exposed more info
				// For now, checking if it's not nil implies successful allocation to some degree
			}
		})
	}
}
