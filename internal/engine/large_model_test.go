//go:build darwin && metal

package engine

import (
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"testing"

	conf "github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

func ValidateInputTokens(tokens []int, vocabSize int) ([]int, error) {
	if len(tokens) == 0 {
		return nil, errors.New("empty input tokens")
	}
	validated := make([]int, len(tokens))
	for i, token := range tokens {
		if token < 0 || token >= vocabSize {
			return nil, fmt.Errorf("input token %d at position %d is out of vocab range [0, %d)", token, i, vocabSize)
		}
		validated[i] = token
	}
	return validated, nil
}

func generateMockGGUF(path string, config map[string]interface{}) error {
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
	if err := binary.Write(f, binary.LittleEndian, uint64(len(config))); err != nil {
		return err
	}

	for k, v := range config {
		if err := writeString(f, k); err != nil {
			return err
		}
		var valType uint32
		switch v.(type) {
		case float64:
			valType = uint32(gguf.GGUFMetadataValueTypeFloat64)
		case float32:
			valType = uint32(gguf.GGUFMetadataValueTypeFloat32)
		case uint64:
			valType = uint32(gguf.GGUFMetadataValueTypeUint64)
		case uint32:
			valType = uint32(gguf.GGUFMetadataValueTypeUint32)
		}
		if err := binary.Write(f, binary.LittleEndian, valType); err != nil {
			return err
		}
		switch val := v.(type) {
		case float64:
			if err := binary.Write(f, binary.LittleEndian, val); err != nil {
				return err
			}
		case float32:
			if err := binary.Write(f, binary.LittleEndian, val); err != nil {
				return err
			}
		case uint64:
			if err := binary.Write(f, binary.LittleEndian, val); err != nil {
				return err
			}
		case uint32:
			if err := binary.Write(f, binary.LittleEndian, val); err != nil {
				return err
			}
		}
	}

	names := []string{
		"token_embd.weight", "output.weight", "output_norm.weight",
		"blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight",
		"blk.0.attn_output.weight", "blk.0.attn_norm.weight",
		"blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
		"blk.0.ffn_norm.weight",
	}

	writeTensor := func(name string, offset uint64, dim int) error {
		if err := writeString(f, name); err != nil {
			return err
		}
		if err := binary.Write(f, binary.LittleEndian, uint32(2)); err != nil {
			return err
		}
		if err := binary.Write(f, binary.LittleEndian, uint64(dim)); err != nil {
			return err
		}
		if err := binary.Write(f, binary.LittleEndian, uint64(1)); err != nil {
			return err
		}
		if err := binary.Write(f, binary.LittleEndian, uint32(gguf.GGMLTypeF16)); err != nil {
			return err
		}
		if err := binary.Write(f, binary.LittleEndian, uint64(offset)); err != nil {
			return err
		}
		return nil
	}

	currentOff := uint64(0)
	dim := 4096
	for _, n := range names {
		if err := writeTensor(n, currentOff, dim); err != nil {
			return err
		}
		currentOff += uint64(dim * 2)
	}

	if _, err := f.Write(make([]byte, 4096)); err != nil {
		return err
	}
	for i := 0; i < len(names); i++ {
		data := make([]float32, dim)
		for j := range data {
			data[j] = 0.01
		}
		if err := binary.Write(f, binary.LittleEndian, data); err != nil {
			return err
		}
	}

	return nil
}

func TestLargeContext_8K_Integration(t *testing.T) {
	config := map[string]interface{}{
		"llama.block_count":                      float64(32),
		"llama.embedding_length":                 float64(4096),
		"llama.attention.head_count":             float64(32),
		"llama.attention.head_count_kv":          float64(8),
		"llama.feed_forward_length":              float64(14336),
		"llama.context_length":                   float64(8000),
		"llama.attention.layer_norm_rms_epsilon": float64(1e-5),
	}

	modelPath := "test_8k.gguf"
	if err := generateMockGGUF(modelPath, config); err != nil {
		t.Fatalf("Failed to generate test GGUF: %v", err)
	}
	defer os.Remove(modelPath)

	engineConfig := conf.Config{

		KVCacheSize: 22,
	}
	engine, err := NewEngine(modelPath, engineConfig)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}
	defer engine.Ctx.Free()

	if engine.Config.SeqLen != 8000 {
		t.Errorf("Expected SeqLen=8000, got %d", engine.Config.SeqLen)
	}

	t.Logf("8K context model: Dim=%d, Heads=%d, KVHeads=%d, SeqLen=%d",
		engine.Config.Dim, engine.Config.Heads, engine.Config.KVHeads, engine.Config.SeqLen)
}

func TestLargeContext_16K_Integration(t *testing.T) {
	config := map[string]interface{}{
		"llama.block_count":                      float64(32),
		"llama.embedding_length":                 float64(4096),
		"llama.attention.head_count":             float64(32),
		"llama.attention.head_count_kv":          float64(8),
		"llama.feed_forward_length":              float64(14336),
		"llama.context_length":                   float64(16000),
		"llama.attention.layer_norm_rms_epsilon": float64(1e-5),
	}

	modelPath := "test_16k.gguf"
	if err := generateMockGGUF(modelPath, config); err != nil {
		t.Fatalf("Failed to generate test GGUF: %v", err)
	}
	defer os.Remove(modelPath)

	engineConfig := conf.Config{

		KVCacheSize: 22,
	}
	engine, err := NewEngine(modelPath, engineConfig)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}
	defer engine.Ctx.Free()

	if engine.Config.SeqLen != 16000 {
		t.Errorf("Expected SeqLen=16000, got %d", engine.Config.SeqLen)
	}

	t.Logf("16K context model loaded successfully")
}

func TestLargeContext_32K_Integration(t *testing.T) {
	config := map[string]interface{}{
		"llama.block_count":                      float64(32),
		"llama.embedding_length":                 float64(4096),
		"llama.attention.head_count":             float64(32),
		"llama.attention.head_count_kv":          float64(8),
		"llama.feed_forward_length":              float64(14336),
		"llama.context_length":                   float64(32000),
		"llama.attention.layer_norm_rms_epsilon": float64(1e-5),
	}

	modelPath := "test_32k.gguf"
	if err := generateMockGGUF(modelPath, config); err != nil {
		t.Fatalf("Failed to generate test GGUF: %v", err)
	}
	defer os.Remove(modelPath)

	engineConfig := conf.Config{

		KVCacheSize: 22,
	}
	engine, err := NewEngine(modelPath, engineConfig)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}
	defer engine.Ctx.Free()

	if engine.Config.SeqLen != 32000 {
		t.Errorf("Expected SeqLen=32000, got %d", engine.Config.SeqLen)
	}

	t.Logf("32K context model loaded successfully")
}

func TestGQA_Configuration_Integration(t *testing.T) {
	testCases := []struct {
		name     string
		heads    int
		kvHeads  int
		validGQA bool
	}{
		{"Qwen-style GQA", 32, 4, true},
		{"Mistral-style GQA", 32, 8, true},
		{"LLaMA 70B GQA", 64, 8, true},
		{"MHA (no GQA)", 32, 32, true},
		{"Invalid ratio", 32, 5, false},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			config := map[string]interface{}{
				"llama.block_count":                      float64(32),
				"llama.embedding_length":                 float64(4096),
				"llama.attention.head_count":             float64(tc.heads),
				"llama.attention.head_count_kv":          float64(tc.kvHeads),
				"llama.feed_forward_length":              float64(14336),
				"llama.context_length":                   float64(4096),
				"llama.attention.layer_norm_rms_epsilon": float64(1e-5),
			}

			modelPath := "test_gqa.gguf"
			if err := generateMockGGUF(modelPath, config); err != nil {
				t.Fatalf("Failed to generate test GGUF: %v", err)
			}
			defer os.Remove(modelPath)

			engineConfig := conf.Config{

				KVCacheSize: 22,
			}
			engine, err := NewEngine(modelPath, engineConfig)
			if err != nil {
				t.Fatalf("Failed to create engine: %v", err)
			}
			defer engine.Ctx.Free()

			if engine.Config.Heads != tc.heads {
				t.Errorf("Expected Heads=%d, got %d", tc.heads, engine.Config.Heads)
			}
			if engine.Config.KVHeads != tc.kvHeads {
				t.Errorf("Expected KVHeads=%d, got %d", tc.kvHeads, engine.Config.KVHeads)
			}

			if tc.validGQA && engine.Config.Heads%engine.Config.KVHeads != 0 {
				t.Error("GQA ratio should be valid")
			}
		})
	}
}

func TestSliding_Window_Integration(t *testing.T) {
	testCases := []struct {
		name       string
		windowSize int
		ctxLen     int
		validSW    bool
	}{
		{"Mistral v0.1", 4096, 32768, true},
		{"Mistral v0.3", 4096, 32768, true},
		{"Full attention", 0, 4096, true},
		{"No sliding (Llama)", 0, 8192, true},
		{"Invalid: window > ctx", 8192, 4096, false},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			config := map[string]interface{}{
				"llama.block_count":                      float64(32),
				"llama.embedding_length":                 float64(4096),
				"llama.attention.head_count":             float64(32),
				"llama.attention.head_count_kv":          float64(8),
				"llama.feed_forward_length":              float64(14336),
				"llama.context_length":                   float64(tc.ctxLen),
				"llama.attention.sliding_window":         float64(tc.windowSize),
				"llama.attention.layer_norm_rms_epsilon": float64(1e-5),
			}

			modelPath := "test_sw.gguf"
			if err := generateMockGGUF(modelPath, config); err != nil {
				t.Fatalf("Failed to generate test GGUF: %v", err)
			}
			defer os.Remove(modelPath)

			engineConfig := conf.Config{

				KVCacheSize: 22,
			}
			engine, err := NewEngine(modelPath, engineConfig)
			if err != nil {
				if tc.validSW {
					t.Errorf("Expected valid config, got error: %v", err)
				}
				return
			}
			defer engine.Ctx.Free()

			if engine.Config.WindowSize != tc.windowSize {
				t.Errorf("Expected WindowSize=%d, got %d", tc.windowSize, engine.Config.WindowSize)
			}
		})
	}
}

func TestQuantized_Tensor_Validation_Integration(t *testing.T) {
	testCases := []struct {
		name       string
		ggufType   gguf.GGMLType
		cols       int
		shouldLoad bool
	}{
		{"F16 valid", gguf.GGMLTypeF16, 4096, true},
		{"F32 valid", gguf.GGMLTypeF32, 4096, true},
		{"Q4_0 valid (div32)", gguf.GGMLTypeQ4_0, 4096, true},
		{"Q4_0 invalid (not div32)", gguf.GGMLTypeQ4_0, 4097, false},
		{"Q4_K valid (div256)", gguf.GGMLTypeQ4_K, 4096, true},
		{"Q4_K invalid (not div256)", gguf.GGMLTypeQ4_K, 4128, false},
		{"Q6_K valid (div256)", gguf.GGMLTypeQ6_K, 4096, true},
		{"Q6_K invalid (not div256)", gguf.GGMLTypeQ6_K, 4128, false},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateTensorDimensionsForType("test.weight", 1, tc.cols, tc.ggufType)
			if tc.shouldLoad && err != nil {
				t.Errorf("Expected tensor to load, got error: %v", err)
			}
			if !tc.shouldLoad && err == nil {
				t.Errorf("Expected tensor validation to fail")
			}
		})
	}
}

func ValidateTensorDimensionsForType(name string, rows, cols int, ggufType gguf.GGMLType) error {
	switch ggufType {
	case gguf.GGMLTypeF16, gguf.GGMLTypeF32:
		if rows <= 0 || cols <= 0 {
			return &ValidationError{Op: "TensorLoad", Msg: "invalid dimensions", Path: name}
		}
	case gguf.GGMLTypeQ4_0:
		if cols%32 != 0 {
			return &ValidationError{Op: "TensorLoad", Msg: "Q4_0 requires cols % 32 == 0", Path: name}
		}
	case gguf.GGMLTypeQ4_K:
		if cols%256 != 0 {
			return &ValidationError{Op: "TensorLoad", Msg: "Q4_K requires cols % 256 == 0", Path: name}
		}
	case gguf.GGMLTypeQ6_K:
		if cols%256 != 0 {
			return &ValidationError{Op: "TensorLoad", Msg: "Q6_K requires cols % 256 == 0", Path: name}
		}
	}
	return nil
}

type ValidationError struct {
	Op   string
	Msg  string
	Path string
}

func (e *ValidationError) Error() string {
	return e.Op + ": " + e.Msg + " (" + e.Path + ")"
}

func TestModelConfig_Validation_Integration(t *testing.T) {
	testCases := []struct {
		name        string
		config      *device.ModelConfig
		expectValid bool
	}{
		{
			"Valid 7B config",
			&device.ModelConfig{Dim: 4096, Layers: 32, Heads: 32, KVHeads: 8, HeadDim: 128, HiddenDim: 14336},
			true,
		},
		{
			"Valid 70B config",
			&device.ModelConfig{Dim: 8192, Layers: 64, Heads: 64, KVHeads: 8, HeadDim: 128, HiddenDim: 28672},
			true,
		},
		{
			"Valid 3B config",
			&device.ModelConfig{Dim: 3200, Layers: 26, Heads: 32, KVHeads: 8, HeadDim: 100, HiddenDim: 8640},
			true,
		},
		{
			"Invalid: heads not divisible by kv",
			&device.ModelConfig{Dim: 4096, Layers: 32, Heads: 32, KVHeads: 7, HeadDim: 128, HiddenDim: 14336},
			false,
		},
		{
			"Invalid: dim/head mismatch",
			&device.ModelConfig{Dim: 4096, Layers: 32, Heads: 32, KVHeads: 8, HeadDim: 100, HiddenDim: 14336},
			false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.config.Validate()
			if tc.expectValid && err != nil {
				t.Errorf("Expected valid config, got error: %v", err)
			}
			if !tc.expectValid && err == nil {
				t.Errorf("Expected invalid config, got no error")
			}
		})
	}
}

func TestInput_Validation_Integration(t *testing.T) {
	testCases := []struct {
		name        string
		inputTokens []int
		vocabSize   int
		expectError bool
	}{
		{"Valid short input", []int{1, 2, 3, 4, 5}, 49152, false},
		{"Valid long input", make([]int, 1000), 49152, false},
		{"Empty input", []int{}, 49152, true},
		{"Out of range token", []int{1, 50000}, 49152, true},
		{"Negative token", []int{-1, 2}, 49152, true},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ValidateInputTokens(tc.inputTokens, tc.vocabSize)
			if tc.expectError && err == nil {
				t.Error("Expected validation error")
			}
			if !tc.expectError && err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
		})
	}
}
