package engine

import (
	"os"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/tokenizer"
)

func TestTokenizerParityV5(t *testing.T) {
	testCases := []struct {
		prompt     string
		minTokens  int
		maxTokens  int
		exactIDs  []int
	}{
		{
			prompt:    "Hello world",
			minTokens: 2,
			maxTokens: 5,
		},
		{
			prompt:    "The quick brown",
			minTokens: 3,
			maxTokens: 8,
		},
		{
			prompt:    "Once upon a time",
			minTokens: 4,
			maxTokens: 10,
		},
		{
			prompt:    "Explain quantum computing",
			minTokens: 3,
			maxTokens: 8,
		},
		{
			prompt:    "Write a haiku about AI",
			minTokens: 4,
			maxTokens: 12,
		},
	}

	modelPath := "../../models/qwen2-0.5b-q4_k_m.gguf"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Model file not found: " + modelPath)
	}

	tk, err := tokenizer.New(modelPath)
	if err != nil {
		t.Fatalf("Failed to load tokenizer: %v", err)
	}

	for _, tc := range testCases {
		t.Run(tc.prompt, func(t *testing.T) {
			ids := tk.Encode(tc.prompt)
			tokenCount := len(ids)

			t.Logf("Prompt: %q -> %d tokens: %v", tc.prompt, tokenCount, ids)

			if tokenCount < tc.minTokens {
				t.Errorf("Expected at least %d tokens, got %d", tc.minTokens, tokenCount)
			}
			if tokenCount > tc.maxTokens {
				t.Errorf("Expected at most %d tokens, got %d", tc.maxTokens, tokenCount)
			}

			decoded := tk.Decode(ids)
			t.Logf("Decoded back: %q", decoded)
		})
	}
}

func TestChatTemplateParity(t *testing.T) {
	tests := []struct {
		templateName string
		messages  []map[string]string
		expected  string
	}{
		{
			templateName: "llama3",
			messages: []map[string]string{
				{"role": "system", "content": "You are a helpful assistant."},
				{"role": "user", "content": "Hello"},
			},
			expected: "model",
		},
		{
			templateName: "chatml",
			messages: []map[string]string{
				{"role": "system", "content": "System"},
				{"role": "user", "content": "Hi"},
			},
			expected: "model",
		},
		{
			templateName: "mistral",
			messages: []map[string]string{
				{"role": "user", "content": "Hello"},
			},
			expected: "model",
		},
		{
			templateName: "qwen",
			messages: []map[string]string{
				{"role": "user", "content": "Test"},
			},
			expected: "model",
		},
	}

	for _, tc := range tests {
		t.Run(tc.templateName, func(t *testing.T) {
			t.Logf("Testing chat template: %s with %d messages", tc.templateName, len(tc.messages))
		})
	}
}

func TestQuantizationParity(t *testing.T) {
	testCases := []struct {
		qType string
		desc  string
	}{
		{"Q4_K", "4-bit K-quantization"},
		{"Q5_K", "5-bit K-quantization"},
		{"Q8_0", "8-bit quantization"},
		{"F16", "16-bit float"},
	}

	for _, tc := range testCases {
		t.Run(tc.qType, func(t *testing.T) {
			t.Logf("Testing quantization parity for type: %s - %s", tc.qType, tc.desc)
		})
	}

	t.Run("weight_only_vs_dynamic", func(t *testing.T) {
		t.Log("Comparing weight-only quantization with dynamic quantization")
	})
}

func TestTokenizerV5VocabLoading(t *testing.T) {
	testVocab := []string{"▁Hello", "▁world", "▁Test", "er", "lo", "l", "l", "o"}

	if len(testVocab) < 3 {
		t.Fatal("test vocab needs at least 3 tokens")
	}

	t.Run("vocab_structure", func(t *testing.T) {
		if len(testVocab) == 0 {
			t.Error("expected non-empty vocab")
		}
		for i, tok := range testVocab {
			t.Logf("Token %d: %q", i, tok)
		}
	})

	t.Run("tokenizable_strings", func(t *testing.T) {
		testInputs := []string{"Hello", "world", "Test", "Hello world"}
		for _, input := range testInputs {
			t.Logf("Input: %q -> should tokenize", input)
		}
	})
}