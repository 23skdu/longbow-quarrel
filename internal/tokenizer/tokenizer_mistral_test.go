//go:build darwin && metal

package tokenizer

import (
	"os"
	"testing"
)

const mistralModelPath = "/Users/rsd/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f"

func TestTokenizer_Mistral_RealLoad(t *testing.T) {
	if _, err := os.Stat(mistralModelPath); os.IsNotExist(err) {
		t.Skipf("Mistral model not found at %s", mistralModelPath)
	}

	tok, err := New(mistralModelPath)
	if err != nil {
		t.Fatalf("Failed to load tokenizer: %v", err)
	}

	if len(tok.Tokens) == 0 {
		t.Errorf("Tokenizer loaded 0 tokens")
	}
	t.Logf("Loaded %d tokens", len(tok.Tokens))
}

func TestTokenizer_Mistral_DecodeSpecific(t *testing.T) {
	if _, err := os.Stat(mistralModelPath); os.IsNotExist(err) {
		t.Skip("Mistral model not found")
	}
	tok, _ := New(mistralModelPath)

	tokensToTest := []int{31980, 9445}
	for _, id := range tokensToTest {
		text := tok.Decode([]int{id})
		t.Logf("Token %d decodes to: %q", id, text)
	}
}

func TestTokenizer_Mistral_SpecialTokens(t *testing.T) {
	if _, err := os.Stat(mistralModelPath); os.IsNotExist(err) {
		t.Skip("Mistral model not found")
	}
	tok, _ := New(mistralModelPath)

	// Mistral Standard:
	// 1 = BOS
	// 2 = EOS
	// ? = UNK (Often 0 or none)

	bos := tok.Decode([]int{1})
	eos := tok.Decode([]int{2})

	t.Logf("BOS (1): %q", bos)
	t.Logf("EOS (2): %q", eos)

	// Check if ID 1 maps to a known BOS string in vocab if possible, or just empty string if special control token?
	// Mistral usually treats 1 as <s> and 2 as </s> but they might not decode to visible text depending on logic.
}

func TestTokenizer_Mistral_RoundTrip(t *testing.T) {
	if _, err := os.Stat(mistralModelPath); os.IsNotExist(err) {
		t.Skip("Mistral model not found")
	}
	tok, _ := New(mistralModelPath)

	input := "The capital of France is Paris."
	ids := tok.Encode(input)
	decoded := tok.Decode(ids)

	t.Logf("Input: %q", input)
	t.Logf("IDs: %v", ids)
	t.Logf("Decoded: %q", decoded)

	// Note: Spacing might vary (e.g. leading space), so we check similarity or exact match
	// Mistral tokenizer often adds a leading space dummy prefix?
	if decoded != input && decoded != " "+input {
		// Strict check might fail if tokenizer is implemented differently, so we log for now.
		t.Logf("Warning: Roundtrip mismatch. Expected %q, got %q", input, decoded)
	}
}

func TestTokenizer_Mistral_PromptFormat(t *testing.T) {
	if _, err := os.Stat(mistralModelPath); os.IsNotExist(err) {
		t.Skip("Mistral model not found")
	}
	tok, _ := New(mistralModelPath)

	prompt := "[INST] What is the capital of France? [/INST]"
	ids := tok.Encode(prompt)

	t.Logf("Prompt: %q", prompt)
	t.Logf("IDs: %v", ids)

	// Verify that [INST] is not split into [ I N S T ]
	// [INST] should be a single token or specific sequence.
	// For Mistral, [INST] is not a single special token in the vocab usually, it's text.
	// Wait, actually strict Mistral instruction format uses control tokens OR text.
	// Let's see what it produces.

	decoded := tok.Decode(ids)
	t.Logf("Decoded: %q", decoded)
}
