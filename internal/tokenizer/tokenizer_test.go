

package tokenizer

import (
	"encoding/binary"
	"os"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

func generateVocabGGUF(path string, vocab []string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()

	// Magic
	binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMagic))
	// Version
	binary.Write(f, binary.LittleEndian, uint32(3))
	// Tensor Count (0)
	binary.Write(f, binary.LittleEndian, uint64(0))
	// KV Count (1) - just tokens
	binary.Write(f, binary.LittleEndian, uint64(1))

	// KV Pair: "tokenizer.ggml.tokens"
	writeString(f, "tokenizer.ggml.tokens")
	// Type: Array
	binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeArray))
	// Array Type: String
	binary.Write(f, binary.LittleEndian, uint32(gguf.GGUFMetadataValueTypeString))
	// Array Len
	binary.Write(f, binary.LittleEndian, uint64(len(vocab)))

	// Array Elements
	for _, v := range vocab {
		writeString(f, v)
	}

	return nil
}

func writeString(f *os.File, s string) {
	binary.Write(f, binary.LittleEndian, uint64(len(s)))
	_, _ = f.WriteString(s)
}

func TestTokenizerDecode(t *testing.T) {
	vocab := []string{"<unk>", "Hello", " ", "World", "!"}
	tmpFile := "test_vocab.gguf"
	if err := generateVocabGGUF(tmpFile, vocab); err != nil {
		t.Fatalf("Failed to generate vocab: %v", err)
	}
	defer func() { _ = os.Remove(tmpFile) }()

	tk, err := New(tmpFile)
	if err != nil {
		t.Fatalf("Failed to create tokenizer: %v", err)
	}

	// Test Decode
	// IDs: 1, 2, 3, 4 -> "Hello World!"
	ids := []int{1, 2, 3, 4}
	text := tk.Decode(ids)
	expected := "Hello World!"

	if text != expected {
		t.Errorf("Expected '%s', got '%s'", expected, text)
	}

	// Test GetVocab
	v := tk.GetVocab()
	if len(v) != len(vocab) {
		t.Errorf("Expected %d vocab items, got %d", len(vocab), len(v))
	}
}

func TestTokenizerDecodeSpecialTokens(t *testing.T) {
	vocab := []string{
		"<unk>",
		"<|endoftext|>",
		"ĠHello",
		"Ċ",
		"\u2581World",
		"ĉ",
	}
	tmpFile := "test_vocab_special.gguf"
	if err := generateVocabGGUF(tmpFile, vocab); err != nil {
		t.Fatalf("Failed to generate vocab: %v", err)
	}
	defer func() { _ = os.Remove(tmpFile) }()

	tk, err := New(tmpFile)
	if err != nil {
		t.Fatalf("Failed to create tokenizer: %v", err)
	}

	// Token 1 is special <|endoftext|> (should be skipped)
	// Token 2 has Ġ -> space
	// Token 3 has Ċ -> newline
	// Token 4 has   -> space
	// Token 5 has ĉ -> tab
	// Token -1 and 999 are out of bounds (should be skipped)
	decoded := tk.Decode([]int{-1, 1, 2, 3, 4, 5, 999})
	expected := " Hello\n World\t"
	if decoded != expected {
		t.Errorf("Expected %q, got %q", expected, decoded)
	}
}

func TestTokenizerSplitWords(t *testing.T) {
	if res := splitWords(""); res != nil {
		t.Errorf("Expected nil for empty string, got %v", res)
	}

	words := splitWords("Hello World  from Quarrel")
	if len(words) < 3 {
		t.Errorf("Unexpected splitWords result: %v", words)
	}
}

func TestTokenizerNewFromGGUF_Errors(t *testing.T) {
	// Missing tokens key
	fEmpty := &gguf.GGUFFile{KV: map[string]interface{}{}}
	if _, err := NewFromGGUF(fEmpty); err == nil {
		t.Error("Expected error for missing tokens")
	}

	// Invalid tokens type
	fInvalid := &gguf.GGUFFile{KV: map[string]interface{}{
		"tokenizer.ggml.tokens": 12345,
	}}
	if _, err := NewFromGGUF(fInvalid); err == nil {
		t.Error("Expected error for non-slice tokens")
	}

	// Non-string token element
	fNonString := &gguf.GGUFFile{KV: map[string]interface{}{
		"tokenizer.ggml.tokens": []interface{}{"valid", 123},
	}}
	if _, err := NewFromGGUF(fNonString); err == nil {
		t.Error("Expected error for non-string token element")
	}

	// Merges parsing
	fWithMerges := &gguf.GGUFFile{KV: map[string]interface{}{
		"tokenizer.ggml.tokens": []interface{}{"<unk>", "H", "e", "He"},
		"tokenizer.ggml.merges": []interface{}{"H e"},
	}}
	tk, err := NewFromGGUF(fWithMerges)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if len(tk.Merges) != 1 || tk.Ranks["H e"] != 0 {
		t.Errorf("Merges not loaded properly: %v, %v", tk.Merges, tk.Ranks)
	}
}

func TestTokenizer_DynamicEOS(t *testing.T) {
	// 1. Explicit metadata eos_token_id (uint32)
	f1 := &gguf.GGUFFile{KV: map[string]interface{}{
		"tokenizer.ggml.tokens":       []interface{}{"hello", "world", "<|im_end|>"},
		"tokenizer.ggml.eos_token_id": uint32(2),
	}}
	tk1, err := NewFromGGUF(f1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !tk1.IsEOS(2) {
		t.Errorf("expected token 2 to be EOS")
	}
	if tk1.IsEOS(0) {
		t.Errorf("expected token 0 not to be EOS")
	}

	// 2. Vocab detection of Qwen / LLaMA / Gemma tokens
	f2 := &gguf.GGUFFile{KV: map[string]interface{}{
		"tokenizer.ggml.tokens": []interface{}{
			"hello",          // 0
			"<|im_end|>",     // 1
			"<|endoftext|>",  // 2
			"<end_of_turn>",  // 3
			"</s>",           // 4
			"<|eot_id|>",     // 5
		},
	}}
	tk2, err := NewFromGGUF(f2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for id := 1; id <= 5; id++ {
		if !tk2.IsEOS(id) {
			t.Errorf("expected token %d to be recognized as EOS", id)
		}
	}
	eosIDs := tk2.GetEOSTokenIDs()
	if len(eosIDs) != 5 {
		t.Errorf("expected 5 EOS IDs, got %d", len(eosIDs))
	}

	// 3. Fallback to 2 when no EOS token is present
	f3 := &gguf.GGUFFile{KV: map[string]interface{}{
		"tokenizer.ggml.tokens": []interface{}{"a", "b", "c"},
	}}
	tk3, err := NewFromGGUF(f3)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !tk3.IsEOS(2) {
		t.Errorf("expected fallback EOS token 2")
	}

	// 4. Add custom EOS token
	tk3.AddEOSTokenID(10)
	if !tk3.IsEOS(10) {
		t.Errorf("expected added EOS token 10")
	}

	// 5. Test with nil EOSTokenIDs
	tkNil := &Tokenizer{}
	if !tkNil.IsEOS(2) || tkNil.IsEOS(1) {
		t.Errorf("expected nil EOSTokenIDs fallback to 2")
	}
	tkNil.AddEOSTokenID(99)
	if !tkNil.IsEOS(99) {
		t.Errorf("expected token 99 after AddEOSTokenID on nil map")
	}

	// 6. Test various numeric types for eos_token_id
	types := []interface{}{
		int32(10), int64(11), float64(12), int(13), uint64(14),
	}
	for _, numVal := range types {
		fNum := &gguf.GGUFFile{KV: map[string]interface{}{
			"tokenizer.ggml.tokens":       []interface{}{"token"},
			"tokenizer.ggml.eos_token_id": numVal,
		}}
		tkNum, err := NewFromGGUF(fNum)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(tkNum.GetEOSTokenIDs()) == 0 {
			t.Errorf("expected non-empty EOS tokens for %T", numVal)
		}
	}
}
