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
	defer f.Close()

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
	f.WriteString(s)
}

func TestTokenizerDecode(t *testing.T) {
	vocab := []string{"<unk>", "Hello", " ", "World", "!"}
	tmpFile := "test_vocab.gguf"
	if err := generateVocabGGUF(tmpFile, vocab); err != nil {
		t.Fatalf("Failed to generate vocab: %v", err)
	}
	defer os.Remove(tmpFile)

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
}
