package tokenizer

import (
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

func TestTokenizer_Robustness(t *testing.T) {
	t.Run("EmptyInput", func(t *testing.T) {
		tk := &Tokenizer{} // Uninitialized
		res := tk.Encode("")
		if len(res) != 0 {
			t.Errorf("expected empty tokens for empty string")
		}
	})

	t.Run("DiversePatterns", func(t *testing.T) {
		tk := &Tokenizer{Vocab: map[string]int{"hello": 1, "world": 2}}
		// Empty Ranks triggers Greedy Max Match (useMaxMatch = true)
		res := tk.Encode("  hello   world!!!  ")
		if len(res) == 0 {
			t.Errorf("greedy match should find tokens")
		}
	})

	t.Run("InvalidModelPath", func(t *testing.T) {
		_, err := New("/non/existent/path")
		if err == nil {
			t.Errorf("expected error for invalid path")
		}
	})

	t.Run("NewFromGGUF_Errors", func(t *testing.T) {
		// Test missing key error path
		f := &gguf.GGUFFile{KV: make(map[string]interface{})}
		_, err := NewFromGGUF(f)
		if err == nil {
			t.Errorf("expected error for empty GGUF")
		}
	})
}
