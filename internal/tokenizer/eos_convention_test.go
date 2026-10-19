package tokenizer

import (
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

// TestTokenIDTwoIsAlwaysEOS pins the convention that token ID 2 is always
// an end-of-sequence token.
//
// Every engine in this repo treated `token == 2` as EOS unconditionally, in
// addition to whatever the tokenizer reported from the model metadata. The
// tokenizer now agrees, which lets the decode loops consult a single source
// of truth instead of duplicating the check.
//
// It matters for models whose metadata names a different EOS token and whose
// vocabulary does not use ID 2 for anything: before this change the
// tokenizer and the engine disagreed about ID 2.
func TestTokenIDTwoIsAlwaysEOS(t *testing.T) {
	// The vocabulary has an EOS token at ID 1, and ID 2 is an ordinary
	// token.
	f := &gguf.GGUFFile{KV: map[string]interface{}{
		"tokenizer.ggml.tokens": []interface{}{
			"hello",      // 0
			"<|im_end|>", // 1
			"ordinary",   // 2
			"another",    // 3
			"<|eot_id|>", // 4
		},
	}}

	tk, err := NewFromGGUF(f)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !tk.IsEOS(2) {
		t.Error("expected token 2 to be EOS; every engine has always " +
			"treated it as EOS and the tokenizer must agree")
	}
	if !tk.IsEOS(1) {
		t.Error("expected the metadata EOS token 1 to be EOS")
	}
	if tk.IsEOS(0) {
		t.Error("expected token 0 not to be EOS")
	}

	ids := tk.GetEOSTokenIDs()
	found := false
	for _, id := range ids {
		if id == 2 {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("GetEOSTokenIDs should include token 2, got %v", ids)
	}
}
