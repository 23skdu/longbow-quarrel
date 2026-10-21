package tokenizer

import (
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

// TestGemma4Tokens checks the tokenizer against a model with the Gemma 4
// metadata.
//
// The test used to load a model from a hard-coded path on the machine of
// the developer who wrote it:
//
//	/home/rsd/.cache/llmfit/models/Gemma4_E2B_Abliterated_Opus_Distilled.Q8_0.gguf
//
// The file is in no repository and on no other machine, so the test
// skipped everywhere, including in CI. The model is built here now.
func TestGemma4Tokens(t *testing.T) {
	path := writeGemma4LikeModel(t)

	f, err := gguf.LoadFile(path)
	if err != nil {
		t.Fatalf("failed to load the model: %v", err)
	}
	defer func() { _ = f.Close() }()

	tok, err := NewFromGGUF(f)
	if err != nil {
		t.Fatalf("failed to create tokenizer: %v", err)
	}

	if len(tok.Tokens) == 0 {
		t.Fatal("the tokenizer holds no tokens")
	}

	// The vocabulary has to encode the text of the model.
	for _, text := range []string{"The capital of France is", " Paris.", "Paris", " Paris"} {
		ids := tok.Encode(text)
		if len(ids) == 0 {
			t.Errorf("the encoder produced no tokens for %q", text)
			continue
		}
		if decoded := tok.Decode(ids); decoded == "" {
			t.Errorf("the decoder produced nothing for %q", text)
		}
	}
}
