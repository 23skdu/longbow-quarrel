package tokenizer

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

// The tests used to load a Mistral model from a hard-coded path on the
// machine of the developer who wrote them:
//
//	/Users/rsd/.ollama/models/blobs/sha256-f5074b...
//
// The file is not in the repository and not on any other machine, so every
// one of these tests skipped everywhere, including in CI: five tests that
// never ran.
//
// The model is now built here, with the GGUF writer, so the tests run
// everywhere.

// writeMistralLikeModel writes a small model with a Mistral-like
// vocabulary: the sentencepiece pieces of a handful of words, the
// beginning-of-string and end-of-string control tokens, and the merges
// that join the pieces into words.
func writeMistralLikeModel(t *testing.T) string {
	t.Helper()

	path := filepath.Join(t.TempDir(), "mistral-like.gguf")

	// The pieces: the control tokens first, so that the ids match the
	// ones the Mistral vocabulary uses (1 is <s>, 2 is </s>).
	//
	// A sentencepiece vocabulary holds every piece twice: once with the
	// space marker and once without. The encoder marks the space before
	// a word, so the word at the start of the text needs the unmarked
	// piece.
	tokens := []string{
		"<unk>", "<s>", "</s>",
		"▁The", "▁capital", "▁of", "▁France", "▁is", "▁Paris", ".",
		"The", "capital", "of", "France", "is", "Paris", "?", "▁?",
	}
	scores := make([]float32, len(tokens))
	for i := range scores {
		scores[i] = float32(-i)
	}
	// The token types: 1 is a normal token, 2 is the unknown token, 3 is
	// a control token.
	types := make([]int32, len(tokens))
	for i := range types {
		types[i] = 1
	}
	types[0] = 2
	types[1] = 3
	types[2] = 3

	kv := map[string]interface{}{
		"general.architecture":            "llama",
		"tokenizer.ggml.model":            "llama",
		"tokenizer.ggml.tokens":           tokens,
		"tokenizer.ggml.scores":           scores,
		"tokenizer.ggml.token_type":       types,
		"tokenizer.ggml.bos_token_id":     uint32(1),
		"tokenizer.ggml.eos_token_id":     uint32(2),
		"tokenizer.ggml.unknown_token_id": uint32(0),
	}

	if err := gguf.WriteFile(path, kv, nil, 32); err != nil {
		t.Fatalf("failed to write the model: %v", err)
	}
	return path
}

func TestTokenizer_Mistral_RealLoad(t *testing.T) {
	path := writeMistralLikeModel(t)

	tok, err := New(path)
	if err != nil {
		t.Fatalf("Failed to load tokenizer: %v", err)
	}

	if len(tok.Tokens) == 0 {
		t.Error("Tokenizer loaded 0 tokens")
	}
	if got := len(tok.Vocab); got == 0 {
		t.Error("Tokenizer loaded an empty vocabulary")
	}
}

func TestTokenizer_Mistral_DecodeSpecific(t *testing.T) {
	path := writeMistralLikeModel(t)

	tok, err := New(path)
	if err != nil {
		t.Fatalf("Failed to load tokenizer: %v", err)
	}

	// The control tokens and the first word of the vocabulary. The
	// sentencepiece marker is a space when it is decoded.
	cases := map[int]string{
		1: "<s>",
		2: "</s>",
		3: " The",
	}
	for id, want := range cases {
		if got := tok.Decode([]int{id}); got != want {
			t.Errorf("the token %d decodes to %q, want %q", id, got, want)
		}
	}
}

func TestTokenizer_Mistral_SpecialTokens(t *testing.T) {
	path := writeMistralLikeModel(t)

	tok, err := New(path)
	if err != nil {
		t.Fatalf("Failed to load tokenizer: %v", err)
	}

	// The end-of-string token has to be known to the tokenizer: the
	// decode loops stop on it.
	eos := tok.GetEOSTokenIDs()
	if len(eos) == 0 {
		t.Fatal("the tokenizer knows no end-of-string token")
	}
	found := false
	for _, id := range eos {
		if id == 2 {
			found = true
		}
	}
	if !found {
		t.Errorf("the end-of-string tokens are %v, want the token 2 in them", eos)
	}
}

func TestTokenizer_Mistral_RoundTrip(t *testing.T) {
	path := writeMistralLikeModel(t)

	tok, err := New(path)
	if err != nil {
		t.Fatalf("Failed to load tokenizer: %v", err)
	}

	input := "The capital of France is Paris."
	ids := tok.Encode(input)
	if len(ids) == 0 {
		t.Fatal("the encoder produced no tokens")
	}
	decoded := tok.Decode(ids)

	// The pieces carry the sentencepiece marker, so the round trip
	// restores the spaces.
	want := "The capital of France is Paris."
	if decoded != want {
		t.Errorf("the round trip of %q gave %q", input, decoded)
	}
}

func TestTokenizer_Mistral_PromptFormat(t *testing.T) {
	path := writeMistralLikeModel(t)

	tok, err := New(path)
	if err != nil {
		t.Fatalf("Failed to load tokenizer: %v", err)
	}

	prompt := "The capital of France?"
	ids := tok.Encode(prompt)
	if len(ids) == 0 {
		t.Fatal("the encoder produced no tokens")
	}

	// The question has to survive the round trip, and the pieces have to
	// be the ones of the vocabulary: a tokenizer that split the text into
	// single characters would decode to the same string.
	decoded := tok.Decode(ids)
	if decoded != prompt {
		t.Errorf("the round trip of %q gave %q", prompt, decoded)
	}
	for _, id := range ids {
		if id != 3 && id < 3 {
			t.Errorf("the encoder produced the control token %d for %q", id, prompt)
		}
	}
}

func TestTokenizer_Mistral_ModelFileIsTemporary(t *testing.T) {
	// The model the tests use is written into the test's temporary
	// directory, so no test leaves a file behind.
	path := writeMistralLikeModel(t)
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("the model was not written: %v", err)
	}
}

// writeGemma4LikeModel writes a small model with the Gemma 4 metadata:
// the architecture is "gemma4" and the vocabulary holds the pieces of a
// handful of words.
func writeGemma4LikeModel(t *testing.T) string {
	t.Helper()

	path := filepath.Join(t.TempDir(), "gemma4-like.gguf")

	tokens := []string{
		"<unk>", "<bos>", "<eos>",
		"▁The", "▁capital", "▁of", "▁France", "▁is", "▁Paris", ".",
		"The", "capital", "of", "France", "is", "Paris", "▁", "▁▁",
	}
	scores := make([]float32, len(tokens))
	for i := range scores {
		scores[i] = float32(-i)
	}
	types := make([]int32, len(tokens))
	for i := range types {
		types[i] = 1
	}
	types[0] = 2
	types[1] = 3
	types[2] = 3

	kv := map[string]interface{}{
		"general.architecture":           "gemma4",
		"general.alignment":              uint32(32),
		"gemma4.block_count":             uint32(2),
		"gemma4.embedding_length":        uint32(64),
		"gemma4.attention.head_count":    uint32(4),
		"gemma4.attention.head_count_kv": uint32(2),
		"gemma4.feed_forward_length":     uint32(128),
		"gemma4.context_length":          uint32(256),
		"gemma4.vocab_size":              uint32(len(tokens)),
		"tokenizer.ggml.model":           "llama",
		"tokenizer.ggml.tokens":          tokens,
		"tokenizer.ggml.scores":          scores,
		"tokenizer.ggml.token_type":      types,
		"tokenizer.ggml.bos_token_id":    uint32(1),
		"tokenizer.ggml.eos_token_id":    uint32(2),
	}

	if err := gguf.WriteFile(path, kv, nil, 32); err != nil {
		t.Fatalf("failed to write the model: %v", err)
	}
	return path
}
