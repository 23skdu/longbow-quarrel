package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
)

// idOnlyTokenizer encodes every prompt to a single chosen id and reports a
// chosen vocabulary, so a test can place that id inside or outside the
// vocabulary.
type idOnlyTokenizer struct {
	id    int
	vocab []string
}

func (t *idOnlyTokenizer) Encode(string) []int { return []int{t.id} }
func (t *idOnlyTokenizer) Decode([]int) string { return "decoded" }
func (t *idOnlyTokenizer) GetVocab() []string  { return t.vocab }

// TestCompletionsRejectsOutOfRangeTokenIDs checks that a prompt that encodes
// to a token id the engine's vocabulary does not have is a 400 at the API
// boundary. Inside the engine such an id is either clamped to token 0, which
// turns into silent nonsense, or read past the end of the embedding table.
//
// This happens for real: a model hot-swap leaves the old tokenizer serving
// ids the new model has no rows for.
func TestCompletionsRejectsOutOfRangeTokenIDs(t *testing.T) {
	const id = 10

	for _, tc := range []struct {
		name      string
		vocabSize int // of the engine's config
		vocabLen  int // of the tokenizer
		wantCode  int
	}{
		{"id inside the model's vocabulary", 32, 2, http.StatusOK},
		{"id outside the model's vocabulary", 5, 2, http.StatusBadRequest},
		{"no model vocabulary, tokenizer has the id", 0, 12, http.StatusOK},
		{"no model vocabulary, tokenizer lacks the id", 0, 2, http.StatusBadRequest},
	} {
		t.Run(tc.name, func(t *testing.T) {
			vocab := make([]string, tc.vocabLen)
			for i := range vocab {
				vocab[i] = "tok"
			}

			srv := &Server{
				Engine:    &mockEngine{cfg: config.Config{VocabSize: tc.vocabSize}},
				Tokenizer: &idOnlyTokenizer{id: id, vocab: vocab},
				MaxMemory: 1024 * 1024 * 1024,
			}

			raw, err := json.Marshal(CompletionRequest{
				Model:     "test",
				Prompt:    "hello",
				MaxTokens: 4,
			})
			if err != nil {
				t.Fatalf("marshal failed: %v", err)
			}

			rec := httptest.NewRecorder()
			srv.CompletionsHandler(rec, httptest.NewRequest(http.MethodPost, "/v1/completions", bytes.NewReader(raw)))

			if rec.Code != tc.wantCode {
				t.Fatalf("got %d, want %d: %s", rec.Code, tc.wantCode, rec.Body.String())
			}
			if tc.wantCode == http.StatusBadRequest && !strings.Contains(rec.Body.String(), "out of vocab range") {
				t.Errorf("the 400 does not explain the rejected token id: %s", rec.Body.String())
			}
		})
	}
}

// TestChatCompletionsRejectsOutOfRangeTokenIDs checks the chat endpoint has
// the same boundary check as the completions endpoint.
func TestChatCompletionsRejectsOutOfRangeTokenIDs(t *testing.T) {
	srv := &Server{
		Engine:    &mockEngine{cfg: config.Config{VocabSize: 5}},
		Tokenizer: &idOnlyTokenizer{id: 10, vocab: []string{"a", "b"}},
		MaxMemory: 1024 * 1024 * 1024,
	}

	raw, err := json.Marshal(ChatCompletionRequest{
		Model: "test",
		Messages: []ChatMessage{
			{Role: "user", Content: "hello"},
		},
		MaxTokens: 4,
	})
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}

	rec := httptest.NewRecorder()
	srv.ChatCompletionsHandler(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(raw)))

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("got %d, want 400: %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "out of vocab range") {
		t.Errorf("the 400 does not explain the rejected token id: %s", rec.Body.String())
	}
}
