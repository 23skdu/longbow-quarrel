package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/engine"
)

// templateTokenizer is a TokenizerShim that also exposes a chat template,
// like the real tokenizer does when the model ships one.
//
// Encode returns one token per byte so the rendered prompt can be recovered
// from the token stream the engine receives.
type templateTokenizer struct {
	mockTokenizer
	template string
}

func (t *templateTokenizer) ChatTemplate() string { return t.template }

func (t *templateTokenizer) Encode(s string) []int {
	tokens := make([]int, len(s))
	for i := range s {
		tokens[i] = int(s[i])
	}
	return tokens
}

// GetVocab reports the 256 byte values Encode emits: the API validates token
// ids against the tokenizer's vocabulary, and a byte-level tokenizer whose
// vocab list held only a handful of entries would reject its own output.
func (t *templateTokenizer) GetVocab() []string {
	vocab := make([]string, 256)
	for i := range vocab {
		vocab[i] = string(rune(i))
	}
	return vocab
}

// capturingEngine records the prompt it was given.
type capturingEngine struct {
	mockEngine
	prompt string
}

func (c *capturingEngine) Infer(tokens []int, count int, cfg engine.SamplerConfig) ([]int, error) {
	var sb strings.Builder
	for _, t := range tokens {
		sb.WriteByte(byte(t))
	}
	c.prompt = sb.String()
	return []int{1}, nil
}

// TestChatCompletions_UsesModelChatTemplate checks that the
// OpenAI-compatible endpoint renders the model's own chat template.
//
// The handler used to hardcode the ChatML format (<|im_start|>...) for
// every model. That is the Qwen format, so a Llama 3, Gemma or Mistral
// model prompted that way receives a prompt it was never trained on.
func TestChatCompletions_UsesModelChatTemplate(t *testing.T) {
	tmpl := `{% for message in messages %}{{ '<|' + message['role'] + '|>' + message['content'] }}{% endfor %}`

	tok := &templateTokenizer{template: tmpl}
	eng := &capturingEngine{}
	s := &Server{Engine: eng, Tokenizer: tok}

	body, _ := json.Marshal(ChatCompletionRequest{
		Messages: []ChatMessage{
			{Role: "user", Content: "hello there"},
		},
		MaxTokens: 4,
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	rr := httptest.NewRecorder()
	s.ChatCompletionsHandler(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}

	if !strings.Contains(eng.prompt, "<|user|>hello there") {
		t.Errorf("the model chat template was not applied; prompt was %q", eng.prompt)
	}
	if strings.Contains(eng.prompt, "<|im_start|>") {
		t.Errorf("the hardcoded ChatML format was used instead of the model template: %q", eng.prompt)
	}
}

// TestChatCompletions_FallsBackWithoutTemplate checks that a model without a
// chat template still gets a usable prompt.
func TestChatCompletions_FallsBackWithoutTemplate(t *testing.T) {
	tok := &templateTokenizer{}
	eng := &capturingEngine{}
	s := &Server{Engine: eng, Tokenizer: tok}

	body, _ := json.Marshal(ChatCompletionRequest{
		Messages: []ChatMessage{
			{Role: "user", Content: "hello there"},
		},
		MaxTokens: 4,
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	rr := httptest.NewRecorder()
	s.ChatCompletionsHandler(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}

	if !strings.Contains(eng.prompt, "hello there") {
		t.Errorf("the fallback prompt lost the user message: %q", eng.prompt)
	}
}
