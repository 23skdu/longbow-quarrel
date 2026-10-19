package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/engine"
)

// specMockEngine records which engine served a request and remembers the
// draft length it was given.
type specMockEngine struct {
	mockEngine
	calls      int
	draftLen   int
	usedDraftK bool
}

func (m *specMockEngine) Infer(tokens []int, count int, cfg engine.SamplerConfig) ([]int, error) {
	m.calls++
	return []int{7, 8, 9}, nil
}

func (m *specMockEngine) SetDraftLength(k int) {
	m.draftLen = k
	m.usedDraftK = true
}

// TestCompletionsSpeculativeSelection checks that the "speculative" request
// field actually selects the speculative engine.
//
// The field used to be accepted and then ignored: the handler always used
// the plain engine, so a request that asked for speculative decoding
// silently got ordinary generation.
func TestCompletionsSpeculativeSelection(t *testing.T) {
	plain := &specMockEngine{}
	spec := &specMockEngine{}

	mux := InitServerWithSpeculative(1024, func() int64 { return 0 }, plain, spec, &mockTokenizer{})

	post := func(body string) {
		t.Helper()
		req := httptest.NewRequest(http.MethodPost, "/v1/completions", bytes.NewBufferString(body))
		rec := httptest.NewRecorder()
		mux.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("got the status %d, want 200: %s", rec.Code, rec.Body.String())
		}
	}

	// A plain request uses the plain engine.
	post(`{"prompt": "hello", "max_tokens": 3}`)
	if plain.calls != 1 {
		t.Errorf("the plain engine served %d requests, want 1", plain.calls)
	}
	if spec.calls != 0 {
		t.Errorf("the speculative engine served %d requests, want 0", spec.calls)
	}

	// A request that asks for speculative decoding uses the speculative
	// engine.
	post(`{"prompt": "hello", "max_tokens": 3, "speculative": true}`)
	if spec.calls != 1 {
		t.Errorf("the speculative engine served %d requests, want 1", spec.calls)
	}
	if plain.calls != 1 {
		t.Errorf("the plain engine served %d requests, want 1", plain.calls)
	}

	// The per-request draft length reaches the engine.
	post(`{"prompt": "hello", "max_tokens": 3, "speculative": true, "draft_k": 6}`)
	if !spec.usedDraftK {
		t.Error("the draft length was not passed to the engine")
	}
	if spec.draftLen != 6 {
		t.Errorf("the draft length is %d, want 6", spec.draftLen)
	}
}

// TestChatCompletionsSpeculativeSelection checks the same for the chat
// endpoint.
func TestChatCompletionsSpeculativeSelection(t *testing.T) {
	plain := &specMockEngine{}
	spec := &specMockEngine{}

	mux := InitServerWithSpeculative(1024, func() int64 { return 0 }, plain, spec, &mockTokenizer{})

	body, err := json.Marshal(map[string]any{
		"messages":    []map[string]string{{"role": "user", "content": "hello"}},
		"max_tokens":  3,
		"speculative": true,
		"draft_k":     5,
	})
	if err != nil {
		t.Fatalf("failed to build the request: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBuffer(body))
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("got the status %d, want 200: %s", rec.Code, rec.Body.String())
	}

	if spec.calls != 1 {
		t.Errorf("the speculative engine served %d requests, want 1", spec.calls)
	}
	if plain.calls != 0 {
		t.Errorf("the plain engine served %d requests, want 0", plain.calls)
	}
	if !spec.usedDraftK || spec.draftLen != 5 {
		t.Errorf("the draft length is %d (set: %v), want 5", spec.draftLen, spec.usedDraftK)
	}
}
