package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)


func TestServer_Healthz(t *testing.T) {
	s := &Server{
		MaxMemory:  1000,
		UsedMemory: func() int64 { return 500 },
	}

	req := httptest.NewRequest("GET", "/healthz", nil)
	rr := httptest.NewRecorder()

	s.HealthzEndpoint(rr, req)

	if status := rr.Code; status != http.StatusOK {
		t.Errorf("handler returned wrong status code: got %v want %v", status, http.StatusOK)
	}

	var resp HealthResponse
	json.NewDecoder(rr.Body).Decode(&resp)
	if resp.LoadPercent != 50 {
		t.Errorf("expected 50%% load, got %d%%", resp.LoadPercent)
	}
}

func TestServer_Completions(t *testing.T) {
	s := &Server{
		Engine:    &mockEngine{},
		Tokenizer: &mockTokenizer{},
	}

	body, _ := json.Marshal(CompletionRequest{
		Prompt:    "hello",
		MaxTokens: 5,
	})

	req := httptest.NewRequest("POST", "/v1/completions", bytes.NewBuffer(body))
	rr := httptest.NewRecorder()

	s.CompletionsHandler(rr, req)

	if status := rr.Code; status != http.StatusOK {
		t.Errorf("handler returned wrong status code: got %v want %v", status, http.StatusOK)
	}

	var resp CompletionResponse
	json.NewDecoder(rr.Body).Decode(&resp)
	if len(resp.Choices) == 0 || resp.Choices[0].Text != "test response" {
		t.Errorf("unexpected response text: %v", resp.Choices)
	}
}

func TestServer_Degraded(t *testing.T) {
	s := &Server{
		MaxMemory:  1000,
		UsedMemory: func() int64 { return 960 }, // 96%
	}

	req := httptest.NewRequest("GET", "/healthz", nil)
	rr := httptest.NewRecorder()

	s.HealthzEndpoint(rr, req)

	if status := rr.Code; status != http.StatusServiceUnavailable {
		t.Errorf("expected 503 for degraded server, got %v", status)
	}
}
