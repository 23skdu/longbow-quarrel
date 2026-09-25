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
	_ = json.NewDecoder(rr.Body).Decode(&resp)
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
	_ = json.NewDecoder(rr.Body).Decode(&resp)
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

func TestServer_Readyz(t *testing.T) {
	// Not ready when engine is nil
	sUnready := &Server{}
	req := httptest.NewRequest("GET", "/readyz", nil)
	rr := httptest.NewRecorder()
	sUnready.ReadyzEndpoint(rr, req)
	if rr.Code != http.StatusServiceUnavailable {
		t.Errorf("expected 503 for unready server, got %v", rr.Code)
	}

	// Ready when engine is present
	sReady := &Server{Engine: &mockEngine{}}
	rrReady := httptest.NewRecorder()
	sReady.ReadyzEndpoint(rrReady, req)
	if rrReady.Code != http.StatusOK {
		t.Errorf("expected 200 for ready server, got %v", rrReady.Code)
	}
}

func TestServer_ListAdapters(t *testing.T) {
	s := &Server{}

	// Test invalid method
	postReq := httptest.NewRequest("POST", "/v1/adapters/list", nil)
	rrPost := httptest.NewRecorder()
	s.ListAdaptersHandler(rrPost, postReq)
	if rrPost.Code != http.StatusMethodNotAllowed {
		t.Errorf("expected 405 for POST /v1/adapters/list, got %v", rrPost.Code)
	}

	// Test valid GET
	getReq := httptest.NewRequest("GET", "/v1/adapters/list", nil)
	rrGet := httptest.NewRecorder()
	s.ListAdaptersHandler(rrGet, getReq)
	if rrGet.Code != http.StatusOK {
		t.Errorf("expected 200 for GET /v1/adapters/list, got %v", rrGet.Code)
	}

	var resp ListAdaptersResponse
	if err := json.NewDecoder(rrGet.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode adapters response: %v", err)
	}
}
