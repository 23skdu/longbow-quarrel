package api

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestAPIIntegration_FailureModes(t *testing.T) {
	s := &Server{
		Engine:     &mockEngine{},
		Tokenizer:  &mockTokenizer{},
		MaxMemory:  1024,
		UsedMemory: func() int64 { return 0 },
	}

	// 1. Method Not Allowed
	req := httptest.NewRequest("GET", "/v1/completions", nil)
	rr := httptest.NewRecorder()
	s.CompletionsHandler(rr, req)
	if rr.Code != http.StatusMethodNotAllowed {
		t.Errorf("expected 405, got %d", rr.Code)
	}

	// 2. Malformed JSON
	req = httptest.NewRequest("POST", "/v1/completions", bytes.NewBuffer([]byte("{invalid}")))
	rr = httptest.NewRecorder()
	s.CompletionsHandler(rr, req)
	if rr.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", rr.Code)
	}
}

func TestInitServer_Coverage(t *testing.T) {
	// hit the global initialization path
	InitServer(1024, func() int64 { return 0 }, &mockEngine{}, &mockTokenizer{})
	if globalServer == nil {
		t.Fatal("globalServer not initialized")
	}
}
