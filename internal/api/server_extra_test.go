package api

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/config"
)

func TestServer_Completions_ErrorBranches(t *testing.T) {
	s := &Server{
		Engine:    &mockEngine{},
		Tokenizer: &mockTokenizer{},
	}

	// 1. OOM condition
	sOOM := &Server{
		MaxMemory:  1000,
		UsedMemory: func() int64 { return 960 },
		Engine:     &mockEngine{},
		Tokenizer:  &mockTokenizer{},
	}
	body, _ := json.Marshal(CompletionRequest{Prompt: "hi"})
	req := httptest.NewRequest("POST", "/v1/completions", bytes.NewBuffer(body))
	rr := httptest.NewRecorder()
	sOOM.CompletionsHandler(rr, req)
	if rr.Code != http.StatusServiceUnavailable {
		t.Errorf("expected 503 for OOM, got %d", rr.Code)
	}

	// 2. Method not allowed
	req = httptest.NewRequest("GET", "/v1/completions", nil)
	rr = httptest.NewRecorder()
	s.CompletionsHandler(rr, req)
	if rr.Code != http.StatusMethodNotAllowed {
		t.Errorf("expected 405 for GET, got %d", rr.Code)
	}

	// 3. Bad JSON
	req = httptest.NewRequest("POST", "/v1/completions", bytes.NewBufferString("{bad"))
	rr = httptest.NewRecorder()
	s.CompletionsHandler(rr, req)
	if rr.Code != http.StatusBadRequest {
		t.Errorf("expected 400 for bad JSON, got %d", rr.Code)
	}

	// 4. Empty prompt
	body, _ = json.Marshal(CompletionRequest{Prompt: ""})
	req = httptest.NewRequest("POST", "/v1/completions", bytes.NewBuffer(body))
	rr = httptest.NewRecorder()
	s.CompletionsHandler(rr, req)
	if rr.Code != http.StatusBadRequest {
		t.Errorf("expected 400 for empty prompt, got %d", rr.Code)
	}

	// 5. Uninitialized components
	sUninit := &Server{}
	body, _ = json.Marshal(CompletionRequest{Prompt: "hi"})
	req = httptest.NewRequest("POST", "/v1/completions", bytes.NewBuffer(body))
	rr = httptest.NewRecorder()
	sUninit.CompletionsHandler(rr, req)
	if rr.Code != http.StatusServiceUnavailable {
		t.Errorf("expected 503 for uninit, got %d", rr.Code)
	}

	// 6. With Grammar
	body, _ = json.Marshal(CompletionRequest{Prompt: "hi", Grammar: "root ::= [a-z]+"})
	req = httptest.NewRequest("POST", "/v1/completions", bytes.NewBuffer(body))
	rr = httptest.NewRecorder()
	s.CompletionsHandler(rr, req)
	if rr.Code != http.StatusOK {
		t.Errorf("expected 200 with grammar, got %d", rr.Code)
	}
}

func TestServer_ChatCompletions_ExtraBranches(t *testing.T) {
	s := &Server{
		Engine:    &mockEngine{cfg: config.Config{Architecture: "gemma4"}},
		Tokenizer: &mockTokenizer{},
	}

	// 1. Method not allowed
	req := httptest.NewRequest("GET", "/v1/chat/completions", nil)
	rr := httptest.NewRecorder()
	s.ChatCompletionsHandler(rr, req)
	if rr.Code != http.StatusMethodNotAllowed {
		t.Errorf("expected 405, got %d", rr.Code)
	}

	// 2. Bad JSON
	req = httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString("{bad"))
	rr = httptest.NewRecorder()
	s.ChatCompletionsHandler(rr, req)
	if rr.Code != http.StatusBadRequest {
		t.Errorf("expected 400 for bad JSON, got %d", rr.Code)
	}

	// 3. Empty messages
	body, _ := json.Marshal(ChatCompletionRequest{Messages: nil})
	req = httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBuffer(body))
	rr = httptest.NewRecorder()
	s.ChatCompletionsHandler(rr, req)
	if rr.Code != http.StatusBadRequest {
		t.Errorf("expected 400 for empty messages, got %d", rr.Code)
	}

	// 4. Uninitialized
	sUninit := &Server{}
	body, _ = json.Marshal(ChatCompletionRequest{Messages: []ChatMessage{{Role: "user", Content: "hi"}}})
	req = httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBuffer(body))
	rr = httptest.NewRecorder()
	sUninit.ChatCompletionsHandler(rr, req)
	if rr.Code != http.StatusServiceUnavailable {
		t.Errorf("expected 503 for uninit, got %d", rr.Code)
	}

	// 5. Multimodal with text and base64 image
	imgBytes := []byte{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A}
	b64Img := "data:image/png;base64," + base64.StdEncoding.EncodeToString(imgBytes)

	multiParts := []interface{}{
		ChatContentPart{Type: "text", Text: "describe image"},
		ChatContentPart{Type: "image_url", ImageURL: &ChatImageURL{URL: b64Img}},
	}
	body, _ = json.Marshal(ChatCompletionRequest{
		Messages: []ChatMessage{
			{Role: "user", Content: multiParts},
		},
		Grammar: "root ::= [0-9]+",
	})
	req = httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBuffer(body))
	rr = httptest.NewRecorder()
	s.ChatCompletionsHandler(rr, req)
	if rr.Code != http.StatusOK {
		t.Errorf("expected 200 for multimodal chat, got %d", rr.Code)
	}
}

func TestServer_ResourceMonitor(t *testing.T) {
	s := &Server{
		MaxMemory:  1024,
		UsedMemory: func() int64 { return 512 },
	}
	if s.getUsedMemory() != 512 {
		t.Errorf("expected 512 used memory, got %d", s.getUsedMemory())
	}
	var nilServer *Server
	if nilServer.getUsedMemory() != 0 {
		t.Errorf("expected 0 for nilServer, got %d", nilServer.getUsedMemory())
	}
}
