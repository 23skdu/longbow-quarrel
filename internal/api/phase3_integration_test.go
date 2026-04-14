package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/engine"
)

func TestPhase3_Integration(t *testing.T) {
	cfg := config.Config{VocabSize: 100}
	eng, _ := engine.NewMockEngine("mock", cfg)
	tok := &mockTokenizer{}
	
	server := &Server{
		Engine:    eng,
		Tokenizer: tok,
		UsedMemory: func() int64 { return 0 },
	}

	t.Run("LoadAdapter_API", func(t *testing.T) {
		reqBody, _ := json.Marshal(LoadAdapterRequest{
			Path: "test.lora",
			ID:   "my-lora",
		})
		req := httptest.NewRequest("POST", "/v1/adapters/load", bytes.NewBuffer(reqBody))
		rr := httptest.NewRecorder()

		server.LoadAdapterHandler(rr, req)

		if rr.Code != http.StatusCreated {
			t.Errorf("expected 201 Created, got %d", rr.Code)
		}
	})

	t.Run("Speculative_Request_API", func(t *testing.T) {
		reqBody, _ := json.Marshal(CompletionRequest{
			Prompt:      "hello",
			MaxTokens:   10,
			Speculative: true,
		})
		req := httptest.NewRequest("POST", "/v1/completions", bytes.NewBuffer(reqBody))
		rr := httptest.NewRecorder()

		server.CompletionsHandler(rr, req)

		if rr.Code != http.StatusOK {
			t.Errorf("expected 200 OK, got %d", rr.Code)
		}
	})
}

