package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestChatCompletionsHandler_Stream(t *testing.T) {
	s := &Server{
		Engine:    &mockEngine{},
		Tokenizer: &mockTokenizer{},
	}

	reqBody := ChatCompletionRequest{
		Model: "test-model",
		Messages: []ChatMessage{
			{Role: "user", Content: "Hello"},
		},
		Stream:    true,
		MaxTokens: 10,
	}
	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		t.Fatalf("failed to marshal request: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(bodyBytes))
	rec := httptest.NewRecorder()

	s.ChatCompletionsHandler(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}

	contentType := rec.Header().Get("Content-Type")
	if !strings.HasPrefix(contentType, "text/event-stream") {
		t.Errorf("expected Content-Type text/event-stream, got %q", contentType)
	}

	bodyStr := rec.Body.String()
	if !strings.Contains(bodyStr, "data: [DONE]") {
		t.Errorf("expected SSE body to end with data: [DONE], got:\n%s", bodyStr)
	}

	// Parse SSE chunks
	lines := strings.Split(bodyStr, "\n")
	chunkCount := 0
	foundRole := false
	foundContent := false
	foundStop := false

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data: ") || line == "data: [DONE]" {
			continue
		}
		rawJSON := strings.TrimPrefix(line, "data: ")
		var chunk ChatCompletionChunk
		if err := json.Unmarshal([]byte(rawJSON), &chunk); err != nil {
			t.Errorf("failed to unmarshal chunk %q: %v", rawJSON, err)
			continue
		}
		chunkCount++
		if chunk.Object != "chat.completion.chunk" {
			t.Errorf("expected object chat.completion.chunk, got %q", chunk.Object)
		}
		if len(chunk.Choices) > 0 {
			choice := chunk.Choices[0]
			if choice.Delta.Role == "assistant" {
				foundRole = true
			}
			if choice.Delta.Content != "" {
				foundContent = true
			}
			if choice.FinishReason != nil && *choice.FinishReason == "stop" {
				foundStop = true
			}
		}
	}

	if chunkCount < 3 {
		t.Errorf("expected at least 3 chunks, got %d", chunkCount)
	}
	if !foundRole {
		t.Errorf("did not find role in any chunk")
	}
	if !foundContent {
		t.Errorf("did not find content in any chunk")
	}
	if !foundStop {
		t.Errorf("did not find finish_reason 'stop' in any chunk")
	}
}
