package api

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestChatCompletionsHandler_Multimodal(t *testing.T) {
	mockE := &mockEngine{}
	mockT := &mockTokenizer{}

	srv := &Server{
		Engine:    mockE,
		Tokenizer: mockT,
		MaxMemory: 1024 * 1024 * 1024,
	}

	// Tiny valid 1x1 PNG in base64
	tinyPNG := "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

	reqBody := ChatCompletionRequest{
		Model: "gemma-4-vlm",
		Messages: []ChatMessage{
			{
				Role: "user",
				Content: []interface{}{
					map[string]interface{}{
						"type": "text",
						"text": "What is in this image?",
					},
					map[string]interface{}{
						"type": "image_url",
						"image_url": map[string]interface{}{
							"url": "data:image/png;base64," + tinyPNG,
						},
					},
				},
			},
		},
		MaxTokens:   10,
		Temperature: 0.7,
	}

	raw, err := json.Marshal(reqBody)
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(raw))
	rec := httptest.NewRecorder()

	srv.ChatCompletionsHandler(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d: %s", rec.Code, rec.Body.String())
	}

	var resp ChatCompletionResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if len(resp.Choices) == 0 {
		t.Fatalf("expected choices, got 0")
	}
	if resp.Choices[0].Message.Role != "assistant" {
		t.Errorf("expected assistant role, got %s", resp.Choices[0].Message.Role)
	}

	_ = base64.StdEncoding
}
