//go:build webui

package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
	"unicode"
)

func FuzzGenerateRequestParsing(f *testing.F) {
	testCases := []string{
		`{"prompt": "hello"}`,
		`{"prompt": "hello", "temperature": 0.5}`,
		`{"prompt": "test", "max_tokens": 100}`,
		`{"prompt": "", "temperature": 0.7}`,
		`{"prompt": "test", "temperature": -1.0}`,
		`{"prompt": "test", "temperature": 3.0}`,
		`{"max_tokens": 0}`,
		`{"model": "test-model"}`,
		`{"topk": 40, "topp": 0.95}`,
		`{}`,
		`{"prompt": "test", "extra_field": "ignored"}`,
		`{"prompt": "こんにちは"}`,
		`{"prompt": "🎉 emoji test"}`,
		`{"prompt": "test\nwith\nnewlines"}`,
		`{"prompt": "test\twith\ttabs"}`,
	}

	for _, tc := range testCases {
		f.Add(tc)
	}

	f.Fuzz(func(t *testing.T, data string) {
		var req GenerateRequest
		err := json.NewDecoder(bytes.NewReader([]byte(data))).Decode(&req)
		if err == nil {
			_ = req.Prompt
			_ = req.Temperature
			_ = req.MaxTokens
		}
	})
}

func FuzzHealthStatusJSON(f *testing.F) {
	f.Fuzz(func(t *testing.T, status, version, uptime string) {
		hs := HealthStatus{
			Status:    status,
			Version:   version,
			Uptime:    uptime,
			Timestamp: time.Now(),
			Checks:    map[string]Status{},
		}

		data, err := json.Marshal(hs)
		if err != nil {
			t.Fatal(err)
		}

		var decoded HealthStatus
		if err := json.Unmarshal(data, &decoded); err != nil {
			t.Fatal(err)
		}
	})
}

func FuzzVersionInfoJSON(f *testing.F) {
	f.Fuzz(func(t *testing.T, version, commit, goVersion string) {
		vi := VersionInfo{
			Version:   version,
			Commit:    commit,
			GoVersion: goVersion,
		}

		data, err := json.Marshal(vi)
		if err != nil {
			t.Fatal(err)
		}

		var decoded VersionInfo
		if err := json.Unmarshal(data, &decoded); err != nil {
			t.Fatal(err)
		}
	})
}

func FuzzInferenceResponseJSON(f *testing.F) {
	f.Fuzz(func(t *testing.T, token string, tokenID int, complete bool, tps float64) {
		resp := InferenceResponse{
			Token:        token,
			TokenID:      tokenID,
			Complete:     complete,
			TokensPerSec: tps,
		}

		data, err := json.Marshal(resp)
		if err != nil {
			t.Fatal(err)
		}

		var decoded InferenceResponse
		if err := json.Unmarshal(data, &decoded); err != nil {
			t.Fatal(err)
		}
	})
}

func FuzzAuthMiddlewareValidation(f *testing.F) {
	validKeys := []string{
		"qk_test123",
		"qk_abc123",
		"qk_",
	}

	invalidKeys := []string{
		"invalid",
		"wrong",
		"",
		"qk_",
	}

	for _, key := range append(validKeys, invalidKeys...) {
		f.Add(key)
	}

	f.Fuzz(func(t *testing.T, apiKey string) {
		middleware := NewAuthMiddleware("qk_test123")
		handlerCalled := false

		handler := middleware.Authenticate(func(w http.ResponseWriter, r *http.Request) {
			handlerCalled = true
			w.WriteHeader(http.StatusOK)
		})

		req := httptest.NewRequest(http.MethodGet, "/test", nil)
		if apiKey != "" {
			req.Header.Set("Authorization", "ApiKey "+apiKey)
		}

		w := httptest.NewRecorder()
		handler(w, req)

		_ = handlerCalled
	})
}

func FuzzPromptValidation(f *testing.F) {
	f.Fuzz(func(t *testing.T, prompt string) {
		isValid := prompt != ""
		if isValid && len(prompt) > 10000 {
			isValid = false
		}

		for _, r := range prompt {
			if unicode.IsControl(r) && r != '\n' && r != '\t' {
				isValid = false
				break
			}
		}

		_ = isValid
	})
}

func FuzzTemperatureValidation(f *testing.F) {
	f.Fuzz(func(t *testing.T, temp float64) {
		isValid := temp >= 0 && temp <= 2.0

		_ = isValid
	})
}

func FuzzMaxTokensValidation(f *testing.F) {
	f.Fuzz(func(t *testing.T, maxTokens int) {
		isValid := maxTokens > 0 && maxTokens <= 8192

		_ = isValid
	})
}

func FuzzTopKValidation(f *testing.F) {
	f.Fuzz(func(t *testing.T, topK int) {
		isValid := topK > 0 && topK <= 100

		_ = isValid
	})
}

func FuzzTopPValidation(f *testing.F) {
	f.Fuzz(func(t *testing.T, topP float64) {
		isValid := topP > 0 && topP <= 1.0

		_ = isValid
	})
}

func FuzzGenerateResponseJSON(f *testing.F) {
	f.Fuzz(func(t *testing.T, text string, tokens int, tps float64) {
		resp := GenerateResponse{
			Text:            text,
			TokensGenerated: tokens,
			TokensPerSec:    tps,
		}

		data, err := json.Marshal(resp)
		if err != nil {
			t.Fatal(err)
		}

		var decoded GenerateResponse
		if err := json.Unmarshal(data, &decoded); err != nil {
			t.Fatal(err)
		}

		if decoded.Text != text {
			t.Errorf("expected text %q, got %q", text, decoded.Text)
		}
		if decoded.TokensGenerated != tokens {
			t.Errorf("expected tokens %d, got %d", tokens, decoded.TokensGenerated)
		}
	})
}
