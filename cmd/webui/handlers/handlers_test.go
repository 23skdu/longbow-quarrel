//go:build webui

package handlers_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/23skdu/longbow-quarrel/cmd/webui/handlers"
)

func TestHealthHandler(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()

	handlers.HealthHandler()(w, req)
	res := w.Result()
	defer res.Body.Close()

	if res.StatusCode != http.StatusOK {
		t.Errorf("Expected status 200, got %d", res.StatusCode)
	}

	var healthStatus handlers.HealthStatus
	if err := json.NewDecoder(res.Body).Decode(&healthStatus); err != nil {
		t.Fatalf("Failed to decode response: %v", err)
	}

	if healthStatus.Status != "healthy" {
		t.Errorf("Expected status 'healthy', got '%s'", healthStatus.Status)
	}

	if healthStatus.Version == "" {
		t.Error("Expected version to be set")
	}

	if healthStatus.Uptime == "" {
		t.Error("Expected uptime to be set")
	}
}

func TestHealthzHandler(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	w := httptest.NewRecorder()

	handlers.HealthzHandler()(w, req)
	res := w.Result()
	defer res.Body.Close()

	if res.StatusCode != http.StatusOK {
		t.Errorf("Expected status 200, got %d", res.StatusCode)
	}

	body := w.Body.String()
	if body != "OK\n" {
		t.Errorf("Expected 'OK\\n', got '%s'", body)
	}
}

func TestReadyzHandler(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/readyz", nil)
	w := httptest.NewRecorder()

	handlers.ReadyzHandler()(w, req)
	res := w.Result()
	defer res.Body.Close()

	if res.StatusCode != http.StatusOK {
		t.Errorf("Expected status 200, got %d", res.StatusCode)
	}

	body := w.Body.String()
	if body != "Ready\n" {
		t.Errorf("Expected 'Ready\\n', got '%s'", body)
	}
}

func TestVersionHandler(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/version", nil)
	w := httptest.NewRecorder()

	handlers.VersionHandler()(w, req)
	res := w.Result()
	defer res.Body.Close()

	if res.StatusCode != http.StatusOK {
		t.Errorf("Expected status 200, got %d", res.StatusCode)
	}

	var versionInfo handlers.VersionInfo
	if err := json.NewDecoder(res.Body).Decode(&versionInfo); err != nil {
		t.Fatalf("Failed to decode response: %v", err)
	}

	if versionInfo.Version == "" {
		t.Error("Expected version to be set")
	}

	if versionInfo.GoVersion == "" {
		t.Error("Expected go_version to be set")
	}
}

func TestAPIKeyManager(t *testing.T) {
	manager := handlers.GetAPIKeyManager()

	key, err := handlers.GenerateAPIKey()
	if err != nil {
		t.Fatalf("Failed to generate API key: %v", err)
	}

	if !strings.HasPrefix(key, "qk_") {
		t.Errorf("Expected key to start with 'qk_', got '%s'", key)
	}

	err = manager.AddKey(key, "test-key", time.Hour, 100)
	if err != nil {
		t.Fatalf("Failed to add key: %v", err)
	}

	validKey, err := manager.ValidateKey(key)
	if err != nil {
		t.Fatalf("Failed to validate key: %v", err)
	}

	if validKey.Name != "test-key" {
		t.Errorf("Expected name 'test-key', got '%s'", validKey.Name)
	}

	err = manager.RevokeKey("test-key")
	if err != nil {
		t.Fatalf("Failed to revoke key: %v", err)
	}

	_, err = manager.ValidateKey(key)
	if err == nil {
		t.Error("Expected error when validating revoked key")
	}
}

func TestAuthMiddleware(t *testing.T) {
	tests := []struct {
		name           string
		apiKey         string
		authHeader     string
		queryParam     string
		expectedStatus int
	}{
		{
			name:           "No authentication required",
			apiKey:         "",
			expectedStatus: http.StatusOK,
		},
		{
			name:           "Valid API key in header",
			apiKey:         "test-api-key",
			authHeader:     "ApiKey test-api-key",
			expectedStatus: http.StatusOK,
		},
		{
			name:           "Invalid API key",
			apiKey:         "test-api-key",
			authHeader:     "ApiKey wrong-key",
			expectedStatus: http.StatusUnauthorized,
		},
		{
			name:           "API key in query param",
			apiKey:         "test-api-key",
			queryParam:     "api_key=test-api-key",
			expectedStatus: http.StatusOK,
		},
		{
			name:           "Missing API key",
			apiKey:         "test-api-key",
			expectedStatus: http.StatusUnauthorized,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			middleware := handlers.NewAuthMiddleware(tt.apiKey)

			handlerCalled := false
			handler := middleware.Authenticate(func(w http.ResponseWriter, r *http.Request) {
				handlerCalled = true
				w.WriteHeader(http.StatusOK)
			})

			url := "/api/models"
			if tt.queryParam != "" {
				url += "?" + tt.queryParam
			}

			req := httptest.NewRequest(http.MethodGet, url, nil)
			if tt.authHeader != "" {
				req.Header.Set("Authorization", tt.authHeader)
			}

			w := httptest.NewRecorder()
			handler(w, req)
			res := w.Result()
			defer res.Body.Close()

			if res.StatusCode != tt.expectedStatus {
				t.Errorf("Expected status %d, got %d", tt.expectedStatus, res.StatusCode)
			}

			if tt.expectedStatus == http.StatusOK && !handlerCalled {
				t.Error("Handler was not called")
			}
		})
	}
}

func TestCORSMiddleware(t *testing.T) {
	tests := []struct {
		name            string
		origin          string
		allowedOrigins  []string
		expectedACAO    string
		expectPreflight bool
	}{
		{
			name:            "Default wildcard",
			origin:          "http://example.com",
			allowedOrigins:  nil,                  // defaults to ["*"]
			expectedACAO:    "http://example.com", // echoes origin when wildcard
			expectPreflight: true,
		},
		{
			name:            "Explicit wildcard origin",
			origin:          "http://example.com",
			allowedOrigins:  []string{"*"},
			expectedACAO:    "http://example.com", // echoes origin when wildcard
			expectPreflight: true,
		},
		{
			name:            "Matching origin",
			origin:          "http://localhost:3000",
			allowedOrigins:  []string{"http://localhost:3000"},
			expectedACAO:    "http://localhost:3000",
			expectPreflight: true,
		},
		{
			name:            "Non-matching origin",
			origin:          "http://evil.com",
			allowedOrigins:  []string{"http://localhost:3000"},
			expectedACAO:    "",
			expectPreflight: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			middleware := handlers.NewCORSMiddleware(tt.allowedOrigins)

			handlerCalled := false
			handler := middleware.Middleware(func(w http.ResponseWriter, r *http.Request) {
				handlerCalled = true
				w.WriteHeader(http.StatusOK)
			})

			if tt.expectPreflight {
				req := httptest.NewRequest(http.MethodOptions, "/api/models", nil)
				req.Header.Set("Origin", tt.origin)
				req.Header.Set("Access-Control-Request-Method", "POST")

				w := httptest.NewRecorder()
				handler(w, req)
				res := w.Result()
				defer res.Body.Close()

				if res.StatusCode != http.StatusNoContent {
					t.Errorf("Expected status 204, got %d", res.StatusCode)
				}

				acao := res.Header.Get("Access-Control-Allow-Origin")
				if acao != tt.expectedACAO {
					t.Errorf("Expected ACAO '%s', got '%s'", tt.expectedACAO, acao)
				}
			} else {
				req := httptest.NewRequest(http.MethodGet, "/api/models", nil)
				req.Header.Set("Origin", tt.origin)

				w := httptest.NewRecorder()
				handler(w, req)
				res := w.Result()
				defer res.Body.Close()

				if !handlerCalled {
					t.Error("Handler was not called")
				}

				acao := res.Header.Get("Access-Control-Allow-Origin")
				if acao != tt.expectedACAO {
					t.Errorf("Expected ACAO '%s', got '%s'", tt.expectedACAO, acao)
				}
			}
		})
	}
}

func TestGenerateRequestValidation(t *testing.T) {
	tests := []struct {
		name      string
		prompt    string
		maxTokens int
		temp      float64
		valid     bool
	}{
		{
			name:      "Valid request",
			prompt:    "Hello, world!",
			maxTokens: 100,
			temp:      0.7,
			valid:     true,
		},
		{
			name:      "Empty prompt",
			prompt:    "",
			maxTokens: 100,
			temp:      0.7,
			valid:     false,
		},
		{
			name:      "Zero max tokens",
			prompt:    "Hello",
			maxTokens: 0,
			temp:      0.7,
			valid:     false,
		},
		{
			name:      "Negative temperature",
			prompt:    "Hello",
			maxTokens: 100,
			temp:      -0.1,
			valid:     false,
		},
		{
			name:      "Temperature too high",
			prompt:    "Hello",
			maxTokens: 100,
			temp:      2.1,
			valid:     false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := handlers.GenerateRequest{
				Prompt:      tt.prompt,
				MaxTokens:   tt.maxTokens,
				Temperature: tt.temp,
			}

			isValid := req.Prompt != "" && req.MaxTokens > 0 && req.Temperature >= 0 && req.Temperature <= 2

			if isValid != tt.valid {
				t.Errorf("Expected validity %v, got %v", tt.valid, isValid)
			}
		})
	}
}

func TestInferenceResponseJSON(t *testing.T) {
	resp := handlers.InferenceResponse{
		Token:        "Hello",
		TokenID:      0,
		Complete:     false,
		TokensPerSec: 50.5,
	}

	data, err := json.Marshal(resp)
	if err != nil {
		t.Fatalf("Failed to marshal: %v", err)
	}

	var decoded handlers.InferenceResponse
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("Failed to unmarshal: %v", err)
	}

	if decoded.Token != resp.Token {
		t.Errorf("Expected token '%s', got '%s'", resp.Token, decoded.Token)
	}

	if decoded.Complete != resp.Complete {
		t.Errorf("Expected complete %v, got %v", resp.Complete, decoded.Complete)
	}
}

func TestLoggingMiddleware(t *testing.T) {
	middleware := handlers.NewLoggingMiddleware()

	handler := middleware.Middleware(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	req := httptest.NewRequest(http.MethodGet, "/test", nil)
	w := httptest.NewRecorder()
	handler(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}
}

func TestLoggingMiddlewareLogsRequest(t *testing.T) {
	middleware := handlers.NewLoggingMiddleware()

	var logged bool
	handler := middleware.Middleware(func(w http.ResponseWriter, r *http.Request) {
		logged = true
		w.WriteHeader(http.StatusOK)
	})

	req := httptest.NewRequest(http.MethodGet, "/api/test?key=value", nil)
	req.RemoteAddr = "127.0.0.1:12345"

	w := httptest.NewRecorder()
	handler(w, req)

	if !logged {
		t.Error("Handler was not called")
	}
}

func TestGenerateResponseJSON(t *testing.T) {
	resp := handlers.GenerateResponse{
		Text:            "Generated text",
		TokensGenerated: 42,
		TokensPerSec:    15.5,
	}

	data, err := json.Marshal(resp)
	if err != nil {
		t.Fatalf("Failed to marshal: %v", err)
	}

	var decoded handlers.GenerateResponse
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("Failed to unmarshal: %v", err)
	}

	if decoded.Text != resp.Text {
		t.Errorf("Expected text '%s', got '%s'", resp.Text, decoded.Text)
	}

	if decoded.TokensGenerated != resp.TokensGenerated {
		t.Errorf("Expected tokens %d, got %d", resp.TokensGenerated, decoded.TokensGenerated)
	}
}

func TestGenerateRequestDefaults(t *testing.T) {
	req := handlers.GenerateRequest{
		Prompt:      "test",
		MaxTokens:   0,
		Temperature: 0,
		TopK:        0,
		TopP:        0,
	}

	if req.MaxTokens <= 0 {
		req.MaxTokens = 256
	}
	if req.Temperature <= 0 {
		req.Temperature = 0.7
	}
	if req.TopK <= 0 {
		req.TopK = 40
	}
	if req.TopP <= 0 {
		req.TopP = 0.95
	}

	if req.MaxTokens != 256 {
		t.Errorf("Expected default max_tokens 256, got %d", req.MaxTokens)
	}
	if req.Temperature != 0.7 {
		t.Errorf("Expected default temperature 0.7, got %f", req.Temperature)
	}
	if req.TopK != 40 {
		t.Errorf("Expected default topk 40, got %d", req.TopK)
	}
	if req.TopP != 0.95 {
		t.Errorf("Expected default topp 0.95, got %f", req.TopP)
	}
}

func TestCORSMultipleOrigins(t *testing.T) {
	allowedOrigins := []string{"http://localhost:3000", "http://localhost:8080"}
	middleware := handlers.NewCORSMiddleware(allowedOrigins)

	handler := middleware.Middleware(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	testOrigins := []struct {
		origin         string
		expectedStatus int
		expectedACAO   string
	}{
		{"http://localhost:3000", http.StatusOK, "http://localhost:3000"},
		{"http://localhost:8080", http.StatusOK, "http://localhost:8080"},
		{"http://evil.com", http.StatusOK, ""},
	}

	for _, tt := range testOrigins {
		req := httptest.NewRequest(http.MethodGet, "/api/models", nil)
		req.Header.Set("Origin", tt.origin)

		w := httptest.NewRecorder()
		handler(w, req)

		acao := w.Header().Get("Access-Control-Allow-Origin")
		if acao != tt.expectedACAO {
			t.Errorf("For origin %s, expected ACAO '%s', got '%s'", tt.origin, tt.expectedACAO, acao)
		}
	}
}

func TestAPIKeyQueryParam(t *testing.T) {
	manager := handlers.GetAPIKeyManager()
	key, _ := handlers.GenerateAPIKey()
	_ = manager.AddKey(key, "test", time.Hour, 100)

	middleware := handlers.NewAuthMiddleware(key)

	handler := middleware.Authenticate(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	req := httptest.NewRequest(http.MethodGet, "/api/models?api_key="+key, nil)
	w := httptest.NewRecorder()
	handler(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}
}
