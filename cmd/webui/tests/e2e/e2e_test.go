//go:build webui

package e2e

import (
	"bufio"
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/23skdu/longbow-quarrel/cmd/webui/handlers"
)

func TestFullAPIHealthCheck(t *testing.T) {
	tests := []struct {
		name         string
		endpoint     string
		method       string
		body         string
		expectedCode int
		checkBody    func([]byte) bool
	}{
		{
			name:         "Health endpoint returns OK",
			endpoint:     "/health",
			method:       http.MethodGet,
			expectedCode: http.StatusOK,
			checkBody: func(body []byte) bool {
				var status handlers.HealthStatus
				return json.Unmarshal(body, &status) == nil && status.Status == "healthy"
			},
		},
		{
			name:         "Liveness probe",
			endpoint:     "/healthz",
			method:       http.MethodGet,
			expectedCode: http.StatusOK,
			checkBody: func(body []byte) bool {
				return string(body) == "OK\n"
			},
		},
		{
			name:         "Readiness probe",
			endpoint:     "/readyz",
			method:       http.MethodGet,
			expectedCode: http.StatusOK,
			checkBody: func(body []byte) bool {
				return string(body) == "Ready\n"
			},
		},
		{
			name:         "Version endpoint",
			endpoint:     "/version",
			method:       http.MethodGet,
			expectedCode: http.StatusOK,
			checkBody: func(body []byte) bool {
				var version handlers.VersionInfo
				return json.Unmarshal(body, &version) == nil && version.Version != ""
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(tt.method, tt.endpoint, strings.NewReader(tt.body))
			w := httptest.NewRecorder()

			switch tt.endpoint {
			case "/health":
				handlers.HealthHandler()(w, req)
			case "/healthz":
				handlers.HealthzHandler()(w, req)
			case "/readyz":
				handlers.ReadyzHandler()(w, req)
			case "/version":
				handlers.VersionHandler()(w, req)
			}

			res := w.Result()
			defer res.Body.Close()

			if res.StatusCode != tt.expectedCode {
				t.Errorf("Expected status %d, got %d", tt.expectedCode, res.StatusCode)
			}

			if !tt.checkBody(w.Body.Bytes()) {
				t.Errorf("Response body check failed: %s", w.Body.String())
			}
		})
	}
}

func TestAPIEndpointsIntegration(t *testing.T) {
	t.Run("Models endpoint returns array", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/api/models", nil)
		w := httptest.NewRecorder()

		handlers.ModelsHandler(handlers.Config{})(w, req)
		res := w.Result()
		defer res.Body.Close()

		if res.StatusCode != http.StatusOK {
			t.Errorf("Expected status 200, got %d", res.StatusCode)
		}

		var models []handlers.ModelInfo
		if err := json.NewDecoder(res.Body).Decode(&models); err != nil {
			t.Fatalf("Failed to decode models: %v", err)
		}
	})

	t.Run("Generate endpoint accepts JSON", func(t *testing.T) {
		body := `{"prompt": "Hello", "max_tokens": 10}`
		req := httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()

		handlers.GenerateHandler(handlers.Config{})(w, req)
		res := w.Result()
		defer res.Body.Close()

		if res.StatusCode != http.StatusOK {
			t.Errorf("Expected status 200, got %d", res.StatusCode)
		}

		var response handlers.GenerateResponse
		if err := json.NewDecoder(res.Body).Decode(&response); err != nil {
			t.Fatalf("Failed to decode response: %v", err)
		}
	})

	t.Run("Stream endpoint sets SSE headers", func(t *testing.T) {
		body := `{"prompt": "Hello", "max_tokens": 10, "stream": true}`
		req := httptest.NewRequest(http.MethodPost, "/api/stream", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()

		handlers.StreamHandler(handlers.Config{})(w, req)
		res := w.Result()
		defer res.Body.Close()

		if res.StatusCode != http.StatusOK {
			t.Errorf("Expected status 200, got %d", res.StatusCode)
		}

		contentType := res.Header.Get("Content-Type")
		if !strings.Contains(contentType, "text/event-stream") {
			t.Errorf("Expected Content-Type to contain 'text/event-stream', got '%s'", contentType)
		}

		cacheControl := res.Header.Get("Cache-Control")
		if cacheControl != "no-cache" {
			t.Errorf("Expected Cache-Control 'no-cache', got '%s'", cacheControl)
		}
	})
}

func TestConcurrentAPIRequests(t *testing.T) {
	numRequests := 50
	var wg sync.WaitGroup
	results := make(chan int, numRequests)

	for i := 0; i < numRequests; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			req := httptest.NewRequest(http.MethodGet, "/api/models", nil)
			w := httptest.NewRecorder()

			handlers.ModelsHandler(handlers.Config{})(w, req)
			results <- w.Code
		}(i)
	}

	wg.Wait()
	close(results)

	for code := range results {
		if code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", code)
		}
	}
}

func TestStreamingResponseFormat(t *testing.T) {
	body := `{"prompt": "Test", "max_tokens": 5, "stream": true}`
	req := httptest.NewRequest(http.MethodPost, "/api/stream", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	handlers.StreamHandler(handlers.Config{})(w, req)

	reader := bufio.NewReader(w.Body)
	lineCount := 0
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			break
		}
		lineCount++

		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")
			data = strings.TrimSpace(data)

			var tokenResp handlers.InferenceResponse
			if err := json.Unmarshal([]byte(data), &tokenResp); err != nil {
				t.Errorf("Failed to parse SSE data: %v (data: %s)", err, data)
			}
		}
	}

	if lineCount == 0 {
		t.Error("Expected at least one line in stream response")
	}
}

func TestWebSocketMessageProtocol(t *testing.T) {
	tests := []struct {
		name    string
		message interface{}
		valid   bool
	}{
		{
			name: "Valid inference request",
			message: handlers.WSMessage{
				Type: "inference",
				Payload: handlers.InferenceRequest{
					Prompt:    "Hello",
					MaxTokens: 10,
				},
			},
			valid: true,
		},
		{
			name: "Valid status request",
			message: handlers.WSMessage{
				Type:    "status",
				Payload: nil,
			},
			valid: true,
		},
		{
			name: "Valid stop request",
			message: handlers.WSMessage{
				Type:    "stop",
				Payload: nil,
			},
			valid: true,
		},
		{
			name: "Unknown message type",
			message: handlers.WSMessage{
				Type:    "unknown",
				Payload: nil,
			},
			valid: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.message)
			if err != nil {
				t.Fatalf("Failed to marshal: %v", err)
			}

			var parsed handlers.WSMessage
			if err := json.Unmarshal(data, &parsed); err != nil {
				t.Fatalf("Failed to unmarshal: %v", err)
			}

			isValid := parsed.Type == "inference" || parsed.Type == "status" || parsed.Type == "stop"
			if isValid != tt.valid {
				t.Errorf("Expected validity %v, got %v", tt.valid, isValid)
			}
		})
	}
}

func TestRateLimiting(t *testing.T) {
	manager := handlers.GetAPIKeyManager()
	key := "test-rate-limit-key"

	err := manager.AddKey(key, "rate-test", time.Hour, 5)
	if err != nil {
		t.Fatalf("Failed to add key: %v", err)
	}

	for i := 0; i < 5; i++ {
		if !manager.CheckRateLimit(key) {
			t.Errorf("Expected rate limit to pass on request %d", i+1)
		}
	}

	if manager.CheckRateLimit(key) {
		t.Error("Expected rate limit to fail on 6th request")
	}
}

func TestGenerateRequestFormat(t *testing.T) {
	tests := []struct {
		name    string
		json    string
		wantErr bool
	}{
		{
			name:    "Valid request",
			json:    `{"prompt": "Hello", "temperature": 0.7, "top_k": 40, "top_p": 0.95, "max_tokens": 100}`,
			wantErr: false,
		},
		{
			name:    "Missing prompt",
			json:    `{"temperature": 0.7}`,
			wantErr: true,
		},
		{
			name:    "Invalid temperature",
			json:    `{"prompt": "Hello", "temperature": 5.0}`,
			wantErr: true,
		},
		{
			name:    "With model specified",
			json:    `{"prompt": "Hello", "model": "smollm2", "max_tokens": 50}`,
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var req handlers.GenerateRequest
			err := json.Unmarshal([]byte(tt.json), &req)

			hasError := err != nil || req.Prompt == "" || req.Temperature < 0 || req.Temperature > 2
			if hasError != tt.wantErr {
				t.Errorf("Expected error=%v, got error=%v (prompt=%s)", tt.wantErr, hasError, req.Prompt)
			}
		})
	}
}

func TestModelInfoFormat(t *testing.T) {
	info := handlers.ModelInfo{
		Name:         "smollm2",
		Path:         "/models/smollm2.gguf",
		Parameters:   "1.7B",
		Quantization: "Q4_K_M",
		Loaded:       true,
	}

	data, err := json.Marshal(info)
	if err != nil {
		t.Fatalf("Failed to marshal: %v", err)
	}

	var decoded handlers.ModelInfo
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("Failed to unmarshal: %v", err)
	}

	if decoded.Name != info.Name {
		t.Errorf("Expected name '%s', got '%s'", info.Name, decoded.Name)
	}

	if decoded.Loaded != info.Loaded {
		t.Errorf("Expected loaded %v, got %v", info.Loaded, decoded.Loaded)
	}
}

func TestHTTPMethodsEnforcement(t *testing.T) {
	endpoints := []string{"/api/models", "/api/generate", "/api/stream"}

	for _, endpoint := range endpoints {
		t.Run(endpoint+" with wrong method", func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, endpoint, nil)
			w := httptest.NewRecorder()

			switch endpoint {
			case "/api/models":
				handlers.ModelsHandler(handlers.Config{})(w, req)
			case "/api/generate":
				handlers.GenerateHandler(handlers.Config{})(w, req)
			case "/api/stream":
				handlers.StreamHandler(handlers.Config{})(w, req)
			}

			if w.Code != http.StatusMethodNotAllowed {
				t.Errorf("Expected status 405, got %d", w.Code)
			}
		})
	}
}

func BenchmarkHealthEndpoint(b *testing.B) {
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()

	for i := 0; i < b.N; i++ {
		handlers.HealthHandler()(w, req)
		w.Body.Reset()
	}
}

func BenchmarkModelsEndpoint(b *testing.B) {
	req := httptest.NewRequest(http.MethodGet, "/api/models", nil)
	w := httptest.NewRecorder()

	for i := 0; i < b.N; i++ {
		handlers.ModelsHandler(handlers.Config{})(w, req)
		w.Body.Reset()
	}
}

func BenchmarkConcurrentRequests(b *testing.B) {
	b.RunParallel(func(pb *testing.PB) {
		req := httptest.NewRequest(http.MethodGet, "/api/models", nil)
		w := httptest.NewRecorder()

		for pb.Next() {
			handlers.ModelsHandler(handlers.Config{})(w, req)
			w.Body.Reset()
		}
	})
}

func TestRequestBodySizeLimits(t *testing.T) {
	largeBody := make([]byte, 1024*1025)
	for i := range largeBody {
		largeBody[i] = 'a'
	}

	req := httptest.NewRequest(http.MethodPost, "/api/generate", bytes.NewReader(largeBody))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	handlers.GenerateHandler(handlers.Config{})(w, req)
	res := w.Result()
	defer res.Body.Close()

	if res.StatusCode != http.StatusRequestEntityTooLarge {
		t.Errorf("Expected status 413, got %d", res.StatusCode)
	}
}

func TestCORSHeadersPresent(t *testing.T) {
	tests := []struct {
		name        string
		origin      string
		allowedOrig []string
		expectCORS  bool
	}{
		{
			name:        "No CORS configured",
			origin:      "http://example.com",
			allowedOrig: nil,
			expectCORS:  false,
		},
		{
			name:        "Origin allowed",
			origin:      "http://localhost:3000",
			allowedOrig: []string{"http://localhost:3000"},
			expectCORS:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			middleware := handlers.NewCORSMiddleware(tt.allowedOrig)

			handler := middleware.Middleware(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusOK)
			})

			req := httptest.NewRequest(http.MethodGet, "/api/models", nil)
			req.Header.Set("Origin", tt.origin)

			w := httptest.NewRecorder()
			handler(w, req)
			res := w.Result()
			defer res.Body.Close()

			acao := res.Header.Get("Access-Control-Allow-Origin")
			hasCORS := acao != ""
			if hasCORS != tt.expectCORS {
				t.Errorf("Expected CORS=%v, got ACAO='%s'", tt.expectCORS, acao)
			}
		})
	}
}
