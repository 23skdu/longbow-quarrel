package api

import (
	"fmt"
	"net"
	"net/http"
	"testing"
	"time"

	"github.com/playwright-community/playwright-go"
)

func TestPlaywrightE2E_API(t *testing.T) {
	// 1. Initialize Playwright
	pw, err := playwright.Run()
	if err != nil {
		t.Skipf("skipping: could not start playwright driver: %v", err)
	}
	defer pw.Stop()

	// 2. Start a local server with a mock engine
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("failed to listen: %v", err)
	}
	port := l.Addr().(*net.TCPAddr).Port
	baseURL := fmt.Sprintf("http://127.0.0.1:%d", port)

	s := &Server{
		MaxMemory:  1024,
		UsedMemory: func() int64 { return 100 },
		Engine:     &mockEngine{},
		Tokenizer:  &mockTokenizer{},
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", s.HealthzEndpoint)
	mux.HandleFunc("/v1/completions", s.CompletionsHandler)

	httpServer := &http.Server{Handler: mux}
	go func() {
		_ = httpServer.Serve(l)
	}()
	defer httpServer.Close()

	// 3. Setup API Request Context
	requestContext, err := pw.Request.NewContext(playwright.APIRequestNewContextOptions{
		BaseURL: &baseURL,
	})
	if err != nil {
		t.Fatalf("could not create api request context: %v", err)
	}

	// 4. Test GET /healthz
	t.Run("Healthz", func(t *testing.T) {
		resp, err := requestContext.Get("/healthz")
		if err != nil {
			t.Fatalf("failed to get healthz: %v", err)
		}
		if resp.Status() != 200 {
			t.Errorf("expected 200, got %d", resp.Status())
		}
		
		var healthResp HealthResponse
		if err := resp.JSON(&healthResp); err != nil {
			t.Errorf("failed to parse JSON: %v", err)
		}
		if healthResp.Status != "ok" {
			t.Errorf("expected status ok, got %s", healthResp.Status)
		}
	})

	// 5. Test POST /v1/completions
	t.Run("Completions", func(t *testing.T) {
		reqBody := map[string]interface{}{
			"prompt":     "hello",
			"max_tokens": 10,
		}
		resp, err := requestContext.Post("/v1/completions", playwright.APIRequestContextPostOptions{
			Data: reqBody,
		})
		if err != nil {
			t.Fatalf("failed to post completions: %v", err)
		}
		if resp.Status() != 200 {
			t.Errorf("expected 200, got %d", resp.Status())
		}

		var compResp CompletionResponse
		if err := resp.JSON(&compResp); err != nil {
			t.Errorf("failed to parse JSON: %v", err)
		}
		if len(compResp.Choices) == 0 || compResp.Choices[0].Text == "" {
			t.Errorf("empty response text")
		}
	})

	// 6. Test Error path (405)
	t.Run("MethodNotAllowed", func(t *testing.T) {
		resp, err := requestContext.Get("/v1/completions")
		if err != nil {
			t.Fatalf("failed call: %v", err)
		}
		if resp.Status() != 405 {
			t.Errorf("expected 405, got %d", resp.Status())
		}
	})
}

// Ensure server is reachable before testing
func waitServer(url string) error {
	for i := 0; i < 10; i++ {
		_, err := http.Get(url + "/healthz")
		if err == nil {
			return nil
		}
		time.Sleep(100 * time.Millisecond)
	}
	return fmt.Errorf("server timeout")
}
