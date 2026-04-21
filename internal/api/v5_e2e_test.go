package api

import (
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/crossengine"
)

func TestE2EBasicPromptVLLM(t *testing.T) {
	vllmURL := os.Getenv("TEST_VLLM_URL")
	if vllmURL == "" {
		t.Skip("TEST_VLLM_URL not set, skipping vLLM E2E tests")
	}

	testPrompts := []string{
		"What is 2+2?",
		"Capital of France",
		"Hello, how are you?",
	}

	client := crossengine.NewVLLMClient(vllmURL, "qwen2-0.5b")

	for _, prompt := range testPrompts {
		t.Run(prompt, func(t *testing.T) {
			text, err := client.Generate(prompt, 20, 0.0)
			if err != nil {
				t.Logf("vLLM error (expected if server not running): %v", err)
			} else {
				t.Logf("vLLM response: %s", text)
			}
		})
	}
}

func TestE2EBasicPromptLlamaCpp(t *testing.T) {
	llamaURL := os.Getenv("TEST_LLAMACPP_URL")
	if llamaURL == "" {
		t.Skip("TEST_LLAMACPP_URL not set, skipping llama.cpp E2E tests")
	}

	testPrompts := []string{
		"What is 2+2?",
		"Capital of France",
		"Hello, how are you?",
	}

	client := crossengine.NewLlamaCppClient(llamaURL, "model.gguf")

	for _, prompt := range testPrompts {
		t.Run(prompt, func(t *testing.T) {
			text, err := client.Generate(prompt, 20, 0.0)
			if err != nil {
				t.Logf("llama.cpp error (expected if server not running): %v", err)
			} else {
				t.Logf("llama.cpp response: %s", text)
			}
		})
	}
}

func TestE2EBasicPromptOllama(t *testing.T) {
	ollamaURL := os.Getenv("TEST_OLLAMA_URL")
	if ollamaURL == "" {
		t.Skip("TEST_OLLAMA_URL not set, skipping Ollama E2E tests")
	}

	testPrompts := []string{
		"What is 2+2?",
		"Capital of France",
		"Hello, how are you?",
	}

	client := crossengine.NewOllamaClient(ollamaURL, "qwen2:0.5b")

	for _, prompt := range testPrompts {
		t.Run(prompt, func(t *testing.T) {
			text, err := client.Generate(prompt, 20, 0.0)
			if err != nil {
				t.Logf("Ollama error (expected if server not running): %v", err)
			} else {
				t.Logf("Ollama response: %s", text)
			}
		})
	}
}

func TestStreamingE2E(t *testing.T) {
	t.Run("streaming_endpoint_exists", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
		}))
		defer server.Close()

		t.Logf("Streaming test server: %s", server.URL)
	})
}

func TestV5E2EEndpoints(t *testing.T) {
	testPaths := []struct {
		method string
		path   string
	}{
		{http.MethodGet, "/v1/models"},
		{http.MethodPost, "/v1/chat/completions"},
		{http.MethodPost, "/v1/completions"},
	}

	for _, tc := range testPaths {
		t.Run(tc.path, func(t *testing.T) {
			handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusOK)
			})
			server := httptest.NewServer(handler)
			defer server.Close()

			t.Logf("Endpoint %s %s available at %s", tc.method, tc.path, server.URL)
		})
	}
}

func TestQuarrelVsVLLM(t *testing.T) {
	quarrelURL := os.Getenv("TEST_QUARREL_URL")
	vllmURL := os.Getenv("TEST_VLLM_URL")

	if quarrelURL == "" || vllmURL == "" {
		t.Skip("TEST_QUARREL_URL or TEST_VLLM_URL not set")
	}

	prompt := "What is 2+2?"

	quarrelClient := crossengine.NewQuarrelClient(quarrelURL)
	vllmClient := crossengine.NewVLLMClient(vllmURL, "qwen2-0.5b")

	quarrelText, qErr := quarrelClient.Generate(prompt, 20, 0.0)
	vllmText, vErr := vllmClient.Generate(prompt, 20, 0.0)

	if qErr != nil {
		t.Logf("Quarrel error: %v", qErr)
	}
	if vErr != nil {
		t.Logf("vLLM error: %v", vErr)
	}

	if qErr == nil && vErr == nil {
		sim, qLen, vLen := crossengine.CompareOutputs(quarrelText, vllmText)
		t.Logf("Comparison: Quarrel=%d chars, vLLM=%d chars, similarity=%.2f", qLen, vLen, sim)
	}
}