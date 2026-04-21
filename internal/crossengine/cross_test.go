package crossengine

import (
	"os"
	"testing"
)

func TestVLLMClientConfig(t *testing.T) {
	t.Run("new_vllm_client", func(t *testing.T) {
		client := NewVLLMClient("http://localhost:8000", "qwen2-0.5b")
		if client.Name() != "vLLM" {
			t.Errorf("Expected vLLM, got %s", client.Name())
		}
	})

	t.Run("default_timeout", func(t *testing.T) {
		client := NewVLLMClient("http://localhost:8000", "test")
		if client.client.Timeout == 0 {
			t.Error("Expected non-zero timeout")
		}
	})
}

func TestLlamaCppClientConfig(t *testing.T) {
	t.Run("new_llama_cpp_client", func(t *testing.T) {
		client := NewLlamaCppClient("http://localhost:8080", "model.gguf")
		if client.Name() != "llama.cpp" {
			t.Errorf("Expected llama.cpp, got %s", client.Name())
		}
	})
}

func TestOllamaClientConfig(t *testing.T) {
	t.Run("new_ollama_client", func(t *testing.T) {
		client := NewOllamaClient("http://localhost:11434", "qwen2:0.5b")
		if client.Name() != "Ollama" {
			t.Errorf("Expected Ollama, got %s", client.Name())
		}
	})

	t.Run("default_2min_timeout", func(t *testing.T) {
		client := NewOllamaClient("http://localhost:11434", "test")
		if client.client.Timeout <= 0 {
			t.Error("Expected non-zero timeout for Ollama")
		}
	})
}

func TestQuarrelClientConfig(t *testing.T) {
	t.Run("new_quarrel_client", func(t *testing.T) {
		client := NewQuarrelClient("http://localhost:8080")
		if client.Name() != "Quarrel" {
			t.Errorf("Expected Quarrel, got %s", client.Name())
		}
	})
}

func TestCompareOutputs(t *testing.T) {
	t.Run("identical_outputs", func(t *testing.T) {
		sim, aLen, bLen := CompareOutputs("hello world", "hello world")
		if sim != 1.0 {
			t.Errorf("Expected similarity 1.0, got %f", sim)
		}
		if aLen != 2 || bLen != 2 {
			t.Errorf("Expected word count 2, got %d and %d", aLen, bLen)
		}
	})

	t.Run("partial_overlap", func(t *testing.T) {
		sim, aLen, bLen := CompareOutputs("hello world", "hello there world")
		t.Logf("Similarity: %f, words: %d vs %d", sim, aLen, bLen)
	})

	t.Run("no_overlap", func(t *testing.T) {
		sim, _, _ := CompareOutputs("hello", "goodbye")
		if sim != 0.0 {
			t.Errorf("Expected similarity 0.0, got %f", sim)
		}
	})
}

func TestMustParseURL(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"http://localhost:8000", "http://localhost:8000"},
		{"localhost:8000", "http://localhost:8000"},
		{" http://localhost:8080 ", "http://localhost:8080"},
		{"http://localhost:8000/", "http://localhost:8000"},
	}

	for _, tc := range tests {
		t.Run(tc.input, func(t *testing.T) {
			result := MustParseURL(tc.input)
			if result != tc.expected {
				t.Errorf("Expected %s, got %s", tc.expected, result)
			}
		})
	}
}

func TestTokenize(t *testing.T) {
	t.Run("simple_text", func(t *testing.T) {
		tokens := Tokenize("hello")
		if len(tokens) == 0 {
			t.Error("Expected non-empty tokens")
		}
		t.Logf("Tokenize result: %v", tokens)
	})

	t.Run("unicode", func(t *testing.T) {
		tokens := Tokenize("hello世界")
		if len(tokens) == 0 {
			t.Error("Expected non-empty tokens for unicode")
		}
		t.Logf("Unicode tokens: %v", tokens)
	})
}

func TestEnvironmentCheck(t *testing.T) {
	t.Run("vllm_url_env", func(t *testing.T) {
		url := os.Getenv("TEST_VLLM_URL")
		if url != "" {
			t.Logf("vLLM URL from env: %s", url)
		} else {
			t.Log("TEST_VLLM_URL not set, skipping live vLLM tests")
		}
	})

	t.Run("llama_cpp_url_env", func(t *testing.T) {
		url := os.Getenv("TEST_LLAMACPP_URL")
		if url != "" {
			t.Logf("llama.cpp URL from env: %s", url)
		} else {
			t.Log("TEST_LLAMACPP_URL not set, skipping live llama.cpp tests")
		}
	})

	t.Run("ollama_url_env", func(t *testing.T) {
		url := os.Getenv("TEST_OLLAMA_URL")
		if url != "" {
			t.Logf("Ollama URL from env: %s", url)
		} else {
			t.Log("TEST_OLLAMA_URL not set, skipping live Ollama tests")
		}
	})
}