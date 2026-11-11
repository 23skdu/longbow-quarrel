package api

import (
	"encoding/json"
	"testing"
)

func FuzzChatCompletionRequestParsing(f *testing.F) {
	f.Add([]byte(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`))
	f.Fuzz(func(t *testing.T, data []byte) {
		var req ChatCompletionRequest
		if err := json.Unmarshal(data, &req); err != nil {
			return
		}
		// Re-marshal must not panic.
		b, err := json.Marshal(req)
		if err != nil {
			t.Fatalf("marshal failed: %v", err)
		}
		if len(b) == 0 {
			t.Fatal("empty marshal output")
		}
	})
}
