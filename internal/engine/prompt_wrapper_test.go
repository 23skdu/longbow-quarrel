package engine

import (
	"strings"
	"testing"
)

func TestPromptWrapper_New(t *testing.T) {
	pw := NewPromptWrapper()
	if pw == nil {
		t.Error("NewPromptWrapper returned nil")
	}
	if pw.ChatTemplate == "" {
		t.Error("ChatTemplate should not be empty")
	}
	if len(pw.StopStrings) == 0 {
		t.Error("StopStrings should not be empty")
	}
}

func TestPromptWrapper_Wrap(t *testing.T) {
	pw := NewPromptWrapper()

	messages := []Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "What is 2+2?"},
	}

	result, err := pw.Wrap(messages)
	if err != nil {
		t.Errorf("Wrap failed: %v", err)
	}

	if !strings.Contains(result, "You are a helpful assistant.") {
		t.Error("System prompt not in result")
	}
	if !strings.Contains(result, "What is 2+2?") {
		t.Error("User message not in result")
	}
}

func TestPromptWrapper_simpleWrap(t *testing.T) {
	pw := NewPromptWrapper()
	pw.ChatTemplate = ""

	messages := []Message{
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "Hi there"},
	}

	result := pw.simpleWrap(messages)

	if !strings.Contains(result, "[USER]Hello") {
		t.Errorf("Expected [USER]Hello in result, got: %s", result)
	}
	if !strings.Contains(result, "[ASSISTANT]Hi there") {
		t.Errorf("Expected [ASSISTANT]Hi there in result, got: %s", result)
	}
}

func TestPromptWrapper_FindStopString(t *testing.T) {
	pw := NewPromptWrapper()

	tests := []struct {
		text    string
		expect int
	}{
		{"Hello [INST] world", 6},
		{"Hello <|end|> world", 6},
		{"No stop string", -1},
		{"first[INST]second", 5},
	}

	for _, tc := range tests {
		idx := pw.FindStopString(tc.text)
		if idx != tc.expect {
			t.Errorf("FindStopString(%q): got %d, want %d", tc.text, idx, tc.expect)
		}
	}
}

func TestPromptWrapper_ChatTemplates(t *testing.T) {
	llama3 := "<|start_header_id|>system<|end_header_id|>\n\n{{ .System }}<|eot_id|>\n\n<|start_header_id|>user<|end_header_id|>\n\n{{ .User }}<|eot_id|>\n\n<|start_header_id|>assistant<|end_header_id|>\n\n"
	
	pw := &PromptWrapper{
		ChatTemplate: llama3,
		StopStrings: []string{"<|eot_id|>"},
	}

	messages := []Message{
		{Role: "system", Content: "You are smart."},
		{Role: "user", Content: "Hello"},
	}

	result, err := pw.Wrap(messages)
	if err != nil {
		t.Errorf("Wrap failed: %v", err)
	}

	if !strings.Contains(result, "<|start_header_id|>system<|end_header_id|>") {
		t.Error("Llama3 template not applied")
	}
}

func BenchmarkPromptWrapper_Wrap(b *testing.B) {
	pw := NewPromptWrapper()
	messages := []Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "What is the capital of France?"},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = pw.Wrap(messages)
	}
}