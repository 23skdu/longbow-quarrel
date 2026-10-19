package engine

import (
	"strings"
	"testing"
)

// TestPromptWrapper_RendersLlama3Template checks that the Jinja2-subset
// renderer handles the Llama 3 template shape.
func TestPromptWrapper_RendersLlama3Template(t *testing.T) {
	tmpl := `{{ bos_token }}{% for message in messages %}{{'<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>'}}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}`

	p := &PromptWrapper{ChatTemplate: tmpl}
	got, err := p.Wrap([]Message{
		{Role: "system", Content: "You are helpful."},
		{Role: "user", Content: "Hello"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	for _, want := range []string{
		"<|start_header_id|>system<|end_header_id|>",
		"You are helpful.",
		"<|start_header_id|>user<|end_header_id|>",
		"Hello",
		"<|eot_id|>",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("rendered prompt is missing %q:\n%s", want, got)
		}
	}
}

// TestPromptWrapper_RendersMistralTemplate checks the Mistral [INST] shape.
func TestPromptWrapper_RendersMistralTemplate(t *testing.T) {
	tmpl := `{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% endif %}{% endfor %}`

	p := &PromptWrapper{ChatTemplate: tmpl}
	got, err := p.Wrap([]Message{
		{Role: "user", Content: "Hello"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(got, "[INST] Hello [/INST]") {
		t.Errorf("rendered prompt does not contain the Mistral instruction block:\n%s", got)
	}
}

// TestPromptWrapper_EmptyTemplateFallsBack checks that an empty template
// still produces a usable prompt.
func TestPromptWrapper_EmptyTemplateFallsBack(t *testing.T) {
	p := &PromptWrapper{}
	got, err := p.Wrap([]Message{{Role: "user", Content: "Hello"}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(got, "Hello") {
		t.Errorf("the fallback prompt does not contain the user message:\n%s", got)
	}
}

// TestPromptWrapper_SystemPromptIsUsed checks that the system prompt is
// applied when no system message is present.
func TestPromptWrapper_SystemPromptIsUsed(t *testing.T) {
	p := &PromptWrapper{
		ChatTemplate: "{{ .System }}|{{ .User }}",
		SystemPrompt: "default system",
	}
	got, err := p.Wrap([]Message{{Role: "user", Content: "hi"}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(got, "default system") {
		t.Errorf("the system prompt was not applied:\n%s", got)
	}
}
