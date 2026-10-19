package engine

import (
	"strings"
	"testing"
)

// The templates below are the real ones shipped with the models, trimmed to
// the parts that matter. They are the regression guard for the Jinja2-subset
// renderer: the body of a `for` block used to render as the empty string, so
// every one of these templates produced just "<s>".

// TestChatTemplate_RealWorldTemplates renders the templates that Llama 3,
// Qwen, Gemma and Mistral ship, and checks the expected markers.
func TestChatTemplate_RealWorldTemplates(t *testing.T) {
	messages := []Message{
		{Role: "system", Content: "Be nice."},
		{Role: "user", Content: "Hi"},
	}

	cases := []struct {
		name     string
		template string
		want     []string
		notWant  []string
	}{
		{
			name: "llama3",
			template: `{{ bos_token }}{% for message in messages %}` +
				`{{'<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>'}}` +
				`{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}`,
			want: []string{
				"<|start_header_id|>system<|end_header_id|>",
				"Be nice.",
				"<|start_header_id|>user<|end_header_id|>",
				"Hi",
				"<|eot_id|>",
				"<|start_header_id|>assistant<|end_header_id|>",
			},
			// The escape sequence must become a real newline, not the two
			// characters \ and n.
			notWant: []string{`\n`},
		},
		{
			name: "qwen",
			template: `{% for message in messages %}` +
				`{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}` +
				`{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}`,
			want: []string{
				"<|im_start|>system\nBe nice.<|im_end|>",
				"<|im_start|>user\nHi<|im_end|>",
				"<|im_start|>assistant\n",
			},
			notWant: []string{`\n`},
		},
		{
			name: "gemma",
			template: `{{ bos_token }}{% for message in messages %}` +
				`{{'<start_of_turn>' + message['role'] + '\n' + message['content'] + '<end_of_turn>\n'}}` +
				`{% endfor %}`,
			want: []string{
				"<start_of_turn>system\nBe nice.<end_of_turn>",
				"<start_of_turn>user\nHi<end_of_turn>",
			},
		},
		{
			name: "mistral",
			template: `{{ bos_token }}{% for message in messages %}` +
				`{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}` +
				`{% elif message['role'] == 'assistant' %}{{ message['content'] + ' ' }}` +
				`{% endif %}{% endfor %}`,
			want: []string{
				"[INST] Hi [/INST]",
			},
			// The system message must not be wrapped in an instruction block.
			notWant: []string{"[INST] Be nice. [/INST]"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := &PromptWrapper{ChatTemplate: tc.template}
			got, err := p.Wrap(messages)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			for _, w := range tc.want {
				if !strings.Contains(got, w) {
					t.Errorf("rendered prompt is missing %q:\n%q", w, got)
				}
			}
			for _, w := range tc.notWant {
				if strings.Contains(got, w) {
					t.Errorf("rendered prompt unexpectedly contains %q:\n%q", w, got)
				}
			}
		})
	}
}

// TestChatTemplate_ForBodyIsRendered is the minimal guard for the bug that
// made every real template render as just "<s>".
func TestChatTemplate_ForBodyIsRendered(t *testing.T) {
	p := &PromptWrapper{
		ChatTemplate: `{% for message in messages %}{{ message.content }}{% endfor %}`,
	}
	got, err := p.Wrap([]Message{
		{Role: "user", Content: "one"},
		{Role: "assistant", Content: "two"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(got, "one") || !strings.Contains(got, "two") {
		t.Errorf("the loop body was not rendered: %q", got)
	}
}

// TestChatTemplate_NestedBlockIsSplitCorrectly checks that a block inside a
// block is handled: the search for the outer end tag must skip over the
// inner one.
func TestChatTemplate_NestedBlockIsSplitCorrectly(t *testing.T) {
	p := &PromptWrapper{
		ChatTemplate: `{% for m in messages %}{% if m.role == 'user' %}[{{ m.content }}]{% endif %}{% endfor %}`,
	}
	got, err := p.Wrap([]Message{
		{Role: "user", Content: "yes"},
		{Role: "assistant", Content: "no"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(got, "[yes]") {
		t.Errorf("the nested if block was not rendered: %q", got)
	}
	if strings.Contains(got, "[no]") {
		t.Errorf("the condition was not evaluated: %q", got)
	}
}
