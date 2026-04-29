package engine

import (
	"strings"
)

// PromptWrapper wraps prompts with system prompts and chat templates
type PromptWrapper struct {
	SystemPrompt string
	ChatTemplate string
	StopStrings []string
	GenParams   GenerationConfig
}

// GenerationConfig wraps generation parameters
type GenerationConfig struct {
	Temperature   float32
	TopP         float32
	TopK         int
	NumCtx       int
	NumPredict   int
	RepeatPenalty float32
}

// NewPromptWrapper creates a new PromptWrapper with default settings
func NewPromptWrapper() *PromptWrapper {
	return &PromptWrapper{
		ChatTemplate: "{{ .System }}{{ .User }}: {{ .Input }}{{ .Response }}:",
		StopStrings: []string{"[INST]", "<|end|>", "<|eot_id|>", "\n\n"},
		GenParams: GenerationConfig{
			Temperature:   0.8,
			TopP:         0.95,
			TopK:         40,
			NumCtx:       4096,
			NumPredict:  2048,
			RepeatPenalty: 1.1,
		},
	}
}

// Wrap applies the chat template to create the full prompt
func (p *PromptWrapper) Wrap(messages []Message) (string, error) {
	if p.ChatTemplate == "" {
		return p.simpleWrap(messages), nil
	}

	var system, user, assistant string

	for _, msg := range messages {
		switch msg.Role {
		case "system":
			system = msg.Content
		case "user":
			if user != "" {
				user += "\n"
			}
			user += msg.Content
		case "assistant":
			if assistant != "" {
				assistant += "\n"
			}
			assistant += msg.Content
		}
	}

	result := p.ChatTemplate
	result = strings.ReplaceAll(result, "{{ .System }}", system)
	result = strings.ReplaceAll(result, "{{ .User }}", user)
	result = strings.ReplaceAll(result, "{{ .Input }}", user)
	result = strings.ReplaceAll(result, "{{ .Response }}", assistant)

	return result, nil
}

// simpleWrap creates a simple concatenated prompt
func (p *PromptWrapper) simpleWrap(messages []Message) string {
	var result strings.Builder

	for _, msg := range messages {
		role := msg.Role
		if role == "" {
			role = "user"
		}
		result.WriteString("[")
		result.WriteString(strings.ToUpper(role))
		result.WriteString("]")
		result.WriteString(msg.Content)
		result.WriteString("\n")
	}

	return result.String()
}

// FindStopString finds the first occurrence of any stop string
func (p *PromptWrapper) FindStopString(text string) int {
	if len(p.StopStrings) == 0 {
		return -1
	}

	lowest := -1
	for _, stop := range p.StopStrings {
		idx := strings.Index(text, stop)
		if idx >= 0 {
			if lowest < 0 || idx < lowest {
				lowest = idx
			}
		}
	}

	return lowest
}

// Message represents a chat message
type Message struct {
	Role    string
	Content string
}