package engine

import (
	"strings"

	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

// PromptWrapper wraps prompts with system prompts and chat templates.
// If a GGUF chat_template is loaded via LoadGGUFChatTemplate, it is rendered
// using a lightweight Jinja2-subset renderer that covers Llama 3, Qwen, Gemma, and Mistral.
type PromptWrapper struct {
	SystemPrompt string
	ChatTemplate string
	StopStrings  []string
	GenParams    GenerationConfig
}

// GenerationConfig wraps generation parameters
type GenerationConfig struct {
	Temperature   float32
	TopP          float32
	TopK          int
	NumCtx        int
	NumPredict    int
	RepeatPenalty float32
}

// NewPromptWrapper creates a new PromptWrapper with default settings.
func NewPromptWrapper() *PromptWrapper {
	return &PromptWrapper{
		ChatTemplate: "{{ .System }}{{ .User }}: {{ .Input }}{{ .Response }}:",
		StopStrings:  []string{"[INST]", "<|end|>", "<|eot_id|>", "\n\n"},
		GenParams: GenerationConfig{
			Temperature:   0.8,
			TopP:          0.95,
			TopK:          40,
			NumCtx:        4096,
			NumPredict:    2048,
			RepeatPenalty: 1.1,
		},
	}
}

// LoadGGUFChatTemplate reads the tokenizer.chat_template key from a GGUF file and
// installs it as the active template.  Falls back to the existing template on error.
func (p *PromptWrapper) LoadGGUFChatTemplate(f *gguf.GGUFFile) {
	if f == nil {
		return
	}
	if tmpl, ok := f.KV["tokenizer.chat_template"].(string); ok && tmpl != "" {
		p.ChatTemplate = tmpl
	}
}

// Wrap applies the chat template to create the full prompt.
// If the template looks like a Jinja2 template (contains {%  or {{ ), the built-in
// Jinja2-subset renderer is used.  Otherwise the legacy Go-template substitution runs.
func (p *PromptWrapper) Wrap(messages []Message) (string, error) {
	if p.ChatTemplate == "" {
		return p.simpleWrap(messages), nil
	}

	// If template contains Go template variables ({{ .Field }} or {{.Field}}), use legacy substitution.
	// Otherwise, detect Jinja2 template ({% or {{).
	if !strings.Contains(p.ChatTemplate, "{{ .") && !strings.Contains(p.ChatTemplate, "{{.") {
		if strings.Contains(p.ChatTemplate, "{%") || strings.Contains(p.ChatTemplate, "{{") {
			result, err := renderJinja2Subset(p.ChatTemplate, messages, p.SystemPrompt)
			if err == nil {
				return result, nil
			}
			// Fall through to legacy rendering on error
		}
	}

	// Legacy Go-template substitution
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
	if system == "" {
		system = p.SystemPrompt
	}

	result := p.ChatTemplate
	result = strings.ReplaceAll(result, "{{ .System }}", system)
	result = strings.ReplaceAll(result, "{{ .User }}", user)
	result = strings.ReplaceAll(result, "{{ .Input }}", user)
	result = strings.ReplaceAll(result, "{{ .Response }}", assistant)
	return result, nil
}

// renderJinja2Subset renders a Jinja2 chat template using a lightweight subset renderer
// that supports the patterns found in Llama 3, Qwen 2/3, Gemma 2/3, and Mistral templates.
// Supported: {{ variable }}, {% if cond %}...{% endif %}, {% for msg in messages %}...{% endfor %},
// {% set var = value %}, {{ msg.role }}, {{ msg.content }}, {{ bos_token }}, {{ eos_token }}.
func renderJinja2Subset(tmpl string, messages []Message, systemPrompt string) (string, error) {
	// Build variable context
	vars := map[string]string{
		"bos_token":   "<s>",
		"eos_token":   "</s>",
		"system":      systemPrompt,
		"add_generation_prompt": "true",
	}

	// Standard chat token patterns recognized in the wild
	tokenAliases := map[string]string{
		"<|begin_of_text|>": "<|begin_of_text|>",
		"<|eot_id|>":        "<|eot_id|>",
		"<|im_start|>":      "<|im_start|>",
		"<|im_end|>":        "<|im_end|>",
		"<end_of_turn>":     "<end_of_turn>",
		"<start_of_turn>":   "<start_of_turn>",
		"[INST]":            "[INST]",
		"[/INST]":           "[/INST]",
	}
	for k, v := range tokenAliases {
		vars[k] = v
	}

	var sb strings.Builder

	// Tokenize template into segments
	remaining := tmpl

	for len(remaining) > 0 {
		// Find next tag
		blockStart := strings.Index(remaining, "{%")
		varStart := strings.Index(remaining, "{{")

		// Determine which comes first
		nextTag := -1
		tagType := ""
		if blockStart >= 0 && (varStart < 0 || blockStart <= varStart) {
			nextTag = blockStart
			tagType = "block"
		} else if varStart >= 0 {
			nextTag = varStart
			tagType = "var"
		}

		if nextTag < 0 {
			// No more tags; emit literal
			sb.WriteString(remaining)
			break
		}

		// Emit literal before tag
		sb.WriteString(remaining[:nextTag])
		remaining = remaining[nextTag:]

		if tagType == "var" {
			end := strings.Index(remaining, "}}")
			if end < 0 {
				sb.WriteString(remaining)
				break
			}
			expr := strings.TrimSpace(remaining[2:end])
			remaining = remaining[end+2:]
			// Evaluate expression
			val := evalVarExpr(expr, vars, messages)
			sb.WriteString(val)

		} else { // block
			end := strings.Index(remaining, "%}")
			if end < 0 {
				sb.WriteString(remaining)
				break
			}
			stmt := strings.TrimSpace(remaining[2:end])
			remaining = remaining[end+2:]

			if strings.HasPrefix(stmt, "for ") {
				// {% for msg in messages %} ... {% endfor %}
				// Find matching endfor
				inner, after := splitBlock(remaining, "endfor")
				remaining = after
				for _, msg := range messages {
					vars["msg.role"] = msg.Role
					vars["msg.content"] = msg.Content
					// Also support loop var if named "message"
					vars["message.role"] = msg.Role
					vars["message.content"] = msg.Content
					rendered, _ := renderJinja2Subset(inner, nil, vars["system"])
					// Pass vars through rendered inner
					rendered = applyVars(rendered, vars)
					sb.WriteString(rendered)
				}
				delete(vars, "msg.role")
				delete(vars, "msg.content")

			} else if strings.HasPrefix(stmt, "if ") {
				// {% if cond %} ... {% else %} ... {% endif %}
				inner, after := splitBlock(remaining, "endif")
				remaining = after
				cond := strings.TrimPrefix(stmt, "if ")
				if evalCondition(cond, vars) {
					// Check for else
					ifPart, _ := splitBlock(inner, "else")
					rendered, _ := renderJinja2Subset(ifPart, messages, vars["system"])
					sb.WriteString(rendered)
				} else {
					_, elsePart := splitBlock(inner, "else")
					if elsePart != "" {
						rendered, _ := renderJinja2Subset(elsePart, messages, vars["system"])
						sb.WriteString(rendered)
					}
				}

			} else if strings.HasPrefix(stmt, "set ") {
				// {% set var = value %}
				parts := strings.SplitN(strings.TrimPrefix(stmt, "set "), "=", 2)
				if len(parts) == 2 {
					k := strings.TrimSpace(parts[0])
					v := strings.Trim(strings.TrimSpace(parts[1]), "\"'")
					vars[k] = v
				}
			}
			// endif / endfor / else are consumed by splitBlock; ignore if encountered raw
		}
	}

	return sb.String(), nil
}

// splitBlock finds the matching end tag (e.g., "endfor", "endif", "else") in a Jinja2
// template fragment, respecting nested blocks. Returns (inner content, content after end tag).
func splitBlock(s, endTag string) (string, string) {
	depth := 0
	i := 0
	for i < len(s) {
		tagStart := strings.Index(s[i:], "{%")
		if tagStart < 0 {
			break
		}
		tagStart += i
		tagEnd := strings.Index(s[tagStart:], "%}")
		if tagEnd < 0 {
			break
		}
		tagEnd += tagStart + 2

		stmt := strings.TrimSpace(s[tagStart+2 : tagEnd-2])
		if strings.HasPrefix(stmt, "for ") || strings.HasPrefix(stmt, "if ") {
			depth++
			i = tagEnd
		} else if stmt == endTag && depth == 0 {
			return s[:tagStart], s[tagEnd:]
		} else if stmt == endTag {
			depth--
			i = tagEnd
		} else {
			i = tagEnd
		}
	}
	return s, ""
}

// evalVarExpr evaluates a Jinja2 variable expression like "msg.role", "bos_token", etc.
func evalVarExpr(expr string, vars map[string]string, messages []Message) string {
	_ = messages
	// Strip filters (e.g., "messages | selectattr(...)")
	if idx := strings.Index(expr, "|"); idx >= 0 {
		expr = strings.TrimSpace(expr[:idx])
	}
	if v, ok := vars[expr]; ok {
		return v
	}
	// Direct message field access
	if expr == "message.role" || expr == "msg.role" {
		if v, ok := vars["msg.role"]; ok {
			return v
		}
	}
	if expr == "message.content" || expr == "msg.content" {
		if v, ok := vars["msg.content"]; ok {
			return v
		}
	}
	return ""
}

// evalCondition evaluates a simple Jinja2 boolean condition.
func evalCondition(cond string, vars map[string]string) bool {
	cond = strings.TrimSpace(cond)
	// Handle "not X"
	if strings.HasPrefix(cond, "not ") {
		return !evalCondition(strings.TrimPrefix(cond, "not "), vars)
	}
	// Handle "X and Y"
	if idx := strings.Index(cond, " and "); idx >= 0 {
		return evalCondition(cond[:idx], vars) && evalCondition(cond[idx+5:], vars)
	}
	// Handle "X or Y"
	if idx := strings.Index(cond, " or "); idx >= 0 {
		return evalCondition(cond[:idx], vars) || evalCondition(cond[idx+4:], vars)
	}
	// Variable lookup
	if v, ok := vars[cond]; ok {
		return v != "" && v != "false" && v != "0"
	}
	// Check for equality test: X == "value"
	if idx := strings.Index(cond, " == "); idx >= 0 {
		lhs := strings.TrimSpace(cond[:idx])
		rhs := strings.Trim(strings.TrimSpace(cond[idx+4:]), "\"'")
		lhsVal := vars[lhs]
		return lhsVal == rhs
	}
	// Check "X != value"
	if idx := strings.Index(cond, " != "); idx >= 0 {
		lhs := strings.TrimSpace(cond[:idx])
		rhs := strings.Trim(strings.TrimSpace(cond[idx+4:]), "\"'")
		lhsVal := vars[lhs]
		return lhsVal != rhs
	}
	// Check "X is defined"
	if strings.HasSuffix(cond, " is defined") {
		key := strings.TrimSuffix(cond, " is defined")
		_, ok := vars[strings.TrimSpace(key)]
		return ok
	}
	return false
}

// applyVars replaces {{ key }} expressions in s using vars map.
func applyVars(s string, vars map[string]string) string {
	for k, v := range vars {
		s = strings.ReplaceAll(s, "{{ "+k+" }}", v)
		s = strings.ReplaceAll(s, "{{"+k+"}}", v)
	}
	return s
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