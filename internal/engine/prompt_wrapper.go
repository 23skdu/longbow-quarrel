package engine

import (
	"strings"
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
	return renderTemplate(tmpl, messages, defaultTemplateVars(systemPrompt))
}

// defaultTemplateVars builds the variable context a template is rendered
// against.
func defaultTemplateVars(systemPrompt string) map[string]string {
	vars := map[string]string{
		"bos_token":             "<s>",
		"eos_token":             "</s>",
		"system":                systemPrompt,
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
	return vars
}

// renderTemplate renders a template fragment against an existing variable
// context.
//
// The context is threaded through the recursion so that the loop variables
// set by a `for` block survive into the body. The previous implementation
// rebuilt the context on every recursive call, which discarded them: the
// body of a `for` block rendered as the empty string, so the real Llama 3,
// Qwen, Gemma and Mistral chat templates produced just "<s>".
func renderTemplate(tmpl string, messages []Message, vars map[string]string) (string, error) {
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

				// The loop variable can be named anything ("msg",
				// "message", "m", ...), so it has to be taken from the
				// statement rather than assumed.
				loopVar := "message"
				if spec := strings.TrimSpace(strings.TrimPrefix(stmt, "for ")); spec != "" {
					if name, _, ok := strings.Cut(spec, " in "); ok {
						loopVar = strings.TrimSpace(name)
					} else {
						loopVar = strings.TrimSpace(spec)
					}
				}

				for _, msg := range messages {
					vars["msg.role"] = msg.Role
					vars["msg.content"] = msg.Content
					// The loop variable under its own name, and the
					// conventional "message" alias.
					vars[loopVar+".role"] = msg.Role
					vars[loopVar+".content"] = msg.Content
					vars["message.role"] = msg.Role
					vars["message.content"] = msg.Content
					rendered, _ := renderTemplate(inner, messages, vars)
					sb.WriteString(rendered)
				}
				delete(vars, "msg.role")
				delete(vars, "msg.content")
				delete(vars, loopVar+".role")
				delete(vars, loopVar+".content")
				delete(vars, "message.role")
				delete(vars, "message.content")

			} else if strings.HasPrefix(stmt, "if ") {
				// {% if cond %} ... {% else %} ... {% endif %}
				inner, after := splitBlock(remaining, "endif")
				remaining = after
				cond := strings.TrimPrefix(stmt, "if ")
				if evalCondition(cond, vars) {
					// Check for else
					ifPart, _ := splitBlock(inner, "else")
					rendered, _ := renderTemplate(ifPart, messages, vars)
					sb.WriteString(rendered)
				} else {
					_, elsePart := splitBlock(inner, "else")
					if elsePart != "" {
						rendered, _ := renderTemplate(elsePart, messages, vars)
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
		switch {
		case strings.HasPrefix(stmt, "for ") || strings.HasPrefix(stmt, "if "):
			depth++
			i = tagEnd
		case stmt == endTag:
			if depth == 0 {
				return s[:tagStart], s[tagEnd:]
			}
			depth--
			i = tagEnd
		case stmt == "endif" || stmt == "endfor":
			// A different block terminator. It closes a nested block, so the
			// nesting depth still decreases even though it is not the tag
			// being looked for. Without this, a `{% if %}` inside a
			// `{% for %}` made the search for "endfor" run past the end of
			// the loop body.
			depth--
			i = tagEnd
		default:
			i = tagEnd
		}
	}
	return s, ""
}

// evalVarExpr evaluates a Jinja2 variable expression.
//
// It supports the forms the shipped chat templates actually use:
//
//	{{ bos_token }}
//	{{ message['role'] }}          {{ message.role }}
//	{{ '[INST] ' + message['content'] + ' [/INST]' }}
//	{{ message['content'] | trim }}
//
// The filter is stripped only at the top level: the Llama 3 template
// contains "{{ '<|start_header_id|>' + ... }}", where the pipe is inside a
// quoted literal and must not be treated as a filter separator.
func evalVarExpr(expr string, vars map[string]string, messages []Message) string {
	_ = messages
	expr = strings.TrimSpace(expr)

	// Strip a trailing filter, but only at the top level.
	if idx := indexTopLevel(expr, "|"); idx >= 0 {
		expr = strings.TrimSpace(expr[:idx])
	}

	// String concatenation: a + b + c
	if parts := splitTopLevel(expr, "+"); len(parts) > 1 {
		var sb strings.Builder
		for _, p := range parts {
			sb.WriteString(evalOperand(p, vars))
		}
		return sb.String()
	}

	return evalOperand(expr, vars)
}

// evalOperand evaluates a single term of an expression.
func evalOperand(expr string, vars map[string]string) string {
	expr = strings.TrimSpace(expr)
	if expr == "" {
		return ""
	}

	// A quoted literal.
	if len(expr) >= 2 {
		if (expr[0] == '\'' && expr[len(expr)-1] == '\'') ||
			(expr[0] == '"' && expr[len(expr)-1] == '"') {
			return unescapeLiteral(expr[1 : len(expr)-1])
		}
	}

	// name['key'] / name["key"]
	if name, key, ok := parseIndexed(expr); ok {
		if v, found := vars[name+"."+key]; found {
			return v
		}
		return ""
	}

	if v, ok := vars[expr]; ok {
		return v
	}
	return ""
}

// unescapeLiteral processes the escape sequences Jinja2 recognises inside a
// quoted string literal.
//
// The chat templates rely on this: the Llama 3 template contains
// "'<|end_header_id|>\n\n' + message['content']", and without the escape
// processing the prompt would contain a literal backslash-n instead of the
// two newlines the model expects.
func unescapeLiteral(s string) string {
	if !strings.ContainsRune(s, '\\') {
		return s
	}
	var sb strings.Builder
	sb.Grow(len(s))
	for i := 0; i < len(s); i++ {
		if s[i] != '\\' || i+1 >= len(s) {
			sb.WriteByte(s[i])
			continue
		}
		i++
		switch s[i] {
		case 'n':
			sb.WriteByte('\n')
		case 't':
			sb.WriteByte('\t')
		case 'r':
			sb.WriteByte('\r')
		case '\\':
			sb.WriteByte('\\')
		case '\'':
			sb.WriteByte('\'')
		case '"':
			sb.WriteByte('"')
		default:
			sb.WriteByte('\\')
			sb.WriteByte(s[i])
		}
	}
	return sb.String()
}

// parseIndexed parses "name['key']" or "name[\"key\"]".
func parseIndexed(expr string) (name, key string, ok bool) {
	open := strings.Index(expr, "[")
	if open <= 0 || !strings.HasSuffix(expr, "]") {
		return "", "", false
	}
	name = strings.TrimSpace(expr[:open])
	inner := strings.TrimSpace(expr[open+1 : len(expr)-1])
	if len(inner) >= 2 {
		if (inner[0] == '\'' && inner[len(inner)-1] == '\'') ||
			(inner[0] == '"' && inner[len(inner)-1] == '"') {
			return name, inner[1 : len(inner)-1], true
		}
	}
	return "", "", false
}

// indexTopLevel returns the index of sep in s, ignoring occurrences inside
// quoted literals. It returns -1 when sep does not occur at the top level.
//
// The whole separator has to match: matching on sep[0] alone would make
// " or ", " and " and " == " all match on a plain space.
func indexTopLevel(s, sep string) int {
	if sep == "" {
		return -1
	}
	inSingle, inDouble := false, false
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '\'':
			if !inDouble {
				inSingle = !inSingle
			}
			continue
		case '"':
			if !inSingle {
				inDouble = !inDouble
			}
			continue
		}
		if inSingle || inDouble {
			continue
		}
		if i+len(sep) <= len(s) && s[i:i+len(sep)] == sep {
			return i
		}
	}
	return -1
}

// splitTopLevel splits s on sep, ignoring separators inside quoted literals.
func splitTopLevel(s, sep string) []string {
	var parts []string
	start := 0
	for {
		idx := indexTopLevel(s[start:], sep)
		if idx < 0 {
			parts = append(parts, s[start:])
			return parts
		}
		parts = append(parts, s[start:start+idx])
		start += idx + len(sep)
	}
}

// evalCondition evaluates a simple Jinja2 boolean condition.
func evalCondition(cond string, vars map[string]string) bool {
	cond = strings.TrimSpace(cond)

	// Handle "not X"
	if strings.HasPrefix(cond, "not ") {
		return !evalCondition(strings.TrimPrefix(cond, "not "), vars)
	}

	// Handle "X and Y" / "X or Y", splitting only at the top level so a
	// quoted literal containing " or " is not mistaken for an operator.
	if parts := splitTopLevel(cond, " or "); len(parts) > 1 {
		for _, p := range parts {
			if evalCondition(p, vars) {
				return true
			}
		}
		return false
	}
	if parts := splitTopLevel(cond, " and "); len(parts) > 1 {
		for _, p := range parts {
			if !evalCondition(p, vars) {
				return false
			}
		}
		return true
	}

	// Check for equality test: X == "value"
	if idx := indexTopLevel(cond, " == "); idx >= 0 {
		lhs := strings.TrimSpace(cond[:idx])
		rhs := strings.TrimSpace(cond[idx+4:])
		return evalOperand(lhs, vars) == evalOperand(rhs, vars)
	}

	// Check "X != value"
	if idx := indexTopLevel(cond, " != "); idx >= 0 {
		lhs := strings.TrimSpace(cond[:idx])
		rhs := strings.TrimSpace(cond[idx+4:])
		return evalOperand(lhs, vars) != evalOperand(rhs, vars)
	}

	// Check "X is defined"
	if strings.HasSuffix(cond, " is defined") {
		key := strings.TrimSuffix(cond, " is defined")
		key = strings.TrimSpace(key)
		if _, ok := vars[key]; ok {
			return true
		}
		_, _, ok := parseIndexed(key)
		return ok
	}

	// Variable lookup
	if v, ok := vars[cond]; ok {
		return v != "" && v != "false" && v != "0"
	}
	return false
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
