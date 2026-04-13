package sampler

import (
	"fmt"
)

// Grammar masks tokens that violate structural state machines like JSON format.
type Grammar struct {
	Active    bool
	JSONState *JSONState
	Vocab     []string
}

type JSONState struct {
	Stack []rune
	Last  rune
}

// NewJSONGrammar initializes structural enforcing for pure JSON.
func NewJSONGrammar(vocab []string) *Grammar {
	return &Grammar{
		Active:    true,
		JSONState: &JSONState{Stack: make([]rune, 0)},
		Vocab:     vocab,
	}
}

// Apply restricts logits *before* Softmax conversion.
func (g *Grammar) Apply(logits []float32) error {
	if !g.Active {
		return nil
	}
	if len(logits) == 0 {
		return fmt.Errorf("empty logits slice")
	}

	// We iterate through all tokens and check if they're allowed in the current JSON state.
	// In a production engine, this is optimized via a Trie or pre-filtered bitmask.
	for i, logit := range logits {
		if i >= len(g.Vocab) {
			break
		}
		
		token := g.Vocab[i]
		if !g.isValidInJSON(token) {
			logits[i] = -1e9
		}
	}

	return nil
}

func (g *Grammar) isValidInJSON(token string) bool {
	// Simple state-machine validation for common JSON tokens
	if len(g.JSONState.Stack) == 0 {
		return token == "{" || token == "["
	}
	
	last := g.JSONState.Stack[len(g.JSONState.Stack)-1]
	
	switch last {
	case '{':
		// Expecting key or end
		return token == "\"" || token == "}"
	case '[':
		// Expecting value or end
		return token == "{" || token == "[" || token == "\"" || token == "]" || (token >= "0" && token <= "9")
	}
	
	return true
}

func (g *Grammar) Update(token string) {
	for _, r := range token {
		switch r {
		case '{', '[':
			g.JSONState.Stack = append(g.JSONState.Stack, r)
		case '}', ']':
			if len(g.JSONState.Stack) > 0 {
				g.JSONState.Stack = g.JSONState.Stack[:len(g.JSONState.Stack)-1]
			}
		}
	}
}
