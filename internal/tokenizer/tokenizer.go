package tokenizer

import (
	"fmt"
	"strings"

	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

type Tokenizer struct {
	Tokens []string
	Vocab  map[string]int
	Scores []float32 // optional
}

func New(path string) (*Tokenizer, error) {
	f, err := gguf.LoadFile(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	// Extract tokens
	val, ok := f.KV["tokenizer.ggml.tokens"]
	if !ok {
		return nil, fmt.Errorf("tokenizer.ggml.tokens not found in GGUF")
	}

	// GGUF array is []interface{}
	arr, ok := val.([]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid type for tokenizer.ggml.tokens")
	}

	tokens := make([]string, len(arr))
	vocab := make(map[string]int, len(arr))
	
	for i, v := range arr {
		s, ok := v.(string)
		if !ok {
			return nil, fmt.Errorf("token %d is not a string", i)
		}
		tokens[i] = s
		vocab[s] = i
	}

	return &Tokenizer{
		Tokens: tokens,
		Vocab:  vocab,
	}, nil
}

func (t *Tokenizer) Encode(text string) []int {
	// Very naive whitespace tokenizer for verification
	words := strings.Fields(text)
	var ids []int
	
	for i, w := range words {
		// Try to find token
		// Llama 3 often uses " word" (space prefix) for middle words
		token := w
		if i > 0 {
			token = "Ġ" + w // GGUF usually uses \u0120 or similar for space, or just space?
			// GGUF tokens are string. Llama 3 uses byte fallback?
			// Let's try literal space " " + w first.
			// But check if token exists.
			if _, ok := t.Vocab[" " + w]; ok {
				token = " " + w
			} else if _, ok := t.Vocab["Ġ" + w]; ok {
				token = "Ġ" + w // Common in BPE
			}
		}
		
		if id, ok := t.Vocab[token]; ok {
			ids = append(ids, id)
		} else {
			// Fallback: try original word if space version failed
			if id, ok := t.Vocab[w]; ok {
				ids = append(ids, id)
			} else {
				// Fallback to chars?
				// Skip for now, simpler verification
				fmt.Printf("Warning: Token not found for '%s'\n", w)
			}
		}
	}
	return ids
}

func (t *Tokenizer) Decode(ids []int) string {
	var sb strings.Builder
	for _, id := range ids {
		if id < 0 || id >= len(t.Tokens) {
			continue // or error?
		}
		// In a real BPE tokenizer we handle space replacement etc.
		// For Llama 3 / GGUF, tokens usually contain the piece.
		// e.g. " Hello" (with leading space char).
		// For our simple test, we just concat.
		sb.WriteString(t.Tokens[id])
	}
	return sb.String()
}
