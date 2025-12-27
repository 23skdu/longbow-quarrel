package tokenizer

import (
	"fmt"
	"strings"

	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

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

	// Extract merges
	var merges []string
	if mVal, ok := f.KV["tokenizer.ggml.merges"]; ok {
		if mArr, ok := mVal.([]interface{}); ok {
			for _, m := range mArr {
				if ms, ok := m.(string); ok {
					merges = append(merges, ms)
				}
			}
		}
	}

	return &Tokenizer{
		Tokens: tokens,
		Vocab:  vocab,
		Merges: merges,
	}, nil
}

type Tokenizer struct {
	Tokens []string
	Vocab  map[string]int
	Merges []string
	Scores []float32 // optional
}

func (t *Tokenizer) Encode(text string) []int {
	if len(text) == 0 {
		return nil
	}

	var ids []int
	remaining := text
	
	for len(remaining) > 0 {
		found := false
		// Try to find the longest prefix of 'remaining' that is in our vocab
		for l := len(remaining); l > 0; l-- {
			sub := remaining[:l]
			
			// Try as is
			if id, ok := t.Vocab[sub]; ok {
				ids = append(ids, id)
				remaining = remaining[l:]
				found = true
				break
			}
			
			// Try replacing leading space with Ġ (BPE space marker)
			if sub[0] == ' ' {
				gSub := "Ġ" + sub[1:]
				if id, ok := t.Vocab[gSub]; ok {
					ids = append(ids, id)
					remaining = remaining[l:]
					found = true
					break
				}
			}
		}
		
		if !found {
			// Skip 1 byte if no match found
			remaining = remaining[1:]
		}
	}
	
	return ids
}

func (t *Tokenizer) Decode(ids []int) string {
	var sb strings.Builder
	for _, id := range ids {
		if id < 0 || id >= len(t.Tokens) {
			continue // Skip invalid token IDs
		}
		
		token := t.Tokens[id]
		
		// Skip special tokens
		if strings.HasPrefix(token, "<|") && strings.HasSuffix(token, "|>") {
			continue
		}
		
		// Replace BPE special characters with actual characters
		// Ġ (U+0120) is used for space in BPE
		// Ċ (U+010A) is used for newline in BPE
		token = strings.ReplaceAll(token, "Ġ", " ")
		token = strings.ReplaceAll(token, "Ċ", "\n")
		
		// Handle other common BPE markers
		token = strings.ReplaceAll(token, "ĉ", "\t")
		
		sb.WriteString(token)
	}
	return sb.String()
}
