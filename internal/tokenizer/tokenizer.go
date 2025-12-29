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
	ranks := make(map[string]int)
	if mVal, ok := f.KV["tokenizer.ggml.merges"]; ok {
		if mArr, ok := mVal.([]interface{}); ok {
			for i, m := range mArr {
				if ms, ok := m.(string); ok {
					merges = append(merges, ms)
					// Rank is explicit index or priority? GGUF stores them in order (low to high priority usually, or just index).
					// Actually, lower index = higher priority (merge earlier)?
					// In BPE, we merge the pairs that appear earliest in the merges list.
					// So map pair -> index. Smallest index wins.
					ranks[ms] = i
				}
			}
		}
	}

	return &Tokenizer{
		Tokens: tokens,
		Vocab:  vocab,
		Merges: merges,
		Ranks:  ranks,
	}, nil
}

type Tokenizer struct {
	Tokens []string
	Vocab  map[string]int
	Merges []string
	Ranks  map[string]int // Pair "a b" -> Rank (Index)
	Scores []float32 // optional
}

func (t *Tokenizer) Encode(text string) []int {
	if len(text) == 0 {
		return nil
	}

	// 1. Basic pre-tokenization (Split by whitespace for now to be safe, or just full string?)
	// GPT-2 style: treats space as part of next token (Ġ).
	// "Hello World" -> "Hello", " World" -> "Hello", "ĠWorld"
	
	// Simple approach: Walk input, processing "words".
	// A "word" is a sequence of non-space chars, OR a sequence of space chars?
	// Actually, we can just replace spaces with Ġ and chars with bytes (if needed), then BPE.
	// But usually, BPE is applied to pre-tokenized words.
	
	// Let's implement a simple whitespace splitter that preserves the space attached to the next word.
	words := splitWords(text)
	
	var allIDs []int
	
	for _, w := range words {
		// Convert to BPE-clean format (replace space with Ġ)
		// Note: The first word in sentence usually doesn't have Ġ unless it had leading space.
		// "Hello World" -> ["Hello", " World"]
		cleanW := strings.ReplaceAll(w, " ", "Ġ")
		
		// Map characters to initial subwords
		// "ĠWorld" -> ["Ġ", "W", "o", "r", "l", "d"]
		// But care: Unicode? GGUF tokens are byte strings often.
		// SmolLM (GPT-2) uses byte-level BPE.
		// For verification, we assume ASCII/UTF-8 works with the Vocab's keys.
		
		subwords := make([]string, 0, len(cleanW))
		for _, r := range cleanW {
			subwords = append(subwords, string(r))
		}
		
		// Iteratively merge
		for {
			if len(subwords) < 2 {
				break
			}
			
			// Find best pair
			bestPairIdx := -1
			bestRank := -1 // Lower is better
			
			for i := 0; i < len(subwords)-1; i++ {
				pair := subwords[i] + " " + subwords[i+1]
				// Check rank
				if rank, ok := t.Ranks[pair]; ok {
					if bestRank == -1 || rank < bestRank {
						bestRank = rank
						bestPairIdx = i
					}
				}
			}
			
			if bestPairIdx == -1 {
				break // No more merges
			}
			
			// Merge best pair
			// subwords[bestPairIdx] += subwords[bestPairIdx+1]
			// Remove subwords[bestPairIdx+1]
			
			merged := subwords[bestPairIdx] + subwords[bestPairIdx+1]
			
			// Rebuild slice (inefficient but safe)
			newSub := make([]string, 0, len(subwords)-1)
			newSub = append(newSub, subwords[:bestPairIdx]...)
			newSub = append(newSub, merged)
			newSub = append(newSub, subwords[bestPairIdx+2:]...)
			subwords = newSub
		}
		
		// Map final subwords to IDs
		for _, s := range subwords {
			if id, ok := t.Vocab[s]; ok {
				allIDs = append(allIDs, id)
			} else {
				// Fallback or Unknown?
				// Try <unk> or byte fallback.
				// For now, if not found, we skip or use 0?
				// Let's assume the vocab covers all chars (byte level BPE).
				// If not found, debug print?
				// fmt.Printf("Unknown token: %s\n", s)
			}
		}
	}
	
	return allIDs
}

// splitWords splits text but keeps leading spaces attached to words.
// "Hello World" -> ["Hello", " World"]
func splitWords(text string) []string {
	if len(text) == 0 {
		return nil
	}
	res := []string{}
	start := 0
	for i := 1; i < len(text); i++ {
		// New word boundary heuristic
		if text[i] == ' ' && text[i-1] != ' ' {
			res = append(res, text[start:i])
			start = i
		} 
	}
	res = append(res, text[start:])
	return res
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
		// Ġ (U+0120) is used for space in BPE (GPT-2/RoBERTa)
		// Ċ (U+010A) is used for newline in BPE
		//  (U+2581) is used for space in SentencePiece (Llama/Mistral)
		token = strings.ReplaceAll(token, "Ġ", " ")
		token = strings.ReplaceAll(token, "Ċ", "\n")
		token = strings.ReplaceAll(token, "\u2581", " ")
		
		// Handle other common BPE markers
		token = strings.ReplaceAll(token, "ĉ", "\t")
		
		sb.WriteString(token)
	}
	return sb.String()
}
