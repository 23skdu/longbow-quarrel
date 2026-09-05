package tokenizer

import (
	"fmt"
	"math"
	"strings"

	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

func New(path string) (*Tokenizer, error) {
	f, err := gguf.LoadFile(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	return NewFromGGUF(f)
}

func NewFromGGUF(f *gguf.GGUFFile) (*Tokenizer, error) {

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
					ranks[ms] = i
				}
			}
		}
	}

	eosMap := make(map[int]bool)
	// 1. Check tokenizer.ggml.eos_token_id in metadata
	if eosVal, ok := f.KV["tokenizer.ggml.eos_token_id"]; ok {
		switch v := eosVal.(type) {
		case uint32:
			eosMap[int(v)] = true
		case int32:
			eosMap[int(v)] = true
		case uint64:
			if v <= math.MaxInt {
				eosMap[int(v)] = true // #nosec G115 -- safe: bounded by math.MaxInt
			}
		case int64:
			if v >= 0 && v <= math.MaxInt {
				eosMap[int(v)] = true // #nosec G115 -- safe: bounded by math.MaxInt
			}
		case float64:
			if v >= 0 && v <= float64(math.MaxInt) {
				eosMap[int(v)] = true // #nosec G115 -- safe: bounded by math.MaxInt
			}
		case int:
			eosMap[v] = true
		}
	}

	// 2. Add common known architecture EOS tokens if present in vocab
	commonEOSTokens := []string{
		"<|im_end|>",     // Qwen 2, 2.5, 3.5
		"<|endoftext|>",  // Qwen, GPT
		"<end_of_turn>",  // Gemma
		"</s>",           // Llama 1/2, Mistral
		"<|eot_id|>",     // Llama 3
	}
	for _, tokStr := range commonEOSTokens {
		if id, exists := vocab[tokStr]; exists {
			eosMap[id] = true
		}
	}

	// 3. Fallback to 2 (Llama 2 default) if none found
	if len(eosMap) == 0 {
		eosMap[2] = true
	}

	return &Tokenizer{
		Tokens:     tokens,
		Vocab:      vocab,
		Merges:     merges,
		Ranks:      ranks,
		EOSTokenIDs: eosMap,
	}, nil
}

type Tokenizer struct {
	Tokens      []string
	Vocab       map[string]int
	Merges      []string
	Ranks       map[string]int // Pair "a b" -> Rank (Index)
	Scores      []float32      // optional
	EOSTokenIDs map[int]bool
}

// IsEOS returns true if the tokenID is recognized as an end-of-sequence token.
func (t *Tokenizer) IsEOS(tokenID int) bool {
	if t.EOSTokenIDs == nil {
		return tokenID == 2
	}
	return t.EOSTokenIDs[tokenID]
}

// GetEOSTokenIDs returns a slice of all recognized EOS token IDs.
func (t *Tokenizer) GetEOSTokenIDs() []int {
	ids := make([]int, 0, len(t.EOSTokenIDs))
	for id := range t.EOSTokenIDs {
		ids = append(ids, id)
	}
	return ids
}

// AddEOSTokenID marks an additional token ID as an EOS token.
func (t *Tokenizer) AddEOSTokenID(tokenID int) {
	if t.EOSTokenIDs == nil {
		t.EOSTokenIDs = make(map[int]bool)
	}
	t.EOSTokenIDs[tokenID] = true
}

func (t *Tokenizer) Encode(text string) []int {
	if len(text) == 0 {
		return nil
	}

	// 1. Basic pre-tokenization
	words := splitWords(text)

	var allIDs []int

	// Fallback to Greedy Max Match if no merges available (e.g. Unigram)
	useMaxMatch := len(t.Ranks) == 0

	for _, w := range words {
		// Convert to BPE-clean format (replace space with U+2581 for Llama/Mistral)
		// Note: GPT-2 uses Ġ, Llama uses  (U+2581)
		cleanW := strings.ReplaceAll(w, " ", "\u2581")

		if useMaxMatch {
			// Greedy Max Match
			remaining := cleanW
			for len(remaining) > 0 {
				bestMatch := ""
				bestLen := 0
				found := false

				// Try to find longest prefix that exists in vocab
				// Optimization: Search from full length down
				// Or simple loop?
				// Max token length?
				// Limit search window? No, naive is fine for now.
				for l := len(remaining); l > 0; l-- {
					sub := remaining[:l]
					if _, ok := t.Vocab[sub]; ok {
						bestMatch = sub
						bestLen = l
						found = true
						break
					}
				}

				if found {
					allIDs = append(allIDs, t.Vocab[bestMatch])
					remaining = remaining[bestLen:]
				} else {
					remaining = remaining[1:]
				}
			}
			continue
		}

		// Normal BPE (Merges)
		subwords := make([]string, 0, len(cleanW))
		for _, r := range cleanW {
			subwords = append(subwords, string(r))
		}

		// Iteratively merge
		for len(subwords) >= 2 {

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

func (t *Tokenizer) GetVocab() []string {
	return t.Tokens
}
