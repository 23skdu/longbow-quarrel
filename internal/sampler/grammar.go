package sampler

import (
	// "fmt" - removed unused
)

// Grammar masks tokens that violate structural state machines like JSON format.
type Grammar struct {
	Active      bool
	JSONState   *JSONState
	Vocab       interface{} // *VocabularyTrie
	Bitmask     []byte      // Pre-calculated bitmask for current state
	VocabSize   int
}

type VocabularyTrie struct {
	Root *TrieNode
}

type TrieNode struct {
	Children map[rune]*TrieNode
	TokenID  int // -1 if not a full token
}

type JSONState struct {
	Stack []rune
	Last  rune
}

// NewJSONGrammar initializes structural enforcing for pure JSON.
func NewJSONGrammar(vocab []string) *Grammar {
	trie := &VocabularyTrie{Root: &TrieNode{Children: make(map[rune]*TrieNode), TokenID: -1}}
	for i, token := range vocab {
		trie.insert(token, i)
	}

	return &Grammar{
		Active:    true,
		JSONState: &JSONState{},
		Vocab:     trie,
		VocabSize: len(vocab),
		Bitmask:   make([]byte, (len(vocab)+7)/8),
	}
}

func (t *VocabularyTrie) insert(token string, id int) {
	node := t.Root
	for _, r := range token {
		if _, ok := node.Children[r]; !ok {
			node.Children[r] = &TrieNode{Children: make(map[rune]*TrieNode), TokenID: -1}
		}
		node = node.Children[r]
	}
	node.TokenID = id
}

// Apply restricts logits *before* Softmax conversion.
func (g *Grammar) Apply(logits []float32) error {
	if !g.Active {
		return nil
	}
	
	// Fast-path: use pre-calculated bitmask
	for i := 0; i < g.VocabSize; i++ {
		byteIdx := i / 8
		bitIdx := uint(i % 8)
		if (g.Bitmask[byteIdx] & (1 << bitIdx)) == 0 {
			logits[i] = -1e9
		}
	}

	return nil
}


func (g *Grammar) Update(token string) {
	if g.JSONState == nil {
		g.JSONState = &JSONState{}
	}
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
