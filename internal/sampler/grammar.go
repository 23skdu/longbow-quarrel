package sampler

import (
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/metrics"
)

type GrammarType int

const (
	GrammarTypeNone GrammarType = iota
	GrammarTypeJSON
	GrammarTypeRegex
	GrammarTypeCFG
)

// Grammar masks tokens that violate structural state machines like JSON format, CFG, or regex.
type Grammar struct {
	mu        sync.Mutex
	Active    bool
	Type      GrammarType
	JSONState *JSONState
	Regex     *regexp.Regexp
	RegexText string
	Current   string // Accumulator of generated text
	Vocab     []string
	VocabSize int
	Bitmask   []byte // Pre-calculated bitmask for current state
}

type VocabularyTrie struct {
	Root *TrieNode
}

type TrieNode struct {
	Children map[rune]*TrieNode
	TokenID  int // -1 if not a full token
}

type JSONExpectation int

const (
	ExpectAnyValue JSONExpectation = iota
	ExpectObjectKeyOrEnd
	ExpectColon
	ExpectObjectValue
	ExpectObjectCommaOrEnd
	ExpectArrayValueOrEnd
	ExpectArrayCommaOrEnd
)

type JSONState struct {
	Stack       []rune
	Expectation JSONExpectation
	InString    bool
	Escaped     bool
	Last        rune
}

// NewJSONGrammar initializes structural enforcing for pure JSON with pushdown automaton.
func NewJSONGrammar(vocab []string) *Grammar {
	g := &Grammar{
		Active:    true,
		Type:      GrammarTypeJSON,
		JSONState: &JSONState{Expectation: ExpectAnyValue},
		Vocab:     vocab,
		VocabSize: len(vocab),
		Bitmask:   make([]byte, (len(vocab)+7)/8),
	}
	g.recomputeBitmask()
	return g
}

// NewRegexGrammar initializes grammar enforcing with regular expressions.
func NewRegexGrammar(pattern string, vocab []string) (*Grammar, error) {
	re, err := regexp.Compile(pattern)
	if err != nil {
		return nil, err
	}
	g := &Grammar{
		Active:    true,
		Type:      GrammarTypeRegex,
		Regex:     re,
		RegexText: pattern,
		Vocab:     vocab,
		VocabSize: len(vocab),
		Bitmask:   make([]byte, (len(vocab)+7)/8),
	}
	g.recomputeBitmask()
	return g, nil
}

// Apply restricts logits *before* Softmax conversion using precalculated bitmask.
func (g *Grammar) Apply(logits []float32) error {
	if g == nil || !g.Active {
		return nil
	}
	start := time.Now()
	defer func() {
		metrics.RecordGrammarFilter(time.Since(start))
	}()

	g.mu.Lock()
	defer g.mu.Unlock()

	limit := g.VocabSize
	if len(logits) < limit {
		limit = len(logits)
	}

	for i := 0; i < limit; i++ {
		byteIdx := i / 8
		bitIdx := uint(i % 8)
		if byteIdx < len(g.Bitmask) {
			if (g.Bitmask[byteIdx] & (1 << bitIdx)) == 0 {
				logits[i] = -1e9
			}
		}
	}

	return nil
}

// Update advances the grammar state given the newly generated token.
func (g *Grammar) Update(token string) {
	if g == nil {
		return
	}
	g.mu.Lock()
	defer g.mu.Unlock()

	g.Current += token

	if g.JSONState == nil {
		g.JSONState = &JSONState{Expectation: ExpectAnyValue}
	}
	for _, r := range token {
		g.updateJSONRune(r)
	}

	g.recomputeBitmask()
}

func (g *Grammar) updateJSONRune(r rune) {
	s := g.JSONState
	if s.InString {
		if s.Escaped {
			s.Escaped = false
		} else if r == '\\' {
			s.Escaped = true
		} else if r == '"' {
			s.InString = false
			if len(s.Stack) > 0 && s.Stack[len(s.Stack)-1] == '{' {
				if s.Expectation == ExpectObjectKeyOrEnd {
					s.Expectation = ExpectColon
				} else {
					s.Expectation = ExpectObjectCommaOrEnd
				}
			} else if len(s.Stack) > 0 && s.Stack[len(s.Stack)-1] == '[' {
				s.Expectation = ExpectArrayCommaOrEnd
			}
		}
		s.Last = r
		return
	}

	if r == ' ' || r == '\t' || r == '\n' || r == '\r' {
		return
	}

	switch r {
	case '{':
		s.Stack = append(s.Stack, '{')
		s.Expectation = ExpectObjectKeyOrEnd
	case '[':
		s.Stack = append(s.Stack, '[')
		s.Expectation = ExpectArrayValueOrEnd
	case '}':
		if len(s.Stack) > 0 && s.Stack[len(s.Stack)-1] == '{' {
			s.Stack = s.Stack[:len(s.Stack)-1]
		}
		if len(s.Stack) > 0 {
			if s.Stack[len(s.Stack)-1] == '{' {
				s.Expectation = ExpectObjectCommaOrEnd
			} else {
				s.Expectation = ExpectArrayCommaOrEnd
			}
		} else {
			s.Expectation = ExpectAnyValue
		}
	case ']':
		if len(s.Stack) > 0 && s.Stack[len(s.Stack)-1] == '[' {
			s.Stack = s.Stack[:len(s.Stack)-1]
		}
		if len(s.Stack) > 0 {
			if s.Stack[len(s.Stack)-1] == '{' {
				s.Expectation = ExpectObjectCommaOrEnd
			} else {
				s.Expectation = ExpectArrayCommaOrEnd
			}
		} else {
			s.Expectation = ExpectAnyValue
		}
	case ':':
		s.Expectation = ExpectObjectValue
	case ',':
		if len(s.Stack) > 0 && s.Stack[len(s.Stack)-1] == '{' {
			s.Expectation = ExpectObjectKeyOrEnd
		} else if len(s.Stack) > 0 && s.Stack[len(s.Stack)-1] == '[' {
			s.Expectation = ExpectArrayValueOrEnd
		}
	case '"':
		s.InString = true
		s.Escaped = false
	default:
		// Literal value (digit, true, false, null)
		if len(s.Stack) > 0 && s.Stack[len(s.Stack)-1] == '{' {
			s.Expectation = ExpectObjectCommaOrEnd
		} else if len(s.Stack) > 0 && s.Stack[len(s.Stack)-1] == '[' {
			s.Expectation = ExpectArrayCommaOrEnd
		}
	}
	s.Last = r
}

func (g *Grammar) isTokenAllowedJSON(token string) bool {
	if token == "" {
		return true
	}
	s := g.JSONState
	if s == nil {
		return true
	}

	trimmed := strings.TrimSpace(token)
	if trimmed == "" {
		return true // Whitespace is generally allowed
	}

	firstRune := rune(trimmed[0])

	if s.InString {
		return true // In string literal, all characters are accepted
	}

	if len(s.Stack) == 0 {
		// Outside any container: must start object or array
		return firstRune == '{' || firstRune == '['
	}

	switch s.Expectation {
	case ExpectAnyValue:
		return firstRune == '{' || firstRune == '[' || firstRune == '"' ||
			(firstRune >= '0' && firstRune <= '9') || firstRune == '-' ||
			firstRune == 't' || firstRune == 'f' || firstRune == 'n'
	case ExpectObjectKeyOrEnd:
		return firstRune == '"' || firstRune == '}'
	case ExpectColon:
		return firstRune == ':'
	case ExpectObjectValue:
		return firstRune == '{' || firstRune == '[' || firstRune == '"' ||
			(firstRune >= '0' && firstRune <= '9') || firstRune == '-' ||
			firstRune == 't' || firstRune == 'f' || firstRune == 'n'
	case ExpectObjectCommaOrEnd:
		return firstRune == ',' || firstRune == '}'
	case ExpectArrayValueOrEnd:
		return firstRune == '{' || firstRune == '[' || firstRune == '"' ||
			(firstRune >= '0' && firstRune <= '9') || firstRune == '-' ||
			firstRune == 't' || firstRune == 'f' || firstRune == 'n' || firstRune == ']'
	case ExpectArrayCommaOrEnd:
		return firstRune == ',' || firstRune == ']'
	}

	return true
}

func (g *Grammar) isTokenAllowedRegex(token string) bool {
	if g.Regex == nil {
		return true
	}
	candidate := g.Current + token
	// If candidate matches or matches prefix
	return g.Regex.MatchString(candidate) || len(candidate) < 128
}

func (g *Grammar) recomputeBitmask() {
	for i := range g.Bitmask {
		g.Bitmask[i] = 0
	}

	for i, token := range g.Vocab {
		allowed := false
		switch g.Type {
		case GrammarTypeJSON:
			allowed = g.isTokenAllowedJSON(token)
		case GrammarTypeRegex:
			allowed = g.isTokenAllowedRegex(token)
		default:
			allowed = true
		}

		if allowed {
			byteIdx := i / 8
			bitIdx := uint(i % 8)
			if byteIdx < len(g.Bitmask) {
				g.Bitmask[byteIdx] |= (1 << bitIdx)
			}
		}
	}
}
