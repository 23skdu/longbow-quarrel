package engine

import (
	"sync"
)

// RadixNode represents a node in the prompt cache RadixTree.
type RadixNode struct {
	Tokens []int
	// Reference to physical blocks in the PagedKVCache holding this prefix
	PhysicalBlocks []int32
	Children       map[int]*RadixNode
	RefCount       int
}

// PromptCache manages a Radix tree of shared prefix sequences to implement Time-To-First-Token optimizations.
type PromptCache struct {
	mu   sync.RWMutex
	root *RadixNode
}

// NewPromptCache creates a globally shared prefix cache.
func NewPromptCache() *PromptCache {
	return &PromptCache{
		root: &RadixNode{
			Tokens:   []int{},
			Children: make(map[int]*RadixNode),
		},
	}
}

// MatchPrefix finds the longest matching cached prompt prefix for the incoming token sequence.
func (pc *PromptCache) MatchPrefix(prompt []int) (matchedTokens int, cachedBlocks []int32) {
	pc.mu.RLock()
	defer pc.mu.RUnlock()

	curr := pc.root
	matchedTokens = 0

	for matchedTokens < len(prompt) {
		token := prompt[matchedTokens]
		if child, ok := curr.Children[token]; ok {
			curr = child
			matchedTokens++
			// In a true Radix, we'd check if child.Tokens matches a sub-slice
			// For this implementation, we assume children are single-token keyed
		} else {
			break
		}
	}

	if curr != pc.root {
		cachedBlocks = make([]int32, len(curr.PhysicalBlocks))
		copy(cachedBlocks, curr.PhysicalBlocks)
	}

	return matchedTokens, cachedBlocks
}

// Insert caches a computed prefix structure, pinning the physical PagedKVCache blocks.
func (pc *PromptCache) Insert(prompt []int, blocks []int32) {
	pc.mu.Lock()
	defer pc.mu.Unlock()

	curr := pc.root
	for _, token := range prompt {
		if child, ok := curr.Children[token]; ok {
			curr = child
		} else {
			newNode := &RadixNode{
				Tokens:   []int{token},
				Children: make(map[int]*RadixNode),
			}
			curr.Children[token] = newNode
			curr = newNode
		}
	}
	curr.PhysicalBlocks = make([]int32, len(blocks))
	copy(curr.PhysicalBlocks, blocks)
	curr.RefCount++
}
