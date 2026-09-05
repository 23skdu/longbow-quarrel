package engine

import (
	"sync"
	"time"
)

// RadixNode represents a node in the prompt cache RadixTree.
type RadixNode struct {
	Tokens []int
	// Reference to physical blocks in the PagedKVCache holding this prefix
	PhysicalBlocks []int32
	Children       map[int]*RadixNode
	RefCount       int
}

type lruNode struct {
	node     *RadixNode
	lastUsed time.Time
	prev     *lruNode
	next     *lruNode
}

// PromptCache manages a Radix tree of shared prefix sequences to implement Time-To-First-Token optimizations.
type PromptCache struct {
	mu   sync.RWMutex
	root *RadixNode

	maxCachedBlocks int
	lruHead         *lruNode
	lruTail         *lruNode
	lruSize         int
}

// NewPromptCache creates a globally shared prefix cache.
func NewPromptCache() *PromptCache {
	return &PromptCache{
		root: &RadixNode{
			Tokens:   []int{},
			Children: make(map[int]*RadixNode),
		},
		maxCachedBlocks: 256,
	}
}

// SetMaxCachedBlocks sets the maximum number of blocks to cache.
func (pc *PromptCache) SetMaxCachedBlocks(max int) {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	pc.maxCachedBlocks = max
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

	pc.moveToHead(curr)
}

func (pc *PromptCache) moveToHead(node *RadixNode) {
	lru := &lruNode{node: node, lastUsed: time.Now()}

	if pc.lruHead == nil {
		pc.lruHead = lru
		pc.lruTail = lru
		pc.lruSize = len(node.PhysicalBlocks)
		return
	}

	for existing := pc.lruHead; existing != nil; existing = existing.next {
		if existing.node == node {
			if existing == pc.lruHead {
				return
			}
			if existing.prev != nil {
				existing.prev.next = existing.next
			}
			if existing.next != nil {
				existing.next.prev = existing.prev
			}
			if existing == pc.lruTail {
				pc.lruTail = existing.prev
			}
			break
		}
	}

	lru.next = pc.lruHead
	pc.lruHead.prev = lru
	pc.lruHead = lru
	pc.lruSize += len(node.PhysicalBlocks)
}

func (pc *PromptCache) removeLRU() {
	if pc.lruTail == nil {
		return
	}

	toRemove := pc.lruTail
	pc.lruSize -= len(toRemove.node.PhysicalBlocks)

	if toRemove.prev != nil {
		toRemove.prev.next = nil
		pc.lruTail = toRemove.prev
	} else {
		pc.lruHead = nil
		pc.lruTail = nil
	}
}

// CurrentBlockCount returns the current number of cached physical blocks in LRU.
func (pc *PromptCache) CurrentBlockCount() int {
	pc.mu.RLock()
	defer pc.mu.RUnlock()
	return pc.lruSize
}

// Evict removes the least recently used cached prompts to free blocks back to the KV cache.
// Returns the number of blocks freed.
func (pc *PromptCache) Evict(kvCache *PagedKVCache) int {
	pc.mu.Lock()
	defer pc.mu.Unlock()

	freed := 0
	for pc.lruSize > pc.maxCachedBlocks && pc.lruTail != nil {
		node := pc.lruTail.node

		if len(node.PhysicalBlocks) > 0 && kvCache != nil {
			for _, block := range node.PhysicalBlocks {
				kvCache.FreeBlock(block)
				freed++
			}
		}

		node.PhysicalBlocks = nil
		node.RefCount = 0
		pc.removeLRU()
	}

	return freed
}
