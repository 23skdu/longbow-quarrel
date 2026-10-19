package engine

import (
	"sync"
	"time"
)

// RadixTreeNode represents a node in a compressed Radix Tree prefix cache.
type RadixTreeNode struct {
	Tokens         []int            // Contiguous tokens along this edge
	PhysicalBlocks []int32          // Immutable physical blocks holding this token span
	Children       []*RadixTreeNode // Child edges
	Parent         *RadixTreeNode   // Parent pointer for bottom-up pruning
	RefCount       int              // Active sequences currently referencing this node
	LastAccess     time.Time        // LRU timestamp
}

// RadixTree implements high-performance prefix caching with token-span compression,
// Copy-on-Write block sharing, and LRU pruning under memory pressure.
type RadixTree struct {
	mu   sync.RWMutex
	root *RadixTreeNode
}

// NewRadixTree instantiates an empty radix prefix cache.
func NewRadixTree() *RadixTree {
	return &RadixTree{
		root: &RadixTreeNode{
			Tokens:     []int{},
			Children:   make([]*RadixTreeNode, 0),
			LastAccess: time.Now(),
		},
	}
}

// MatchPrefix finds the longest matching token prefix in the Radix Tree.
// Returns the count of matched tokens, accumulated physical block IDs, and the deepest matching node.
func (rt *RadixTree) MatchPrefix(tokens []int) (int, []int32, *RadixTreeNode) {
	rt.mu.RLock()
	defer rt.mu.RUnlock()

	curr := rt.root
	tokenOffset := 0
	matchedBlocks := make([]int32, 0)

	for tokenOffset < len(tokens) {
		matchedChild := false
		for _, child := range curr.Children {
			if len(child.Tokens) == 0 {
				continue
			}

			// Check common prefix between child.Tokens and tokens[tokenOffset:]
			common := 0
			minLen := len(child.Tokens)
			rem := len(tokens) - tokenOffset
			if rem < minLen {
				minLen = rem
			}

			for common < minLen && child.Tokens[common] == tokens[tokenOffset+common] {
				common++
			}

			if common > 0 {
				matchedChild = true
				if common == len(child.Tokens) {
					// Full match along this edge: advance and accumulate blocks
					matchedBlocks = append(matchedBlocks, child.PhysicalBlocks...)
					tokenOffset += common
					curr = child
					curr.LastAccess = time.Now()
					break
				} else {
					// Partial match along edge: cannot consume full child edge
					tokenOffset += common
					return tokenOffset, matchedBlocks, curr
				}
			}
		}

		if !matchedChild {
			break
		}
	}

	return tokenOffset, matchedBlocks, curr
}

// Insert inserts a token sequence and associated physical KV blocks into the radix tree.
// Edge splitting is applied when a new sequence branches off an existing edge.
func (rt *RadixTree) Insert(tokens []int, blocks []int32) *RadixTreeNode {
	rt.mu.Lock()
	defer rt.mu.Unlock()

	if len(tokens) == 0 {
		return rt.root
	}

	curr := rt.root
	tokenOffset := 0
	blockOffset := 0

	for tokenOffset < len(tokens) {
		var matchedChild *RadixTreeNode
		common := 0

		for _, child := range curr.Children {
			if len(child.Tokens) == 0 {
				continue
			}
			minLen := len(child.Tokens)
			rem := len(tokens) - tokenOffset
			if rem < minLen {
				minLen = rem
			}

			c := 0
			for c < minLen && child.Tokens[c] == tokens[tokenOffset+c] {
				c++
			}

			if c > 0 {
				matchedChild = child
				common = c
				break
			}
		}

		if matchedChild == nil {
			// No branch exists: create a new leaf child with the remainder
			newNode := &RadixTreeNode{
				Tokens:         append([]int{}, tokens[tokenOffset:]...),
				PhysicalBlocks: append([]int32{}, blocks[blockOffset:]...),
				Children:       make([]*RadixTreeNode, 0),
				Parent:         curr,
				LastAccess:     time.Now(),
			}
			curr.Children = append(curr.Children, newNode)
			return newNode
		}

		if common < len(matchedChild.Tokens) {
			// Split the matched child edge:
			// 1. Create a split node holding the common prefix
			splitTokens := append([]int{}, matchedChild.Tokens[:common]...)
			// Apportion physical blocks proportionally
			splitBlocksCount := (common * len(matchedChild.PhysicalBlocks)) / len(matchedChild.Tokens)
			splitBlocks := append([]int32{}, matchedChild.PhysicalBlocks[:splitBlocksCount]...)

			splitNode := &RadixTreeNode{
				Tokens:         splitTokens,
				PhysicalBlocks: splitBlocks,
				Children:       make([]*RadixTreeNode, 0),
				Parent:         curr,
				RefCount:       matchedChild.RefCount,
				LastAccess:     time.Now(),
			}

			// 2. Shorten matched child to the remaining suffix
			matchedChild.Tokens = append([]int{}, matchedChild.Tokens[common:]...)
			matchedChild.PhysicalBlocks = append([]int32{}, matchedChild.PhysicalBlocks[splitBlocksCount:]...)
			matchedChild.Parent = splitNode
			splitNode.Children = append(splitNode.Children, matchedChild)

			// 3. Replace matchedChild in curr.Children with splitNode
			for i, ch := range curr.Children {
				if ch == matchedChild {
					curr.Children[i] = splitNode
					break
				}
			}

			tokenOffset += common
			blockOffset += splitBlocksCount
			curr = splitNode

			if tokenOffset < len(tokens) {
				// Create the new sibling leaf for remaining tokens
				newLeaf := &RadixTreeNode{
					Tokens:         append([]int{}, tokens[tokenOffset:]...),
					PhysicalBlocks: append([]int32{}, blocks[blockOffset:]...),
					Children:       make([]*RadixTreeNode, 0),
					Parent:         splitNode,
					LastAccess:     time.Now(),
				}
				splitNode.Children = append(splitNode.Children, newLeaf)
				return newLeaf
			}
			return splitNode
		}

		// Full match along this edge: advance down
		tokenOffset += common
		blockOffset += len(matchedChild.PhysicalBlocks)
		if blockOffset > len(blocks) {
			blockOffset = len(blocks)
		}
		curr = matchedChild
		curr.LastAccess = time.Now()
	}

	return curr
}

// AttachSequence increments reference counts on the node and its physical blocks.
func (rt *RadixTree) AttachSequence(node *RadixTreeNode, kvCache *PagedKVCache) {
	if node == nil {
		return
	}
	rt.mu.Lock()
	defer rt.mu.Unlock()

	curr := node
	for curr != nil && curr != rt.root {
		curr.RefCount++
		curr.LastAccess = time.Now()
		if kvCache != nil {
			for _, b := range curr.PhysicalBlocks {
				kvCache.mu.Lock()
				kvCache.blockRefs[b]++
				kvCache.mu.Unlock()
			}
		}
		curr = curr.Parent
	}
}

// DetachSequence decrements reference counts when a sequence finishes.
func (rt *RadixTree) DetachSequence(node *RadixTreeNode, kvCache *PagedKVCache) {
	if node == nil {
		return
	}
	rt.mu.Lock()
	defer rt.mu.Unlock()

	curr := node
	for curr != nil && curr != rt.root {
		if curr.RefCount > 0 {
			curr.RefCount--
		}
		curr = curr.Parent
	}
}

// EvictUnderPressure prunes unreferenced leaves from the radix tree when memory pressure is high.
// Returns the number of physical blocks freed back to PagedKVCache.
func (rt *RadixTree) EvictUnderPressure(pressure float64, kvCache *PagedKVCache) int {
	if pressure < 0.85 {
		return 0
	}

	rt.mu.Lock()
	defer rt.mu.Unlock()

	freed := 0
	var pruneLeaves func(node *RadixTreeNode)
	pruneLeaves = func(node *RadixTreeNode) {
		if node == nil {
			return
		}

		// Recurse to children first (bottom-up leaf pruning)
		remainingChildren := make([]*RadixTreeNode, 0, len(node.Children))
		for _, child := range node.Children {
			pruneLeaves(child)
			// A child is eligible for eviction if it is an unreferenced leaf
			if len(child.Children) == 0 && child.RefCount == 0 {
				if kvCache != nil {
					for _, b := range child.PhysicalBlocks {
						kvCache.FreeBlock(b)
						freed++
					}
				}
				child.PhysicalBlocks = nil
				child.Parent = nil
			} else {
				remainingChildren = append(remainingChildren, child)
			}
		}
		node.Children = remainingChildren
	}

	pruneLeaves(rt.root)
	return freed
}
