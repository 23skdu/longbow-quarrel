package engine

import (
	"reflect"
	"testing"

	conf "github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
)

func TestRadixTree_PrefixMatchAndEdgeSplit(t *testing.T) {
	rt := NewRadixTree()

	// Insert "You are a helpful AI assistant." -> tokens: [100, 200, 300, 400, 500]
	prompt1 := []int{100, 200, 300, 400, 500}
	blocks1 := []int32{10, 11, 12}
	node1 := rt.Insert(prompt1, blocks1)
	if node1 == nil {
		t.Fatal("Insert returned nil node")
	}

	// Exact match
	matched, blocks, deepest := rt.MatchPrefix(prompt1)
	if matched != 5 {
		t.Fatalf("Expected 5 matched tokens, got %d", matched)
	}
	if !reflect.DeepEqual(blocks, blocks1) {
		t.Fatalf("Expected blocks %v, got %v", blocks1, blocks)
	}
	if deepest != node1 {
		t.Fatal("Expected deepest node to equal node1")
	}

	// Insert branching prompt: "You are a helpful coding expert." -> tokens: [100, 200, 300, 700, 800]
	prompt2 := []int{100, 200, 300, 700, 800}
	blocks2 := []int32{10, 21, 22}
	node2 := rt.Insert(prompt2, blocks2)
	if node2 == nil {
		t.Fatal("Insert branch returned nil node")
	}

	// Query prompt2
	matched2, blocksOut2, _ := rt.MatchPrefix(prompt2)
	if matched2 != 5 {
		t.Fatalf("Expected 5 matched tokens for prompt2, got %d", matched2)
	}
	if len(blocksOut2) == 0 {
		t.Fatal("Expected cached blocks for prompt2")
	}

	// Query partial prefix [100, 200, 300, 999]
	matchedPart, _, _ := rt.MatchPrefix([]int{100, 200, 300, 999})
	if matchedPart != 3 {
		t.Fatalf("Expected 3 matched tokens for common prefix, got %d", matchedPart)
	}
}

func TestRadixTree_CoWAndEvictionUnderPressure(t *testing.T) {
	ctx := device.NewContext()
	defer ctx.Free()

	cache := &PagedKVCache{}
	config := conf.Config{
		KVHeads:    2,
		HeadDim:    64,
		Layers:     1,
		WindowSize: 256,
	}
	if err := cache.Init(ctx, config); err != nil {
		t.Fatalf("Init PagedKVCache failed: %v", err)
	}
	defer cache.Free()

	rt := NewRadixTree()

	// Allocate 2 blocks in cache for prefix
	phys1, err1 := cache.allocateBlock()
	phys2, err2 := cache.allocateBlock()
	if err1 != nil || err2 != nil {
		t.Fatalf("allocateBlock failed: %v, %v", err1, err2)
	}

	prompt := []int{1, 2, 3, 4, 5, 6}
	blocks := []int32{phys1, phys2}
	node := rt.Insert(prompt, blocks)

	// Attach sequence 1
	rt.AttachSequence(node, cache)
	if node.RefCount != 1 {
		t.Fatalf("Expected RefCount 1, got %d", node.RefCount)
	}
	if cache.blockRefs[phys1] < 2 {
		t.Fatalf("Expected blockRefs > 1 for CoW, got %d", cache.blockRefs[phys1])
	}

	// Evict under pressure while referenced -> should NOT evict
	freed := rt.EvictUnderPressure(0.90, cache)
	if freed != 0 {
		t.Fatalf("Expected 0 freed while referenced, got %d", freed)
	}

	// Detach sequence
	rt.DetachSequence(node, cache)
	if node.RefCount != 0 {
		t.Fatalf("Expected RefCount 0 after detach, got %d", node.RefCount)
	}

	// Evict under pressure when unreferenced -> should prune leaf and free blocks
	freed = rt.EvictUnderPressure(0.90, cache)
	if freed != 2 {
		t.Fatalf("Expected 2 freed blocks after unreferenced eviction, got %d", freed)
	}
}
