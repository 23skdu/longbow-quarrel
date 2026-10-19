package engine

import (
	"fmt"
	"testing"

	conf "github.com/23skdu/longbow-quarrel/internal/config"
	"github.com/23skdu/longbow-quarrel/internal/device"
)

func TestPagedKVCache_StreamingLLM_AttentionSinksAndWindow(t *testing.T) {
	ctx := device.NewContext()
	defer ctx.Free()

	cache := &PagedKVCache{}
	config := conf.Config{
		KVHeads:    2,
		HeadDim:    64,
		Layers:     1,
		WindowSize: 256, // 16 blocks of 16 tokens
	}

	err := cache.Init(ctx, config)
	if err != nil {
		t.Fatalf("Init failed: %v", err)
	}
	defer cache.Free()

	seqID := "stream-seq-42"
	sinkBlocks := 2   // 32 tokens
	windowBlocks := 4 // 64 tokens
	cache.SetStreamingLLM(seqID, sinkBlocks, windowBlocks)

	params, ok := cache.GetStreamingLLM(seqID)
	if !ok || !params.Enabled || params.SinkBlocks != sinkBlocks || params.WindowBlocks != windowBlocks {
		t.Fatalf("StreamingLLM params mismatch: %+v", params)
	}

	k := ctx.NewTensorFP32(1, 2*64)
	v := ctx.NewTensorFP32(1, 2*64)
	if err := k.LoadFrom(make([]float32, 2*64)); err != nil {
		t.Fatalf("failed to seed key tensor: %v", err)
	}
	if err := v.LoadFrom(make([]float32, 2*64)); err != nil {
		t.Fatalf("failed to seed value tensor: %v", err)
	}

	var initialSinkBlocks []int32

	// Stream 500 tokens (well beyond the 6-block limit of 96 tokens)
	totalTokens := 500
	for pos := 0; pos < totalTokens; pos++ {
		err := cache.Update(seqID, 0, pos, k, v)
		if err != nil {
			t.Fatalf("Update failed at pos %d: %v", pos, err)
		}

		blocks := cache.GetSequenceBlocks(seqID)
		maxExpected := sinkBlocks + windowBlocks
		if len(blocks) > maxExpected {
			t.Fatalf("At pos %d: block table length %d exceeds max %d", pos, len(blocks), maxExpected)
		}

		if pos == (sinkBlocks*cache.blockSize - 1) {
			// Record the sink blocks once filled
			initialSinkBlocks = make([]int32, sinkBlocks)
			copy(initialSinkBlocks, blocks[:sinkBlocks])
		}

		// If past the sink blocks, verify the sinks are NEVER evicted
		if len(initialSinkBlocks) == sinkBlocks {
			for b := 0; b < sinkBlocks; b++ {
				if blocks[b] != initialSinkBlocks[b] {
					t.Fatalf("At pos %d: Attention sink block %d changed from %d to %d",
						pos, b, initialSinkBlocks[b], blocks[b])
				}
			}
		}
	}

	// Verify physical positions mapping for sinks remain immutable
	posMap, err := cache.GetPhysicalPositions(seqID, 0, 16)
	if err != nil {
		t.Fatalf("GetPhysicalPositions failed for sink: %v", err)
	}
	expectedFirstPos := int(initialSinkBlocks[0]) * cache.blockSize
	if posMap[0] != expectedFirstPos {
		t.Fatalf("Sink token 0 physical position expected %d, got %d", expectedFirstPos, posMap[0])
	}

	// Verify free sequence cleans up correctly
	cache.FreeSequence(seqID)
	if _, ok := cache.GetStreamingLLM(seqID); ok {
		t.Fatalf("StreamingLLM params should be deleted after FreeSequence")
	}
}

func TestPagedKVCache_StreamingLLM_InfiniteStreamStability(t *testing.T) {
	ctx := device.NewContext()
	defer ctx.Free()

	cache := &PagedKVCache{}
	config := conf.Config{
		KVHeads:    1,
		HeadDim:    32,
		Layers:     1,
		WindowSize: 128, // 8 blocks of 16 tokens
	}

	if err := cache.Init(ctx, config); err != nil {
		t.Fatalf("Init failed: %v", err)
	}
	defer cache.Free()

	seqID := "infinite-stream-test"
	cache.SetStreamingLLM(seqID, 2, 3) // 2 sink blocks, 3 window blocks = 5 blocks total

	k := ctx.NewTensorFP32(1, 32)
	v := ctx.NewTensorFP32(1, 32)
	if err := k.LoadFrom(make([]float32, 32)); err != nil {
		t.Fatalf("failed to seed key tensor: %v", err)
	}
	if err := v.LoadFrom(make([]float32, 32)); err != nil {
		t.Fatalf("failed to seed value tensor: %v", err)
	}

	// Stream 2,000 consecutive tokens without OOM
	for pos := 0; pos < 2000; pos++ {
		if err := cache.Update(seqID, 0, pos, k, v); err != nil {
			t.Fatalf("Streaming token %d failed: %v", pos, err)
		}
	}

	blocks := cache.GetSequenceBlocks(seqID)
	if len(blocks) != 5 {
		t.Fatalf("Expected strictly 5 blocks in cache, got %d", len(blocks))
	}
	fmt.Printf("Successfully streamed 2,000 tokens with constant %d blocks\n", len(blocks))
}
