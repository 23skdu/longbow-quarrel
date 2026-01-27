//go:build darwin && metal && integration

package integration

import (
	"reflect"
	"testing"

	"github.com/23skdu/longbow-quarrel/internal/device"
	"github.com/23skdu/longbow-quarrel/internal/engine"
)

// TestSlidingWindowCoherency verifies that SlidingWindowKVCache produces
// identical results to TensorKVCache when operating within the window.
func TestSlidingWindowCoherency(t *testing.T) {
	ctx := device.NewContext()
	defer ctx.Free()

	// Configuration
	seqLen := 32
	dims := 4
	kvHeads := 1
	config := engine.LlamaConfig{
		Layers:     1,
		WindowSize: seqLen, // Match sequence length for comparison
		KVHeads:    kvHeads,
		HeadDim:    dims,
		SeqLen:     seqLen,
	}

	// 1. Initialize Standard Cache
	stdCache := &engine.TensorKVCache{}
	if err := stdCache.Init(ctx, config); err != nil {
		t.Fatalf("Standard Init failed: %v", err)
	}
	defer stdCache.Free()

	// 2. Initialize Sliding Window Cache
	swCache := &engine.SlidingWindowKVCache{}
	if err := swCache.Init(ctx, config); err != nil {
		t.Fatalf("SW Init failed: %v", err)
	}
	defer swCache.Free()

	// 3. Run parallel updates and verify state match
	kvDim := kvHeads * dims
	kIn := ctx.NewTensor(1, kvDim)
	vIn := ctx.NewTensor(1, kvDim)

	kData := make([]float32, kvDim)
	vData := make([]float32, kvDim)

	for i := 0; i < seqLen; i++ {
		// Generate random-ish data
		for j := 0; j < kvDim; j++ {
			kData[j] = float32(i*10 + j)
			vData[j] = float32(i*10 + j + 5)
		}
		kIn.LoadFrom(kData)
		vIn.LoadFrom(vData)

		// Update Standard
		if err := stdCache.Update(0, i, kIn, vIn); err != nil {
			t.Fatalf("Std Update %d failed: %v", i, err)
		}

		// Update SW
		if err := swCache.Update(0, i, kIn, vIn); err != nil {
			t.Fatalf("SW Update %d failed: %v", i, err)
		}

		// Verify Match
		// Note: We need ToHost() to verify.
		// Assuming Get() returns the full cache tensor.
		// For TensorKVCache it is [SeqLen, Dim].
		// For SlidingWindowKVCache it is [WindowSize, Dim].
		// Since WindowSize == SeqLen, they should be structurally identical.

		stdK, stdV := stdCache.Get(0)
		swK, swV := swCache.Get(0)

		if stdK == nil || swK == nil {
			t.Fatalf("Get returned nil tensors")
		}

		// Reading back entire tensor is expensive, checking periodically or last row
		if i == seqLen-1 {
			// Full check at end
			stdKHost := stdK.ToHost()
			swKHost := swK.ToHost()

			if !reflect.DeepEqual(stdKHost, swKHost) {
				t.Errorf("Mismatch in K Cache content at step %d", i)
			}

			stdVHost := stdV.ToHost()
			swVHost := swV.ToHost()

			if !reflect.DeepEqual(stdVHost, swVHost) {
				t.Errorf("Mismatch in V Cache content at step %d", i)
			}
		}
	}
}

// TestSlidingWindowWrap verifies that wrapping works without error
// and maintains valid internal state (basic check).
func TestSlidingWindowWrap(t *testing.T) {
	ctx := device.NewContext()
	defer ctx.Free()

	windowSize := 16
	config := engine.LlamaConfig{
		Layers:     1,
		WindowSize: windowSize,
		KVHeads:    1,
		HeadDim:    4,
		SeqLen:     100, // Larger than window
	}

	swCache := &engine.SlidingWindowKVCache{}
	if err := swCache.Init(ctx, config); err != nil {
		t.Fatalf("SW Init failed: %v", err)
	}
	defer swCache.Free()

	kvDim := 4
	kIn := ctx.NewTensor(1, kvDim)
	vIn := ctx.NewTensor(1, kvDim)

	// Run past window size
	for i := 0; i < windowSize*2; i++ {
		if err := swCache.Update(0, i, kIn, vIn); err != nil {
			t.Fatalf("Update %d failed: %v", i, err)
		}
	}

	// Check size is still windowSize
	k, _ := swCache.Get(0)
	if k.Rows() != windowSize {
		t.Errorf("Expected cache rows %d, got %d", windowSize, k.Rows())
	}
}
