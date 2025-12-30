package device

import (
	"math"
	"testing"
)

// TestKVCacheGQA_Mistral verifies KV cache storage and retrieval for Mistral's GQA architecture
func TestKVCacheGQA_Mistral(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()
	
	// Mistral config: 8 KV heads, 128 head_dim
	const (
		kvHeads = 8
		headDim = 128
		seqLen  = 4
	)
	
	kvDim := kvHeads * headDim  // 1024
	
	// Create K, V, and cache tensors
	k := ctx.NewTensor(1, kvDim)  // Current K projection
	v := ctx.NewTensor(1, kvDim)  // Current V projection
	kCache := ctx.NewTensor(seqLen, kvDim)  // K cache [4, 1024]
	vCache := ctx.NewTensor(seqLen, kvDim)  // V cache [4, 1024]
	
	// Initialize K and V with distinct values
	kData := make([]float32, kvDim)
	vData := make([]float32, kvDim)
	for i := range kData {
		kData[i] = float32(i % 100) / 10.0  // 0.0, 0.1, 0.2, ..., 9.9, 0.0, ...
		vData[i] = float32(i%100)/10.0 + 100.0  // 100.0, 100.1, ..., 109.9, 100.0, ...
	}
	k.LoadFrom(kData)
	v.LoadFrom(vData)
	
	// Store at position 2
	pos := 2
	k.StoreKV(v, kCache, vCache, pos, kvHeads, headDim)
	
	// Retrieve and verify
	kCacheData := kCache.ToHostF32()
	vCacheData := vCache.ToHostF32()
	
	// Check K cache at position 2
	offset := pos * kvDim
	tolerance := float32(0.1)  // Tolerance for FP16 precision (~0.05 typical error)
	for i := 0; i < kvDim; i++ {
		expected := kData[i]
		actual := kCacheData[offset+i]
		diff := actual - expected
		if diff < 0 {
			diff = -diff
		}
		if diff > tolerance {
			t.Errorf("K cache mismatch at pos=%d, idx=%d: got %.2f, want %.2f (diff=%.4f)", 
				pos, i, actual, expected, diff)
			if i > 5 {
				t.Fatalf("Too many errors, stopping")
			}
		}
	}
	
	// Check V cache at position 2
	for i := 0; i < kvDim; i++ {
		expected := vData[i]
		actual := vCacheData[offset+i]
		diff := actual - expected
		if diff < 0 {
			diff = -diff
		}
		if diff > tolerance {
			t.Errorf("V cache mismatch at pos=%d, idx=%d: got %.2f, want %.2f (diff=%.4f)", 
				pos, i, actual, expected, diff)
			if i > 5 {
				t.Fatalf("Too many errors, stopping")
			}
		}
	}
	
	// Verify other positions are zero (not overwritten)
	for p := 0; p < seqLen; p++ {
		if p == pos {
			continue
		}
		offset := p * kvDim
		for i := 0; i < 10; i++ {  // Check first 10 elements
			if kCacheData[offset+i] != 0 {
				t.Errorf("K cache position %d should be zero, got %.2f at idx %d", 
					p, kCacheData[offset+i], i)
				break
			}
			if vCacheData[offset+i] != 0 {
				t.Errorf("V cache position %d should be zero, got %.2f at idx %d", 
					p, vCacheData[offset+i], i)
				break
			}
		}
	}
	
	t.Logf("✓ KV cache storage verified: both K and V correctly stored at position %d", pos)
	t.Logf("✓ Other positions remain zero (no corruption)")
}

// TestKVCacheSequential tests sequential KV cache updates
func TestKVCacheSequential(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()
	
	const (
		kvHeads = 8
		headDim = 128
		seqLen  = 4
	)
	
	kvDim := kvHeads * headDim
	
	k := ctx.NewTensor(1, kvDim)
	v := ctx.NewTensor(1, kvDim)
	kCache := ctx.NewTensor(seqLen, kvDim)
	vCache := ctx.NewTensor(seqLen, kvDim)
	
	// Store different values at each position
	for pos := 0; pos < seqLen; pos++ {
		kData := make([]float32, kvDim)
		vData := make([]float32, kvDim)
		for i := range kData {
			kData[i] = float32(pos*10 + i)  // Unique but small
			vData[i] = float32(pos*10 + i + 100)
		}
		k.LoadFrom(kData)
		v.LoadFrom(vData)
		
		k.StoreKV(v, kCache, vCache, pos, kvHeads, headDim)
		ctx.Synchronize()
	}
	
	// Verify all positions
	kCacheData := kCache.ToHostF32()
	vCacheData := vCache.ToHostF32()
	
	for pos := 0; pos < seqLen; pos++ {
		offset := pos * kvDim
		// Check first and last element of each position
		expectedK0 := float32(pos * 10)
		expectedKLast := float32(pos*10 + kvDim - 1)
		expectedV0 := float32(pos*10 + 100)
		expectedVLast := float32(pos*10 + kvDim - 1 + 100)
		
		if math.Abs(float64(kCacheData[offset] - expectedK0)) > 0.1 {
			t.Errorf("Position %d: K[0] = %.0f, want %.0f", pos, kCacheData[offset], expectedK0)
		}
		if math.Abs(float64(kCacheData[offset+kvDim-1] - expectedKLast)) > 0.1 {
			t.Errorf("Position %d: K[last] = %.0f, want %.0f", pos, kCacheData[offset+kvDim-1], expectedKLast)
		}
		if math.Abs(float64(vCacheData[offset] - expectedV0)) > 0.1 {
			t.Errorf("Position %d: V[0] = %.0f, want %.0f", pos, vCacheData[offset], expectedV0)
		}
		if math.Abs(float64(vCacheData[offset+kvDim-1] - expectedVLast)) > 0.1 {
			t.Errorf("Position %d: V[last] = %.0f, want %.0f", pos, vCacheData[offset+kvDim-1], expectedVLast)
		}
	}
	
	t.Logf("✓ Sequential KV cache updates verified: all %d positions stored correctly", seqLen)
}
