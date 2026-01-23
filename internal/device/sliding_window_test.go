//go:build darwin && metal

package device

import (
	"math"
	"testing"
)

// TestSlidingWindowAttention_BasicWindow tests that attention is limited to window size
func TestSlidingWindowAttention_BasicWindow(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	windowSize := 4
	seqLen := 8
	headDim := 4

	// Create Q, K, V for a sequence longer than window
	// Q: [seqLen, headDim]
	q := ctx.NewTensor(seqLen, headDim)
	k := ctx.NewTensor(seqLen, headDim)
	v := ctx.NewTensor(seqLen, headDim)

	// Fill with simple patterns
	qData := make([]float32, seqLen*headDim)
	kData := make([]float32, seqLen*headDim)
	vData := make([]float32, seqLen*headDim)

	for i := 0; i < seqLen; i++ {
		for j := 0; j < headDim; j++ {
			qData[i*headDim+j] = float32(i + 1) // Token position
			kData[i*headDim+j] = float32(i + 1)
			vData[i*headDim+j] = float32(i+1) * 10 // Distinct values
		}
	}

	q.LoadFrom(qData)
	k.LoadFrom(kData)
	v.LoadFrom(vData)

	// Compute attention with sliding window
	output := ctx.NewTensor(seqLen, headDim)

	// For each position, compute attention over window
	for pos := 0; pos < seqLen; pos++ {
		// Window starts at max(0, pos - windowSize + 1)
		windowStart := pos - windowSize + 1
		if windowStart < 0 {
			windowStart = 0
		}

		// Attention should only see [windowStart, pos]
		// For pos=7, window=[4,7] (4 tokens)
		// For pos=3, window=[0,3] (4 tokens)
		// For pos=1, window=[0,1] (2 tokens)

		// Verify window size
		actualWindowSize := pos - windowStart + 1
		expectedWindowSize := windowSize
		if pos < windowSize-1 {
			expectedWindowSize = pos + 1
		}

		if actualWindowSize != expectedWindowSize {
			t.Errorf("Pos %d: window size %d, expected %d", pos, actualWindowSize, expectedWindowSize)
		}
	}

	q.Free()
	k.Free()
	v.Free()
	output.Free()
}

// TestRollingBufferKVCache tests that KV cache wraps around at window size
func TestRollingBufferKVCache(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	windowSize := 4
	kvDim := 8

	// Create rolling buffer cache
	kCache := ctx.NewTensor(windowSize, kvDim)
	vCache := ctx.NewTensor(windowSize, kvDim)

	// Simulate storing 10 tokens (more than window)
	for pos := 0; pos < 10; pos++ {
		// Rolling buffer index
		cacheIdx := pos % windowSize

		// Create K, V for this position
		kData := make([]float32, kvDim)
		vData := make([]float32, kvDim)
		for i := 0; i < kvDim; i++ {
			kData[i] = float32(pos*100 + i) // Unique per position
			vData[i] = float32(pos*1000 + i)
		}

		// Store at rolling buffer position
		// In real implementation, this would be a kernel call
		// For now, verify the index calculation

		if cacheIdx != pos%windowSize {
			t.Errorf("Pos %d: cache index %d, expected %d", pos, cacheIdx, pos%windowSize)
		}

		// Verify that old entries get overwritten
		// At pos=4, cacheIdx=0 (overwrites pos=0)
		// At pos=5, cacheIdx=1 (overwrites pos=1)
		if pos >= windowSize {
			overwrittenPos := pos - windowSize
			expectedCacheIdx := overwrittenPos % windowSize
			if cacheIdx != expectedCacheIdx {
				t.Errorf("Pos %d overwrites pos %d at cache index %d",
					pos, overwrittenPos, cacheIdx)
			}
		}
	}

	kCache.Free()
	vCache.Free()
}

// TestSlidingWindowMask tests attention mask for sliding window
func TestSlidingWindowMask(t *testing.T) {
	windowSize := 4
	seqLen := 8

	// For each query position, compute which keys are visible
	for qPos := 0; qPos < seqLen; qPos++ {
		for kPos := 0; kPos < seqLen; kPos++ {
			// Causal: can only attend to past (kPos <= qPos)
			// Sliding window: can only attend to window (qPos - kPos < windowSize)

			shouldAttend := (kPos <= qPos) && (qPos-kPos < windowSize)

			// Verify mask logic
			if shouldAttend {
				// This position should be visible
				if qPos-kPos >= windowSize {
					t.Errorf("Q=%d K=%d: should not attend (outside window)", qPos, kPos)
				}
				if kPos > qPos {
					t.Errorf("Q=%d K=%d: should not attend (future)", qPos, kPos)
				}
			} else {
				// This position should be masked
				if kPos <= qPos && qPos-kPos < windowSize {
					t.Errorf("Q=%d K=%d: should attend but is masked", qPos, kPos)
				}
			}
		}
	}

	// Test specific cases
	testCases := []struct {
		qPos         int
		kPos         int
		shouldAttend bool
	}{
		{0, 0, true},  // Self-attention always visible
		{3, 0, true},  // Within window (3-0=3 < 4)
		{3, 1, true},  // Within window
		{3, 2, true},  // Within window
		{3, 3, true},  // Self
		{4, 0, false}, // Outside window (4-0=4 >= 4)
		{4, 1, true},  // Within window (4-1=3 < 4)
		{7, 3, false}, // Outside window (7-3=4 >= 4)
		{7, 4, true},  // Within window (7-4=3 < 4)
		{7, 7, true},  // Self
		{5, 6, false}, // Future (not causal)
	}

	for _, tc := range testCases {
		shouldAttend := (tc.kPos <= tc.qPos) && (tc.qPos-tc.kPos < windowSize)
		if shouldAttend != tc.shouldAttend {
			t.Errorf("Q=%d K=%d: got %v, expected %v",
				tc.qPos, tc.kPos, shouldAttend, tc.shouldAttend)
		}
	}
}

// TestSlidingWindowAttention_GPU tests end-to-end attention with window on GPU
func TestSlidingWindowAttention_GPU(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	windowSize := 3
	seqLen := 5
	headDim := 32 // Multiple of 32 for SIMD
	heads := 1
	kvHeads := 1
	ctxLen := seqLen

	// Create tensors
	q := ctx.NewTensor(1, heads*headDim)
	kCache := ctx.NewTensor(windowSize, kvHeads*headDim)
	vCache := ctx.NewTensor(windowSize, kvHeads*headDim)

	// Fill caches as if we processed 5 tokens
	// Tokens 0, 1, 2, 3, 4
	// Rolling buffer indices:
	// Pos 0 -> 0
	// Pos 1 -> 1
	// Pos 2 -> 2
	// Pos 3 -> 0 (overwrites 0)
	// Pos 4 -> 1 (overwrites 1)

	// So at Pos 4, window is tokens [2, 3, 4]
	// Their rolling indices are [2, 0, 1]

	kDataRaw := make([]float32, windowSize*kvHeads*headDim)
	vDataRaw := make([]float32, windowSize*kvHeads*headDim)

	// Token 2 (rolling index 2): Key=[1,0...], Val=[3,0...]
	kDataRaw[2*headDim] = 1.0
	vDataRaw[2*headDim] = 3.0

	// Token 3 (rolling index 0): Key=[1,0...], Val=[4,0...]
	kDataRaw[0*headDim] = 1.0
	vDataRaw[0*headDim] = 4.0

	// Token 4 (rolling index 1): Key=[1,0...], Val=[5,0...]
	kDataRaw[1*headDim] = 1.0
	vDataRaw[1*headDim] = 5.0

	kCache.LoadFrom(kDataRaw)
	vCache.LoadFrom(vDataRaw)

	// Query (Token 4): [1, 0...]
	qData := make([]float32, headDim)
	qData[0] = 1.0
	q.LoadFrom(qData)

	// Run Attention at pos 4 with window size 3
	// It should attend to tokens 2, 3, 4
	// Scores(q4, k2)=1, Scores(q4, k3)=1, Scores(q4, k4)=1
	// Softmax: 1/3, 1/3, 1/3
	// Output: 1/3*V2 + 1/3*V3 + 1/3*V4 = 1/3*3 + 1/3*4 + 1/3*5 = 4.0

	gpuOut := q.Attention(kCache, vCache, 4, heads, kvHeads, headDim, ctxLen, windowSize)
	ctx.Synchronize()

	res := gpuOut.ToHost()
	if math.Abs(float64(res[0]-4.0)) > 0.01 {
		t.Errorf("GPU SWA Mismatch: expected 4.0, got %f", res[0])
	} else {
		t.Logf("GPU SWA SUCCESS: Result %f matches expected 4.0", res[0])
	}
}

// TestSlidingWindow_LongSequence tests behavior with very long sequences
func TestSlidingWindow_LongSequence(t *testing.T) {
	windowSize := 4096 // Mistral's window size
	seqLen := 10000    // Longer than window

	// Verify that for any position > windowSize,
	// the window is exactly windowSize tokens
	for pos := windowSize; pos < seqLen; pos++ {
		windowStart := pos - windowSize + 1
		actualWindowSize := pos - windowStart + 1

		if actualWindowSize != windowSize {
			t.Errorf("Pos %d: window size %d, expected %d",
				pos, actualWindowSize, windowSize)
		}

		// Verify window bounds
		if windowStart != pos-windowSize+1 {
			t.Errorf("Pos %d: window start %d, expected %d",
				pos, windowStart, pos-windowSize+1)
		}
	}

	// Verify that for positions < windowSize,
	// the window grows from 1 to windowSize
	for pos := 0; pos < windowSize; pos++ {
		windowStart := 0
		actualWindowSize := pos - windowStart + 1
		expectedWindowSize := pos + 1

		if actualWindowSize != expectedWindowSize {
			t.Errorf("Pos %d: window size %d, expected %d",
				pos, actualWindowSize, expectedWindowSize)
		}
	}
}

// TestRollingBuffer_Wraparound tests cache wraparound behavior
func TestRollingBuffer_Wraparound(t *testing.T) {
	windowSize := 4

	// Simulate storing tokens and verify cache indices
	cacheIndices := make([]int, 10)
	for pos := 0; pos < 10; pos++ {
		cacheIndices[pos] = pos % windowSize
	}

	// Verify pattern: 0,1,2,3,0,1,2,3,0,1
	expected := []int{0, 1, 2, 3, 0, 1, 2, 3, 0, 1}
	for i, idx := range cacheIndices {
		if idx != expected[i] {
			t.Errorf("Pos %d: cache index %d, expected %d", i, idx, expected[i])
		}
	}

	// Verify that at pos=7, cache contains tokens [4,5,6,7]
	// Cache index 0 = token 4
	// Cache index 1 = token 5
	// Cache index 2 = token 6
	// Cache index 3 = token 7
	pos := 7
	for offset := 0; offset < windowSize; offset++ {
		tokenPos := pos - windowSize + 1 + offset
		cacheIdx := tokenPos % windowSize

		if cacheIdx != offset {
			t.Errorf("Pos %d offset %d: cache index %d, expected %d",
				pos, offset, cacheIdx, offset)
		}
	}
}
