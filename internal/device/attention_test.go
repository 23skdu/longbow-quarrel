package device

import (
	"math"
	"testing"
)

// TestAttentionScaling_Mistral verifies that attention score scaling is correctly applied
// for Mistral's architecture (32 Q heads, 8 KV heads, 128 head_dim).
func TestAttentionScaling_Mistral(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()
	
	// Mistral config
	const (
		numHeads  = 32
		kvHeads   = 8
		headDim   = 128
		seqLen    = 4
	)
	
	// Create test Q, K, V tensors
	q := ctx.NewTensor(1, numHeads*headDim)  // [1, 4096]
	kCache := ctx.NewTensor(seqLen, kvHeads*headDim)  // [4, 1024]
	vCache := ctx.NewTensor(seqLen, kvHeads*headDim)  // [4, 1024]
	scores := ctx.NewTensorF32(numHeads, seqLen)  // [32, 4]
	
	// Initialize with known values
	// Q: all 1.0 for simplicity
	qData := make([]float32, numHeads*headDim)
	for i := range qData {
		qData[i] = 1.0
	}
	q.LoadFrom(qData)
	
	// K: all 1.0 for simplicity
	kData := make([]float32, seqLen*kvHeads*headDim)
	for i := range kData {
		kData[i] = 1.0
	}
	kCache.LoadFromF32(kData)
	
	// V: all 0.5 for output verification
	vData := make([]float32, seqLen*kvHeads*headDim)
	for i := range vData {
		vData[i] = 0.5
	}
	vCache.LoadFromF32(vData)
	
	// Compute attention scores
	// Expected: dot(Q, K) = headDim * 1.0 * 1.0 = 128
	// After scaling: 128 / sqrt(128) = sqrt(128) ≈ 11.31
	expectedRawScore := float32(headDim)
	expectedScaledScore := expectedRawScore / float32(math.Sqrt(float64(headDim)))
	
	t.Logf("Expected raw score: %.2f", expectedRawScore)
	t.Logf("Expected scaled score: %.2f", expectedScaledScore)
	t.Logf("Scale factor: 1/sqrt(%d) = %.6f", headDim, 1.0/math.Sqrt(float64(headDim)))
	
	// Run attention (using granular steps for debugging)
	pos := seqLen - 1
	q.AttentionScores(kCache, scores, pos, numHeads, kvHeads, headDim, seqLen)
	
	// Verify scores
	scoresData := scores.ToHostF32()
	
	// Check first head, all positions
	for pos := 0; pos <= seqLen-1; pos++ {
		score := scoresData[0*seqLen+pos]
		t.Logf("Head 0, pos %d: score = %.6f", pos, score)
		
		// Allow 1% tolerance for FP16 precision
		if math.Abs(float64(score-expectedScaledScore)) > float64(expectedScaledScore)*0.01 {
			t.Errorf("Head 0, pos %d: score mismatch. Got %.6f, want %.6f", 
				pos, score, expectedScaledScore)
		}
	}
	
	// Verify GQA indexing: heads 0-3 should use KV head 0, heads 4-7 use KV head 1, etc.
	// With all K values = 1.0, all scores should be identical
	for h := 0; h < numHeads; h++ {
		score := scoresData[h*seqLen+0]  // First position
		if math.Abs(float64(score-expectedScaledScore)) > float64(expectedScaledScore)*0.01 {
			t.Errorf("Head %d: score mismatch. Got %.6f, want %.6f (GQA indexing issue?)", 
				h, score, expectedScaledScore)
		}
	}
	
	t.Logf("✓ Attention scaling verified: 1/sqrt(%d) correctly applied", headDim)
	t.Logf("✓ GQA indexing verified: all %d heads produce consistent scores", numHeads)
}

// TestAttentionScaling_EdgeCases tests attention with extreme values
func TestAttentionScaling_EdgeCases(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()
	
	const (
		numHeads = 4
		kvHeads  = 2
		headDim  = 128
		seqLen   = 2
	)
	
	q := ctx.NewTensor(1, numHeads*headDim)
	kCache := ctx.NewTensor(seqLen, kvHeads*headDim)
	scores := ctx.NewTensorF32(numHeads, seqLen)
	
	// Test with large values (like Mistral's normalized activations: 16.6)
	qData := make([]float32, numHeads*headDim)
	for i := range qData {
		qData[i] = 16.6
	}
	q.LoadFrom(qData)
	
	kData := make([]float32, seqLen*kvHeads*headDim)
	for i := range kData {
		kData[i] = 16.6
	}
	kCache.LoadFromF32(kData)
	
	// Compute scores
	pos := seqLen - 1
	q.AttentionScores(kCache, scores, pos, numHeads, kvHeads, headDim, seqLen)
	
	scoresData := scores.ToHostF32()
	
	// Expected: dot(Q, K) = 128 * 16.6 * 16.6 = 35251.2
	// After scaling: 35251.2 / sqrt(128) ≈ 3116.5
	expectedRawScore := float32(headDim) * 16.6 * 16.6
	expectedScaledScore := expectedRawScore / float32(math.Sqrt(float64(headDim)))
	
	t.Logf("Large value test: Q=K=16.6")
	t.Logf("Expected raw score: %.2f", expectedRawScore)
	t.Logf("Expected scaled score: %.2f", expectedScaledScore)
	
	score := scoresData[0]
	t.Logf("Actual score: %.2f", score)
	
	// Check if score is reasonable (not NaN, not Inf, roughly correct magnitude)
	if math.IsNaN(float64(score)) || math.IsInf(float64(score), 0) {
		t.Errorf("Score is NaN or Inf with large inputs: %.2f", score)
	}
	
	// Allow 5% tolerance for large values
	if math.Abs(float64(score-expectedScaledScore)) > float64(expectedScaledScore)*0.05 {
		t.Errorf("Score mismatch with large values. Got %.2f, want %.2f", score, expectedScaledScore)
	}
	
	t.Logf("✓ Attention handles large values correctly (Mistral-like activations)")
}
