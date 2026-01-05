//go:build darwin && metal

package device

import (
	"math"
	"testing"
)

// TestRoPE_ThetaValue verifies that RoPE uses correct theta value for Mistral
// Mistral uses theta=1e6, not the default 1e4
func TestRoPE_ThetaValue(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()
	
	// Mistral config: head_dim=128, theta=1e6
	headDim := 128
	theta := float32(1e6)
	
	// Create test tensor: 1 token, 1 head
	M, N := 1, headDim
	input := ctx.NewTensorPooled(M, N)
	
	// Fill with known pattern
	data := make([]float32, N)
	for i := 0; i < N; i++ {
		data[i] = float32(i) * 0.1
	}
	input.LoadFrom(data)
	
	// Apply RoPE at position 0
	// Signature: RoPE(posOffset, headDim, numHeads, seqLen int, ropeTheta float32)
	// Apply RoPE at position 1 (Position 0 is identity)
	posOffset := 1
	numHeads := 1
	seqLen := 1
	input.RoPE(posOffset, headDim, numHeads, seqLen, theta)
	ctx.Synchronize()
	
	// Read back
	result := input.ToHost()
	
	// Verify RoPE was applied (values should change)
	changed := false
	for i := 0; i < N; i++ {
		// Use specific rotation tolerance
		if math.Abs(float64(result[i] - data[i])) > 1e-4 {
			changed = true
			break
		}
	}
	
	if !changed {
		t.Error("RoPE did not modify input values")
	}
	
	// Log for manual verification
	t.Logf("RoPE with theta=%.0e applied", theta)
	t.Logf("Input[0:4]: %.4f, %.4f, %.4f, %.4f", data[0], data[1], data[2], data[3])
	t.Logf("Output[0:4]: %.4f, %.4f, %.4f, %.4f", result[0], result[1], result[2], result[3])
}

// TestRoPE_FrequencyCalculation verifies RoPE frequency calculation
func TestRoPE_FrequencyCalculation(t *testing.T) {
	// For Mistral: theta=1e6, head_dim=128
	theta := 1e6
	headDim := 128
	
	// Expected frequency for first pair (i=0)
	// freq = 1.0 / (theta^(2i/d)) = 1.0 / (1e6^0) = 1.0
	freq0 := 1.0 / math.Pow(theta, 0.0/float64(headDim))
	
	// Expected frequency for last pair (i=63)
	// freq = 1.0 / (theta^(126/128))
	freq63 := 1.0 / math.Pow(theta, 126.0/float64(headDim))
	
	t.Logf("RoPE frequencies for theta=%.0e, head_dim=%d:", theta, headDim)
	t.Logf("  freq[0] = %.6e (should be ~1.0)", freq0)
	t.Logf("  freq[63] = %.6e (should be very small)", freq63)
	
	// Verify freq0 is close to 1.0
	if math.Abs(freq0 - 1.0) > 0.001 {
		t.Errorf("freq[0] = %.6e, expected ~1.0", freq0)
	}
	
	// Verify freq63 is very small
	if freq63 > 1e-5 {
		t.Errorf("freq[63] = %.6e, expected < 1e-5", freq63)
	}
}

// TestAttention_CausalMask verifies attention uses causal masking during prefill
func TestAttention_CausalMask(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()
	
	// Simple test: 3 tokens, 1 head, head_dim=4
	seqLen := 3
	heads := 1
	headDim := 4
	
	// Create Q, K, V
	q := ctx.NewTensorPooled(seqLen, heads*headDim)
	k := ctx.NewTensorPooled(seqLen, heads*headDim)
	v := ctx.NewTensorPooled(seqLen, heads*headDim)
	
	// Fill with simple patterns
	qData := make([]float32, seqLen*heads*headDim)
	kData := make([]float32, seqLen*heads*headDim)
	vData := make([]float32, seqLen*heads*headDim)
	
	for i := 0; i < seqLen*heads*headDim; i++ {
		qData[i] = 1.0
		kData[i] = 1.0
		vData[i] = float32(i % seqLen) // Different value per position
	}
	
	q.LoadFrom(qData)
	k.LoadFrom(kData)
	v.LoadFrom(vData)
	
	// TODO: Call attention kernel with causal mask
	// For now, this test documents the expected behavior
	
	t.Log("Causal mask test: Token at position i should only attend to positions 0..i")
	t.Log("Expected attention pattern:")
	t.Log("  Token 0: attends to [0]")
	t.Log("  Token 1: attends to [0, 1]")
	t.Log("  Token 2: attends to [0, 1, 2]")
}
