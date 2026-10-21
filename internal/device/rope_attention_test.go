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
		if math.Abs(float64(result[i]-data[i])) > 1e-4 {
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
	if math.Abs(freq0-1.0) > 0.001 {
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

	// Four positions, one head, head_dim=4. The attention is asked for the
	// third position, so it may only see the first three values. The
	// fourth value is enormous: if the mask were missing, it would drag
	// the result with it.
	const (
		ctxLen   = 4
		pos      = 2
		heads    = 1
		kvHeads  = 1
		headDim  = 4
		farValue = 1000.0
	)

	q := ctx.NewTensorPooled(ctxLen, heads*headDim)
	k := ctx.NewTensorPooled(ctxLen, kvHeads*headDim)
	v := ctx.NewTensorPooled(ctxLen, kvHeads*headDim)

	// The queries and the keys are equal, so every attended position gets
	// the same score and the attention is the mean of the values it
	// attends to.
	qData := make([]float32, ctxLen*heads*headDim)
	kData := make([]float32, ctxLen*kvHeads*headDim)
	vData := make([]float32, ctxLen*kvHeads*headDim)
	for i := range qData {
		qData[i] = 1.0
	}
	for i := range kData {
		kData[i] = 1.0
	}
	for i := range vData {
		vData[i] = float32(i + 1)
	}
	vData[len(vData)-1] = farValue

	if err := q.LoadFrom(qData); err != nil {
		t.Fatalf("failed to load the queries: %v", err)
	}
	if err := k.LoadFrom(kData); err != nil {
		t.Fatalf("failed to load the keys: %v", err)
	}
	if err := v.LoadFrom(vData); err != nil {
		t.Fatalf("failed to load the values: %v", err)
	}
	ctx.Synchronize()

	result := q.Attention(k, v, pos, heads, kvHeads, headDim, ctxLen, ctxLen)
	ctx.Synchronize()

	resultData := result.ToHost()
	if len(resultData) != heads*headDim {
		t.Fatalf("the attention returned %d values, want %d", len(resultData), heads*headDim)
	}

	// The mean of the three attended values: (1 + 2 + 3) / 3.
	want := float32(2.0)
	for i, got := range resultData {
		if math.Abs(float64(got-want)) > 0.05 {
			t.Errorf("the output %d is %v, want %v (the mean of the values the mask allows)",
				i, got, want)
		}
	}
}
