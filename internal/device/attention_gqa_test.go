package device

import (
	"math"
	"testing"
)

// TestAttention_GQA verifies Grouped Query Attention logic
// specifically ensuring correct KV head mapping.
func TestAttention_GQA(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	// Config
	heads := 4
	kvHeads := 2 // Group Size = 2
	headDim := 64
	seqLen := 1
	
	// Create Tensors
	q := ctx.NewTensorFP32(1, heads * headDim) // 1 token, 4 heads
	kCache := ctx.NewTensor(seqLen, kvHeads * headDim) // 1 token, 2 KV heads
	vCache := ctx.NewTensor(seqLen, kvHeads * headDim)
	out := ctx.NewTensor(1, heads * headDim)
	
	// Initialize Data
	// Head 0 (Group 0): Matches KV 0
	// Head 1 (Group 0): Matches KV 0
	// Head 2 (Group 1): Matches KV 1
	// Head 3 (Group 1): Matches KV 1
	
	// Pattern:
	// KV 0: [1, 0, 0...]
	// KV 1: [0, 1, 0...]
	
	kData := make([]float32, seqLen * kvHeads * headDim)
	// KV 0 (Pos 0)
	kData[0*headDim + 0] = 1.0 
	// KV 1 (Pos 0)
	kData[1*headDim + 1] = 1.0
	
	kCache.LoadFrom(kData)
	
	// Values: 
	// KV 0 -> 10.0
	// KV 1 -> 20.0
	vData := make([]float32, seqLen * kvHeads * headDim)
	for i := 0; i < headDim; i++ {
		vData[0*headDim + i] = 10.0
		vData[1*headDim + i] = 20.0
	}
	vCache.LoadFrom(vData)
	
	// Queries
	qData := make([]float32, heads * headDim)
	
	// Head 0: [1, 0...] -> Matches KV 0. Expect V=10.
	qData[0*headDim + 0] = 1.0
	
	// Head 2: [0, 1...] -> Matches KV 1. Expect V=20.
	qData[2*headDim + 1] = 1.0
	
	// Head 1: [0, 1...] -> Matches KV 1? NO. Head 1 maps to KV 0.
	// KV 0 is [1, 0...]. Dot([0,1...], [1,0...]) = 0.
	// So Head 1 should have score 0 (if softmax handles 0 correctly vs -inf).
	// Actually with 1 token, score is unique. Softmax([0]) = 1.0.
	// So Head 1 will output V[KV 0] = 10.0.
	qData[1*headDim + 1] = 1.0
	
	q.LoadFrom(qData)
	
	// Run Attention (Pos 0)
	q.AttFused(kCache, vCache, out, 0, heads, kvHeads, headDim)
	
	res := out.ToHost()
	
	// Verify Head 0
	// Expected: 10.0
	valH0 := float64(res[0*headDim])
	if math.Abs(valH0 - 10.0) > 1.0 {
		t.Errorf("Head 0 (Map->KV0) Failed. Got %f, Expected 10.0", valH0)
	}
	
	// Verify Head 2
	// Expected: 20.0
	valH2 := float64(res[2*headDim])
	if math.Abs(valH2 - 20.0) > 1.0 {
		t.Errorf("Head 2 (Map->KV1) Failed. Got %f, Expected 20.0", valH2)
	}
	
	// Verify Head 1
	// Mapped to KV 0. Q=[0,1], K=[1,0]. Dot=0.
	// Softmax(0) = 1.0 (since only 1 pos).
	// Result = V[KV 0] = 10.0.
	// If it incorrectly mapped to KV 1 ([0,1]), Dot would be 1.0. V=20.0.
	// So if we get 20.0, mapping is WRONG.
	valH1 := float64(res[1*headDim])
	if math.Abs(valH1 - 10.0) > 1.0 {
		t.Errorf("Head 1 (Map->KV0) Failed. Got %f, Expected 10.0. (If 20, mapped to KV1)", valH1)
	} else {
		t.Logf("Head 1 PASSED: Got %f (Correctly mapped to KV0 despite pattern match with KV1)", valH1)
	}
}
