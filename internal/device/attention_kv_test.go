package device

import (
	"math"
	"testing"
)

func TestAttentionKV_Retrieval_Mistral(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	// Mistral Params
	dim := 128     // Head Dim
	kvHeads := 1   // Simplify to 1 KV head
	heads := 1     // 1 Q head
	seqLen := 4    // Small sequence
	
	// Create Tensors
	q := ctx.NewTensorFP32(1, dim)  // Query (1 token)
	kCache := ctx.NewTensor(seqLen, kvHeads*dim) // KV Cache (F16)
	vCache := ctx.NewTensor(seqLen, kvHeads*dim) // KV Cache (F16)
	out := ctx.NewTensor(1, dim)    // Output (F16)
	
	// Initialize Data
	// Query = [1, 0, 0, ...]. Matches with Key [1, 0, ...] perfectly.
	qData := make([]float32, dim)
	qData[0] = 1.0
	q.LoadFrom(qData)
	
	// Keys: Position 2 should match perfectly. Others orthogonal.
	kData := make([]float32, seqLen * dim)
	// Pos 0: [0, 1, 0...] (Orthogonal)
	kData[0*dim + 1] = 1.0
	// Pos 1: [0, 0, 1...] (Orthogonal)
	kData[1*dim + 2] = 1.0
	// Pos 2: [1, 0, 0...] (Match! Dot product = 1.0 * 1.0 = 1.0)
	kData[2*dim + 0] = 1.0
	// Pos 3: [0, 0, 0...] (Zero)
	
	kCache.LoadFrom(kData)
	
	// Values: Distinct values to identify which position was attended
	vData := make([]float32, seqLen * dim)
	// Pos 0: [10, 10...]
	for i := 0; i < dim; i++ { vData[0*dim+i] = 10.0 }
	// Pos 1: [20, 20...]
	for i := 0; i < dim; i++ { vData[1*dim+i] = 20.0 }
	// Pos 2: [30, 30...] (Target)
	for i := 0; i < dim; i++ { vData[2*dim+i] = 30.0 }
	// Pos 3: [40, 40...]
	for i := 0; i < dim; i++ { vData[3*dim+i] = 40.0 }
	
	vCache.LoadFrom(vData)
	
	// Run Attention
	// Pos = 2 (We are at step 2, attending to 0, 1, 2)
	// Masking: Causal usually allows attending to <= pos.
	// We need to call the Attention Kernel directly or via Layer wrapper?
	// metal.go doesn't expose `AttFused` directly easily without setup.
	// But `Metal_AttFused_F16` is exposed via CGO if we wrap it.
	// Or use `t.Layer` but that runs everything.
	
	// Run Attention Fused Kernel
	// Signature: AttFused(kCache, vCache, out, pos, numHeads, kvHeads, headDim)
	// We are at pos 2 (attending to 0,1,2).
	q.AttFused(kCache, vCache, out, 2, heads, kvHeads, dim)
	
	// Get Result
	outData := out.ToHost()
	
	// Verification
	// We attended to Pos 2 (match).
	// Softmax should be peaked at Pos 2 (since K[2] matches Q).
	// But Q=1, K[2]=1. Dot=1. Scale=1/sqrt(128) ~= 0.088.
	// K[0], K[1] are orthogonal (Dot=0).
	// Scores: Pos0=0, Pos1=0, Pos2=0.088.
	// Softmax([0, 0, 0.088]) -> exp(0)=1, exp(0.088)=1.09.
	// Sum = 1+1+1.09 = 3.09.
	// Prob0 = 1/3.09 = 0.32
	// Prob1 = 1/3.09 = 0.32
	// Prob2 = 1.09/3.09 = 0.35
	//
	// Result = 0.32*V[0] + 0.32*V[1] + 0.35*V[2]
	// V[0]=10, V[1]=20, V[2]=30
	// Exp = 3.2 + 6.4 + 10.5 = ~20.1
	//
	// Wait, standard Attention usually uses large scale inputs so Softmax is sharp.
	// With small inputs, Softmax is diffuse.
	// Let's verify we get ~20.1.
	
	val := float64(outData[0])
	expected := 20.1
	
	if math.Abs(val - expected) > 2.0 {
		t.Errorf("Attention Output Mismatch: got %f, expected diffuse avg ~%f", val, expected)
		// Dump some info
		t.Logf("Got %f", val)
	} else {
		t.Logf("Attention Retrieval SUCCESS: Got %f (Matches diffuse attention mixture)", val)
	}
}
