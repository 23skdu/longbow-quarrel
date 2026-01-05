//go:build darwin && metal

package device

import (
	"math"
	"testing"
)

func TestAttention_Fused_Correctness(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	// Test case: 256 tokens, 8 heads, 128 headDim
	// This fits in the original 1024 s_mem but will test the new logic.
	pos := 255
	numHeads := 8
	kvHeads := 8
	headDim := 128

	q := ctx.NewTensor(1, numHeads*headDim)
	kCache := ctx.NewTensor(pos+1, kvHeads*headDim)
	vCache := ctx.NewTensor(pos+1, kvHeads*headDim)
	out := ctx.NewTensor(1, numHeads*headDim)

	// Init with ones to make it predictable
	qData := make([]float32, numHeads*headDim)
	for i := range qData {
		qData[i] = 0.1
	}
	q.LoadFrom(qData)

	kvData := make([]float32, (pos+1)*kvHeads*headDim)
	for i := range kvData {
		kvData[i] = 0.1
	}
	kCache.LoadFrom(kvData)
	vCache.LoadFrom(kvData)

	// Run fused attention
	q.AttFused(kCache, vCache, out, pos, numHeads, kvHeads, headDim, 0)
	ctx.Synchronize()

	res := out.ToHost()

	// Since all Q and K are 0.1, scores are sum(0.1*0.1) = headDim * 0.01 = 128 * 0.01 = 1.28
	// Scaling: 1.28 / sqrt(128) = 1.28 / 11.31 = 0.113
	// Softmax of identical values is 1/(pos+1) = 1/256
	// Output: sum(softmax * V) = sum(1/256 * 0.1) over 256 tokens = 0.1

	t.Logf("Result[0]: %f", res[0])
	if math.Abs(float64(res[0]-0.1)) > 0.01 {
		t.Errorf("Mismatch: expected ~0.1, got %f", res[0])
	}
}

func TestAttention_Fused_LongContext(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	// Test case: 1500 tokens. This EXCEEDS the old 1024 s_mem limit.
	pos := 1499
	numHeads := 1
	kvHeads := 1
	headDim := 128

	q := ctx.NewTensor(1, numHeads*headDim)
	kCache := ctx.NewTensor(pos+1, kvHeads*headDim)
	vCache := ctx.NewTensor(pos+1, kvHeads*headDim)
	out := ctx.NewTensor(1, numHeads*headDim)

	qData := make([]float32, numHeads*headDim)
	for i := range qData {
		qData[i] = 0.05
	}
	q.LoadFrom(qData)

	kvData := make([]float32, (pos+1)*kvHeads*headDim)
	for i := range kvData {
		kvData[i] = 0.05
	}
	kCache.LoadFrom(kvData)
	vCache.LoadFrom(kvData)

	q.AttFused(kCache, vCache, out, pos, numHeads, kvHeads, headDim, 0)
	ctx.Synchronize()

	res := out.ToHost()
	t.Logf("Long Context (1500) Result[0]: %f", res[0])

	if math.Abs(float64(res[0]-0.05)) > 0.01 {
		t.Errorf("Mismatch in long context: expected ~0.05, got %f", res[0])
	}
}
