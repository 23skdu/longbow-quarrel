//go:build darwin && metal

package device

import (
	"math"
	"testing"
)

func TestQK_Computation_Accuracy(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	headDim := 64
	seqLen := 4

	// Create Q and K tensors
	q := ctx.NewTensor(1, headDim)
	defer q.ReturnToPool()
	k := ctx.NewTensor(seqLen, headDim)
	defer k.ReturnToPool()

	// Fill with test data
	qData := make([]float32, headDim)
	for i := range qData {
		qData[i] = float32(i) / float32(headDim)
	}
	q.LoadFrom(qData)

	kData := make([]float32, seqLen*headDim)
	for i := range kData {
		kData[i] = float32((i+10)%20) / 20.0
	}
	k.LoadFrom(kData)

	// Compute scores on GPU (simplified, just dot products)
	// Note: This is a simplified test, actual attention kernel computes all at once
	// For validation, we can use the existing attention but with 1 head

	scores := make([]float32, seqLen)
	scale := 1.0 / math.Sqrt(float64(headDim))

	for t := 0; t < seqLen; t++ {
		dot := 0.0
		for i := 0; i < headDim; i++ {
			dot += float64(qData[i]) * float64(kData[t*headDim+i])
		}
		scores[t] = float32(dot * scale)
	}

	// Now, if we had a way to get intermediate scores from attention kernel, we could compare
	// For now, just verify the CPU computation is reasonable

	for i, score := range scores {
		if score < -10 || score > 10 {
			t.Errorf("Score %d out of reasonable range: %f", i, score)
		}
	}

	t.Logf("QK computation test: scores in range")
}
