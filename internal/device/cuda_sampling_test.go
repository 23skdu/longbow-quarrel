//go:build linux && cuda

package device

import (
	"math/rand"
	"testing"
)

func TestCUDA_Sample_ArgmaxGreedy(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	vocabSize := 32000
	logitsData := make([]float32, vocabSize)
	for i := range logitsData {
		logitsData[i] = (rand.Float32() - 0.5) * 5.0
	}

	// Make token 1234 the definitive max
	targetToken := 1234
	logitsData[targetToken] = 100.0

	logitsTensor := ctx.NewTensorFP32(1, vocabSize)
	defer logitsTensor.Free()
	logitsTensor.LoadFrom(logitsData)

	sampled := ctx.SampleTopKTopP(logitsTensor, nil, 1.0, 0.0, 0.0, 0.0, 1, 0.9, 42, vocabSize)
	if sampled != targetToken {
		t.Fatalf("Greedy sampling mismatch: expected %d, got %d", targetToken, sampled)
	}
}

func TestCUDA_Sample_RepetitionPenalty(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	vocabSize := 1000
	logitsData := make([]float32, vocabSize)
	for i := range logitsData {
		logitsData[i] = 1.0
	}

	// Token 50 is slightly higher than token 60
	logitsData[50] = 10.0
	logitsData[60] = 9.0

	logitsTensor := ctx.NewTensorFP32(1, vocabSize)
	defer logitsTensor.Free()
	logitsTensor.LoadFrom(logitsData)

	// With token 50 in history and repetition penalty 2.0:
	// token 50 logit becomes 10.0 / 2.0 = 5.0
	// token 60 logit remains 9.0 -> token 60 should be selected!
	prev := []int32{50}
	sampled := ctx.SampleTopKTopP(logitsTensor, prev, 2.0, 0.0, 0.0, 0.1, 1, 0.9, 42, vocabSize)
	if sampled != 60 {
		t.Fatalf("Repetition penalty should suppress token 50 to select token 60, got %d", sampled)
	}
}

func BenchmarkCUDA_Sample_Vocab32k(b *testing.B) {
	ctx := NewContext()
	defer ctx.Free()

	vocabSize := 32000
	logitsData := make([]float32, vocabSize)
	for i := range logitsData {
		logitsData[i] = rand.Float32() * 5.0
	}

	logitsTensor := ctx.NewTensorFP32(1, vocabSize)
	defer logitsTensor.Free()
	logitsTensor.LoadFrom(logitsData)

	prev := []int32{10, 20, 30}

	b.ResetTimer()
	b.ReportAllocs()
	var seed uint64
	for b.Loop() {
		seed++
		_ = ctx.SampleTopKTopP(logitsTensor, prev, 1.1, 0.1, 0.1, 0.7, 50, 0.9, seed, vocabSize)
	}
}
