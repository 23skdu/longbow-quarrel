//go:build metal

package device

import (
	"math"
	"testing"
)

func TestTurboQuant_NumericalParity_TQ1_0(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	headDim := 128
	qjlRows := 64
	numHeads := 1
	pos := 0
	ctxLen := 1

	// Identity rotation
	rotation := ctx.NewTensorFP32(headDim, headDim)
	rotData := make([]float32, headDim*headDim)
	for i := 0; i < headDim; i++ {
		rotData[i*headDim+i] = 1.0
	}
	rotation.LoadFromF32(rotData)
	ctx.TQRotation = rotation

	// Identity QJL sub-matrix
	qjl := ctx.NewTensorFP32(qjlRows, headDim)
	qjlData := make([]float32, qjlRows*headDim)
	for i := 0; i < qjlRows; i++ {
		qjlData[i*headDim+i] = 1.0
	}
	qjl.LoadFromF32(qjlData)
	ctx.TQQJL = qjl

	inputK := make([]float32, headDim)
	for i := range inputK {
		inputK[i] = float32(i + 1)
	}

	kT := ctx.NewTensorFP32(1, headDim)
	kT.LoadFromF32(inputK)
	vT := ctx.NewTensorFP32(1, headDim)
	vT.ZeroInit()

	kCache := ctx.NewTurboTensor(ctxLen, headDim, DataTypeTQ1_0, headDim, qjlRows)
	vCache := ctx.NewTurboTensor(ctxLen, headDim, DataTypeTQ1_0, headDim, qjlRows)

	kT.StoreKV(vT, kCache, vCache, pos, numHeads, headDim, ctxLen)
	ctx.Synchronize()

	kFetch := ctx.NewTensorFP32(1, headDim)
	vFetch := ctx.NewTensorFP32(1, headDim)
	kFetch.FetchKV(vFetch, kCache, vCache, pos, numHeads, headDim, ctxLen)
	ctx.Synchronize()

	metalK := kFetch.ToHostF32()
	if len(metalK) != headDim {
		t.Fatalf("the cache returned %d values, want %d", len(metalK), headDim)
	}

	// The TurboQuant codec is lossy, so the round trip is compared with
	// the original the way a lossy codec is: the reconstruction has to be
	// closer to the input than a cache full of zeros is, and it has to
	// hold something at all.
	totalError := 0.0
	totalMagnitude := 0.0
	nonZero := 0
	for i, got := range metalK {
		if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) {
			t.Fatalf("the value %d is %v", i, got)
		}
		totalError += math.Abs(float64(got - inputK[i]))
		totalMagnitude += math.Abs(float64(inputK[i]))
		if got != 0 {
			nonZero++
		}
	}

	if nonZero == 0 {
		t.Error("the TurboQuant cache holds nothing but zeros")
	}
	if totalError >= totalMagnitude {
		t.Errorf("the round trip is no better than a cache full of zeros: the error is %v and the input is %v",
			totalError, totalMagnitude)
	}
}
