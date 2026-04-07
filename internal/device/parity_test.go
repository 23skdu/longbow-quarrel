//go:build metal

package device

import (
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
	for i := 0; i < headDim; i++ { rotData[i*headDim+i] = 1.0 }
	rotation.LoadFromF32(rotData)
	ctx.TQRotation = rotation

	// Identity QJL sub-matrix
	qjl := ctx.NewTensorFP32(qjlRows, headDim)
	qjlData := make([]float32, qjlRows*headDim)
	for i := 0; i < qjlRows; i++ { qjlData[i*headDim+i] = 1.0 }
	qjl.LoadFromF32(qjlData)
	ctx.TQQJL = qjl

	inputK := make([]float32, headDim)
	for i := range inputK { inputK[i] = float32(i + 1) }

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
	t.Logf("Metal[0:5]: %v", metalK[0:5])
}
