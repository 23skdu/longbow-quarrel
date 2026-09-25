//go:build linux && amd64 && cuda && cgo

package device

import (
	"math"
	"testing"
)

func TestCUDA_StoreKV_TurboQuant(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	heads := 4
	headDim := 128
	qjlRows := 64
	windowSize := 5

	// Setup TurboQuant matrices in context
	rot := ctx.NewTensorFP32(headDim, headDim)
	rotData := make([]float32, headDim*headDim)
	for i := 0; i < headDim; i++ {
		rotData[i*headDim+i] = 1.0 // Identity matrix for predictable rotation
	}
	_ = rot.LoadFromF32(rotData)
	ctx.TQRotation = rot
	defer rot.Free()

	qjl := ctx.NewTensorFP32(qjlRows, headDim)
	_ = qjl.LoadFromF32(make([]float32, qjlRows*headDim))
	ctx.TQQJL = qjl
	defer qjl.Free()

	kCache := ctx.NewTurboTensor(windowSize, heads*headDim, DataTypeTQ2_0, headDim, qjlRows)
	vCache := ctx.NewTurboTensor(windowSize, heads*headDim, DataTypeTQ2_0, headDim, qjlRows)
	if kCache == nil || vCache == nil {
		t.Fatal("Failed to allocate TurboQuant KV cache tensors")
	}
	defer kCache.Free()
	defer vCache.Free()

	k := ctx.NewTensorFP32(1, heads*headDim)
	v := ctx.NewTensorFP32(1, heads*headDim)
	defer k.Free()
	defer v.Free()

	kData := make([]float32, heads*headDim)
	vData := make([]float32, heads*headDim)
	for i := 0; i < heads*headDim; i++ {
		kData[i] = float32(math.Cos(float64(i)))
		vData[i] = float32(math.Sin(float64(i)))
	}
	_ = k.LoadFromF32(kData)
	_ = v.LoadFromF32(vData)

	pos := 1
	k.StoreKV(v, kCache, vCache, pos, heads, headDim, windowSize)

	kOut := ctx.NewTensorFP32(1, heads*headDim)
	vOut := ctx.NewTensorFP32(1, heads*headDim)
	defer kOut.Free()
	defer vOut.Free()

	kOut.FetchKV(vOut, kCache, vCache, pos, heads, headDim, windowSize)

	kOutData := kOut.ToHostF32()
	vOutData := vOut.ToHostF32()

	if len(kOutData) != heads*headDim || len(vOutData) != heads*headDim {
		t.Fatalf("Fetched dimensions mismatch: k=%d, v=%d", len(kOutData), len(vOutData))
	}

	for i := 0; i < heads*headDim; i++ {
		if (kData[i] > 0.1 && kOutData[i] <= 0) || (kData[i] < -0.1 && kOutData[i] >= 0) {
			t.Errorf("FetchKV: numerical mismatch at k[%d]: got %f, want sign of %f", i, kOutData[i], kData[i])
		}
	}
}
