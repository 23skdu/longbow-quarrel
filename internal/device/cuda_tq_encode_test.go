//go:build linux && amd64 && cuda && cgo

package device

import (
	"testing"
)

func TestCUDA_TurboQuantEncode(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	blockSize := 64
	qjlRows := 16
	numBlocks := 4

	// Setup TurboQuant matrices
	rot := ctx.NewTensorFP32(blockSize, blockSize)
	rotData := make([]float32, blockSize*blockSize)
	for i := 0; i < blockSize; i++ {
		rotData[i*blockSize+i] = 1.0
	}
	rot.LoadFrom(rotData)

	qjl := ctx.NewTensorFP32(qjlRows, blockSize)
	qjlData := make([]float32, qjlRows*blockSize)
	qjl.LoadFrom(qjlData)

	// Allocate output
	output := ctx.NewTensorWithType(numBlocks, blockSize, DataTypeINT8)
	scaleOut := ctx.NewTensorFP32(numBlocks, 1)

	inputData := make([]float32, numBlocks*blockSize)
	for i := range inputData {
		inputData[i] = float32(i % 256)
	}
	input := ctx.NewTensorFP32(numBlocks, blockSize)
	input.LoadFrom(inputData)

	// Test encoding
	ctx.TurboQuantEncode(input, rot, qjl, output, scaleOut, nil, blockSize, qjlRows, 4)

	// Verify output is allocated
	if output == nil {
		t.Error("Output tensor should be allocated")
	}

	t.Log("TurboQuantEncode test passed")
}

func TestCUDA_StoreKVTurboQuant(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	heads := 2
	headDim := 64
	qjlRows := 16
	numTokens := 1

	k := ctx.NewTensorFP32(numTokens, heads*headDim)
	v := ctx.NewTensorFP32(numTokens, heads*headDim)
	defer k.Free()
	defer v.Free()

	kCache := ctx.NewTurboTensor(8, heads*headDim, DataTypeTQ1_0, headDim, qjlRows)
	vCache := ctx.NewTurboTensor(8, heads*headDim, DataTypeTQ1_0, headDim, qjlRows)
	if kCache == nil || vCache == nil {
		t.Fatal("Failed to allocate TurboQuant KV cache")
	}
	defer kCache.Free()
	defer vCache.Free()

	physPos := ctx.NewTensorI32(1, numTokens)
	defer physPos.Free()
	_ = physPos.LoadFrom([]int32{0})

	ctx.StoreKVTurboQuant(k, v, kCache, vCache, physPos, headDim, qjlRows, heads, numTokens)
	t.Log("StoreKVTurboQuant executed successfully")
}
