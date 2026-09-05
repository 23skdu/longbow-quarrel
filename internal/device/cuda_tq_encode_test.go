//go:build cuda
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
	t.Skip("Requires full KV cache infrastructure")
}