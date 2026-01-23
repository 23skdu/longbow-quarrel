//go:build darwin && metal

package device

import (
	"testing"
	"time"
)

func TestQ6K_SimpleDebug(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	block := make([]byte, 210)
	block[208] = 0x00
	block[209] = 0x3C
	block[192] = 32
	block[0] = 0xFF

	weightTensor, err := ctx.NewQ6KTensor(1, 256)
	if err != nil {
		t.Fatalf("Failed to create Q6K tensor: %v", err)
	}
	defer weightTensor.Free()

	if err := weightTensor.LoadFromRaw(block); err != nil {
		t.Fatalf("Failed to load Q6K data: %v", err)
	}

	inputData := make([]float32, 256)
	for i := range inputData {
		inputData[i] = 1.0
	}

	inputTensor := ctx.NewTensor(1, 256)
	defer inputTensor.Free()
	inputTensor.LoadFrom(inputData)

	outputTensor := weightTensor.MatMul(inputTensor)
	defer outputTensor.Free()

	if err := ctx.WaitWithTimeout(2 * time.Second); err != nil {
		t.Fatalf("GPU operation failed: %v", err)
	}

	gpuResult := outputTensor.ToHost()

	if len(gpuResult) != 1 {
		t.Fatalf("Expected 1 output, got %d", len(gpuResult))
	}

	result := gpuResult[0]
	t.Logf("Single output value: %f", result)

	if result != 0.0 {
		t.Errorf("Expected 0.0, got %f", result)
	} else {
		t.Logf("Got expected 0.0 result")
	}
}
