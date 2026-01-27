//go:build darwin && metal

package device

import (
	"encoding/binary"
	"math"
	"testing"
	"time"
)

func TestQ6K_SimpleDebug(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	// Manually construct 1 block (210 bytes) for 256 weights
	// All weights will be 0.
	block := make([]byte, 210)

	// In Q6K, dequantized value = (float(sc[l]) * d) * (int8_t((hval << 4) | (b & 0xF)) - 32)
	// To get 0.0:
	// 1. Set d = 1.0 (float16: 0x3C00)
	binary.LittleEndian.PutUint16(block[208:210], 0x3C00)

	// 2. Set all 16 scales to 1 (int8 scales start at block[192])
	for i := 0; i < 16; i++ {
		block[192+i] = 1
	}

	// 3. Set all quants (ql and qh) such that (hval << 4 | b) == 32
	// For b (ql): lower and upper nibbles should be 0 (since 32 & 0xF is 0)
	// For h (qh): bits should be 2 (since 32 >> 4 is 2)
	// ql: block[0:128] -> 0
	// qh: block[128:192] -> each byte has four 2-bit values.
	// 2 binary is 10. So byte of four '2's is 10101010 = 0xAA
	for i := 0; i < 64; i++ {
		block[128+i] = 0xAA
	}

	// Now dequantized weights should all be 1.0 * (32 - 32) = 0.0

	// Weight: [1, 256] (1 row, 1 standard block)
	weightTensor, err := ctx.NewQ6KTensor(1, 256)
	if err != nil {
		t.Fatalf("Failed to create Q6K tensor: %v", err)
	}
	defer weightTensor.Free()

	if err := weightTensor.LoadFromRaw(block); err != nil {
		t.Fatalf("Failed to load Q6K data: %v", err)
	}

	// Input: [256, 1] (Column vector)
	inputData := make([]float32, 256)
	for i := range inputData {
		inputData[i] = 1.0
	}

	inputTensor := ctx.NewTensor(256, 1)
	defer inputTensor.Free()
	inputTensor.LoadFrom(inputData)

	// Result: [1, 256] * [256, 1] -> [1, 1]
	outputTensor := weightTensor.MatMul(inputTensor)
	defer outputTensor.Free()

	if err := ctx.WaitWithTimeout(2 * time.Second); err != nil {
		t.Fatalf("GPU operation failed: %v", err)
	}

	// Read full buffer to avoid dimension mismatch panic in test
	gpuResult := outputTensor.ToHost()

	// First element is the logical result
	result := gpuResult[0]
	t.Logf("Debug output value: %f (Buffer size: %d)", result, len(gpuResult))

	if math.Abs(float64(result)) > 1e-4 {
		t.Errorf("Expected 0.0, got %f", result)
	}
}
