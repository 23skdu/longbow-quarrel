//go:build darwin && metal

package device

import (
	"encoding/binary"
	"math"
	"os"
	"testing"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

// TestQ6K_DequantizationReference compares our Q6K dequantization against llama.cpp reference
func TestQ6K_DequantizationReference(t *testing.T) {
	// Create a test block with known values
	block := createTestQ6KBlock()

	// Dequantize using our implementation
	ourWeights := gguf.DequantizeQ6K(block, 256)

	refWeights := DequantizeQ6K_ReferenceTest(block)

	// Compare results
	maxDiff := 0.0
	totalDiff := 0.0
	for i := 0; i < 256; i++ {
		diff := math.Abs(float64(ourWeights[i] - refWeights[i]))
		totalDiff += diff
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > 1e-6 {
			t.Errorf("Weight %d: our=%f, ref=%f, diff=%e", i, ourWeights[i], refWeights[i], diff)
		}
	}

	avgDiff := totalDiff / 256.0
	t.Logf("Q6K Dequantization Comparison:")
	t.Logf("  Max difference: %e", maxDiff)
	t.Logf("  Avg difference: %e", avgDiff)

	if maxDiff > 1e-6 {
		t.Errorf("Q6K dequantization differs from reference by more than 1e-6")
	}
}

// TestQ6K_LinearAccuracy tests Q6K * F16 -> F32 linear operations
func TestQ6K_LinearAccuracy(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	// Create test Q6K weights
	block := createTestQ6KBlock()
	q6kWeights := gguf.DequantizeQ6K(block, 256)

	// Create Q6K tensor on GPU
	weightTensor, err := ctx.NewQ6KTensor(16, 16) // 16x16 = 256 weights
	if err != nil {
		t.Fatalf("Failed to create Q6K tensor: %v", err)
	}
	defer weightTensor.Free()

	if err := weightTensor.LoadFromRaw(block); err != nil {
		t.Fatalf("Failed to load Q6K data: %v", err)
	}

	inputData := make([]float32, 16)
	for i := range inputData {
		inputData[i] = float32(i+1) * 0.1
	}

	inputTensor := ctx.NewTensor(16, 1)
	defer inputTensor.Free()
	inputTensor.LoadFrom(inputData)

	outputTensor := weightTensor.MatMul(inputTensor)
	defer outputTensor.Free()

	if err := ctx.WaitWithTimeout(2 * time.Second); err != nil {
		t.Fatalf("GPU operation failed: %v", err)
	}

	gpuResult := outputTensor.ToHost()

	// Compute CPU reference: Q6K(dequantized) * F16 -> F32
	cpuResult := make([]float32, 16)
	for i := 0; i < 16; i++ {
		sum := 0.0
		for j := 0; j < 16; j++ {
			sum += float64(q6kWeights[i*16+j]) * float64(inputData[j])
		}
		cpuResult[i] = float32(sum)
	}

	// Compare results
	maxDiff := 0.0
	totalDiff := 0.0
	for i := 0; i < 16; i++ {
		diff := math.Abs(float64(gpuResult[i] - cpuResult[i]))
		totalDiff += diff
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > 1e-4 {
			t.Errorf("Output %d: gpu=%f, cpu=%f, diff=%e", i, gpuResult[i], cpuResult[i], diff)
		}
	}

	avgDiff := totalDiff / 16.0
	t.Logf("Q6K Linear Operation Comparison:")
	t.Logf("  Max difference: %e", maxDiff)
	t.Logf("  Avg difference: %e", avgDiff)

	if maxDiff > 1e-3 {
		t.Errorf("Q6K linear operation differs from CPU reference by more than 1e-3")
	}
}

// TestQ6K_RealMistralBlock tests with actual Mistral model data
func TestQ6K_RealMistralBlock(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	// Load real Q6K block from Mistral model
	blockData, err := os.ReadFile("../../token_the_q6k_block.bin")
	if err != nil {
		t.Skip("Real Q6K block file not found - skipping real data test")
	}

	if len(blockData) < 210 {
		t.Fatalf("Invalid Q6K block size: got %d, want 210", len(blockData))
	}

	ourWeights := gguf.DequantizeQ6K(blockData, 256)
	refWeights := DequantizeQ6K_ReferenceTest(blockData)

	maxDiff := 0.0
	for i := 0; i < 256; i++ {
		diff := math.Abs(float64(ourWeights[i] - refWeights[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	t.Logf("Real Mistral Q6K Block - Max dequantization diff: %e", maxDiff)
	if maxDiff > 1e-6 {
		t.Errorf("Real Q6K block dequantization differs from reference")
	}

	// Test linear operation with real data
	weightTensor, err := ctx.NewQ6KTensor(16, 16)
	if err != nil {
		t.Fatalf("Failed to create Q6K tensor: %v", err)
	}
	defer weightTensor.Free()

	if err := weightTensor.LoadFromRaw(blockData); err != nil {
		t.Fatalf("Failed to load Q6K data: %v", err)
	}

	inputData := []float32{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}

	inputTensor := ctx.NewTensor(16, 1)
	defer inputTensor.Free()
	inputTensor.LoadFrom(inputData)

	outputTensor := weightTensor.MatMul(inputTensor)
	defer outputTensor.Free()

	if err := ctx.WaitWithTimeout(2 * time.Second); err != nil {
		t.Fatalf("GPU operation failed: %v", err)
	}

	gpuResult := outputTensor.ToHost()

	// Compute expected result (sum of weights since input is all ones)
	expectedSum := 0.0
	for i := 0; i < 256; i++ {
		expectedSum += float64(ourWeights[i])
	}

	// Each output should be sum of 16 weights from its row
	for i := 0; i < 16; i++ {
		rowSum := 0.0
		for j := 0; j < 16; j++ {
			rowSum += float64(ourWeights[i*16+j])
		}
		diff := math.Abs(float64(gpuResult[i]) - rowSum)
		if diff > 1e-3 {
			t.Errorf("Row %d: gpu=%f, expected=%f, diff=%e", i, gpuResult[i], rowSum, diff)
		}
	}

	t.Logf("Real Mistral Q6K linear test passed")
}

func createTestQ6KBlock() []byte {
	block := make([]byte, 210)

	// Fill with predictable pattern
	for i := range block {
		block[i] = byte(i % 256)
	}

	// Set d (super-scale) to a reasonable value
	binary.LittleEndian.PutUint16(block[208:210], Float32ToFloat16(0.5))

	// Set scales to reasonable values
	for i := 0; i < 16; i++ {
		block[192+i] = byte(int8(i - 8)) // Range from -8 to 7
	}

	return block
}

func DequantizeQ6K_ReferenceTest(block []byte) []float32 {
	ql := block[0:128]
	qh := block[128:192]
	scales := block[192:208]
	d := Float16ToFloat32(binary.LittleEndian.Uint16(block[208:210]))

	out := make([]float32, 256)

	for l := 0; l < 16; l++ {
		s := d * float32(int8(scales[l]))
		subOffset := l * 16
		for k := 0; k < 16; k += 2 {
			idx := subOffset + k

			b := ql[idx/2]
			q1 := int8(b & 0xF)
			q2 := int8(b >> 4)

			hbyte1 := qh[idx/4]
			hval1 := (hbyte1 >> ((idx % 4) * 2)) & 3

			hbyte2 := qh[(idx+1)/4]
			hval2 := (hbyte2 >> (((idx + 1) % 4) * 2)) & 3

			w0_raw := int8((hval1 << 4) | byte(q1))
			w0 := s * (float32(w0_raw) - 32.0)

			w1_raw := int8((hval2 << 4) | byte(q2))
			w1 := s * (float32(w1_raw) - 32.0)

			out[idx] = w0
			out[idx+1] = w1
		}
	}
	return out
}
