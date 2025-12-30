package device

import (
	"encoding/binary"
	"math"
	"os"
	"testing"
)

// TestQ4K_RealMistralBlock validates CPU and GPU dequantization using a real block from Mistral model
func TestQ4K_RealMistralBlock(t *testing.T) {
	// Load the real Q4K block from the extracted file
	blockData, err := os.ReadFile("/Users/rsd/REPOS/longbow-quarrel/mistral_q4k_block_0.bin")
	if err != nil {
		t.Fatalf("Failed to load real Q4K block: %v", err)
	}
	
	if len(blockData) != 144 {
		t.Fatalf("Expected 144 bytes, got %d", len(blockData))
	}
	
	// Parse the block structure
	d := Float16ToFloat32(binary.LittleEndian.Uint16(blockData[0:2]))
	dmin := Float16ToFloat32(binary.LittleEndian.Uint16(blockData[2:4]))
	
	t.Logf("Real Mistral Block:")
	t.Logf("  d (scale):     %.12f", d)
	t.Logf("  dmin (offset): %.12f", dmin)
	
	// Verify these values match what we saw in extraction
	expectedD := float32(0.000024795532)
	expectedDmin := float32(0.000224232674)
	
	if math.Abs(float64(d-expectedD)) > 0.000001 {
		t.Errorf("d mismatch: got %.12f, expected %.12f", d, expectedD)
	}
	if math.Abs(float64(dmin-expectedDmin)) > 0.000001 {
		t.Errorf("dmin mismatch: got %.12f, expected %.12f", dmin, expectedDmin)
	}
	
	// Run CPU reference dequantization
	cpuWeights := DequantizeQ4K_Reference(blockData)
	
	// Log statistics
	var minW, maxW, sumW float32 = math.MaxFloat32, -math.MaxFloat32, 0
	nonZeroCount := 0
	for i := 0; i < 256; i++ {
		w := cpuWeights[i]
		if w < minW {
			minW = w
		}
		if w > maxW {
			maxW = w
		}
		sumW += w
		if w != 0 {
			nonZeroCount++
		}
	}
	meanW := sumW / 256
	
	t.Logf("CPU Dequantized Weights:")
	t.Logf("  Min:      %.12f", minW)
	t.Logf("  Max:      %.12f", maxW)
	t.Logf("  Mean:     %.12f", meanW)
	t.Logf("  Range:    %.12f", maxW-minW)
	t.Logf("  NonZero:  %d/256", nonZeroCount)
	t.Logf("  First 8:  %v", cpuWeights[0:8])
	
	// Critical validation: weights should NOT all be zero
	if nonZeroCount == 0 {
		t.Errorf("CRITICAL: All CPU dequantized weights are ZERO - subnormal bug still present!")
	}
	
	// Test GPU kernel with the same real block
	ctx := NewContext()
	defer ctx.Free()
	
	// Create input tensor (all ones for simple dot product test)
	input := ctx.NewTensorFP32(1, 256)
	defer input.Free()
	inputData := make([]float32, 256)
	for i := range inputData {
		inputData[i] = 1.0
	}
	input.LoadFrom(inputData)
	
	// Create weight tensor and load the real Q4K block
	// For a single block: rows=1, cols=256
	weight := ctx.NewTensor(1, 256)
	defer weight.Free()
	weight.LoadRaw(blockData)
	
	// Create output tensor
	output := ctx.NewTensorFP32(1, 1)
	defer output.Free()
	
	// Run GPU linear projection
	input.LinearInto(weight, output)
	
	// Read GPU result
	gpuResult := output.ToHostF32()
	
	// Expected result: sum of all CPU dequantized weights (dot product with all-ones)
	expectedSum := sumW
	
	t.Logf("GPU vs CPU Comparison:")
	t.Logf("  CPU sum (expected): %.12f", expectedSum)
	t.Logf("  GPU result:         %.12f", gpuResult[0])
	
	// Check for NaN first
	if math.IsNaN(float64(gpuResult[0])) {
		t.Fatalf("GPU returned NaN! This means the kernel has a critical bug.")
	}
	
	// Calculate relative error
	relErr := math.Abs(float64(gpuResult[0]-expectedSum)) / math.Max(math.Abs(float64(expectedSum)), 1e-9)
	t.Logf("  Relative Error:     %.6f%%", relErr*100)
	
	// Test tolerance: relative error should be < 1%
	if relErr > 0.01 {
		t.Errorf("GPU vs CPU mismatch too large: %.6f%% (threshold: 1%%)", relErr*100)
	}
}
