//go:build darwin && metal

package device

import (
	"math"
	"testing"
)

// TestAttentionOutputProjection_RealScales tests the attention output projection
// with realistic Q4K weight scales observed in Mistral (~0.0001)
func TestAttentionOutputProjection_RealScales(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	// Mistral attention output: 4096x4096, Q4K quantized
	// Observed scales: ~0.00007 to ~0.0002
	// Input magnitude: ~0.2 to ~3.6 (from attention weighted sum)
	// Expected output: Should be proportional to input * weight_scale
	
	// Simplified test: 256x256 matrix
	M := 1    // Single token
	N := 256  // Output dim
	K := 256  // Input dim (1 Q4K block)
	
	// Create input with magnitude ~1.0
	input := ctx.NewTensorPooled(M, K)
	inputData := make([]float32, K)
	for i := 0; i < K; i++ {
		inputData[i] = 1.0 // All 1.0
	}
	input.LoadFrom(inputData)
	
	// Create Q4K weight with very small scale (mimicking attention output weights)
	weight := ctx.NewQ4KTensor(N, K)
	
	// Construct Q4K block with d=0.0001 (similar to observed scales)
	blockSize := 144
	numBlocks := (N * K) / 256
	rawData := make([]byte, numBlocks * blockSize)
	
	// d = 0.0001 in FP16
	dF16 := Float32ToFloat16(0.0001)
	
	for b := 0; b < numBlocks; b++ {
		offset := b * blockSize
		
		// Set d (bytes 0-1)
		rawData[offset] = byte(dF16 & 0xFF)
		rawData[offset+1] = byte(dF16 >> 8)
		
		// dmin = 0.0 (bytes 2-3 already zero)
		
		// Set scales: use value 1 for first scale group
		rawData[offset+4] = 1 // sc[0] = 1
		
		// Set quants: use value 1 for all weights
		for i := 16; i < blockSize; i++ {
			rawData[offset+i] = 0x11 // Each byte encodes two 4-bit values = 1
		}
	}
	
	weight.LoadRaw(rawData)
	
	// Compute output
	output := ctx.NewTensorPooled(M, N)
	input.LinearInto(weight, output, 1.0)
	ctx.Synchronize()
	
	// Check output
	outputData := output.ToHost()
	
	// Expected: With d=0.0001, sc=1, q=1, each weight ≈ 0.0001
	// Input is all 1.0, so output[i] ≈ sum(0.0001 * 1.0) over K elements
	// But only first 32 elements have sc=1, rest have sc=0
	// So expected ≈ 32 * 0.0001 = 0.0032
	
	maxVal := float32(0)
	for _, v := range outputData {
		if math.Abs(float64(v)) > float64(maxVal) {
			maxVal = float32(math.Abs(float64(v)))
		}
	}
	
	t.Logf("Attention Output Projection Test:")
	t.Logf("  Input magnitude: 1.0")
	t.Logf("  Weight scale (d): 0.0001")
	t.Logf("  Output max: %.6f", maxVal)
	t.Logf("  Expected range: ~0.001 to ~0.01")
	
	// Output should be small but non-zero
	if maxVal < 0.0001 {
		t.Errorf("Output too small: %.6f (expected > 0.0001)", maxVal)
	}
	if maxVal > 0.1 {
		t.Errorf("Output too large: %.6f (expected < 0.1)", maxVal)
	}
	
	// Now test with larger input magnitude (like actual attention outputs)
	inputData2 := make([]float32, K)
	for i := 0; i < K; i++ {
		inputData2[i] = 2.0 // Magnitude 2.0
	}
	input.LoadFrom(inputData2)
	
	input.LinearInto(weight, output, 1.0)
	ctx.Synchronize()
	
	outputData2 := output.ToHost()
	maxVal2 := float32(0)
	for _, v := range outputData2 {
		if math.Abs(float64(v)) > float64(maxVal2) {
			maxVal2 = float32(math.Abs(float64(v)))
		}
	}
	
	t.Logf("With input magnitude 2.0:")
	t.Logf("  Output max: %.6f", maxVal2)
	t.Logf("  Ratio to first test: %.2f", maxVal2/maxVal)
	
	// Output should scale linearly with input
	ratio := maxVal2 / maxVal
	if ratio < 1.5 || ratio > 2.5 {
		t.Errorf("Output scaling incorrect: ratio=%.2f (expected ~2.0)", ratio)
	}
}

// TestAttentionOutputProjection_Comparison compares Q4K vs F16 linear transformation
func TestAttentionOutputProjection_Comparison(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()
	
	M, N, K := 1, 128, 128
	
	// Create identical input
	input := ctx.NewTensorPooled(M, K)
	inputData := make([]float32, K)
	for i := 0; i < K; i++ {
		inputData[i] = float32(i%10) * 0.1 // Pattern 0.0, 0.1, ..., 0.9
	}
	input.LoadFrom(inputData)
	
	// Create F16 weight
	weightF16 := ctx.NewTensorPooled(N, K)
	weightDataF16 := make([]float32, N*K)
	for i := 0; i < N*K; i++ {
		weightDataF16[i] = 0.01 // Small uniform weight
	}
	weightF16.LoadFrom(weightDataF16)
	
	// Compute F16 output
	outputF16 := ctx.NewTensorPooled(M, N)
	input.LinearInto(weightF16, outputF16, 1.0)
	ctx.Synchronize()
	
	outputDataF16 := outputF16.ToHost()
	
	// Log F16 results
	t.Logf("F16 Linear Transformation:")
	t.Logf("  Output[0]: %.6f", outputDataF16[0])
	t.Logf("  Output max: %.6f", findMax(outputDataF16))
	
	// Note: Q4K comparison would require constructing equivalent Q4K weights
	// which is complex. This test establishes F16 baseline.
}

func findMax(data []float32) float32 {
	maxVal := float32(0)
	for _, v := range data {
		if math.Abs(float64(v)) > float64(maxVal) {
			maxVal = float32(math.Abs(float64(v)))
		}
	}
	return maxVal
}
