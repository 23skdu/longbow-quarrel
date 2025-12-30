package device

import (
	"encoding/binary"
	"math"
	"os"
	"testing"
)

// TestQ4K_RealMistralBlock validates CPU and GPU dequantization using a real block from Mistral model
func TestQ4K_RealMistralBlock(t *testing.T) {
	blockData, err := os.ReadFile("mistral_q4k_block_0.bin")
	if err != nil {
		t.Skip("Real block file mistral_q4k_block_0.bin not found, skipping")
	}
	
	if len(blockData) != 144 {
		t.Fatalf("Expected 144 bytes, got %d", len(blockData))
	}
	
	// Parse the block structure
	d := Float16ToFloat32(binary.LittleEndian.Uint16(blockData[0:2]))
	
	// Verify these values match what we saw in extraction
	expectedD := float32(0.000024795532)
	
	if math.Abs(float64(d-expectedD)) > 0.000001 {
		t.Errorf("d mismatch: got %.12f, expected %.12f", d, expectedD)
	}
	
	cpuWeights := DequantizeQ4K_Reference(blockData)
	
	// Stats
	var sumW float32
	nonZeroCount := 0
	for i := 0; i < 256; i++ {
		w := cpuWeights[i]
		sumW += w
		if w != 0 { nonZeroCount++ }
	}
	t.Logf("CPU Sum: %f", sumW)
	
	if nonZeroCount == 0 {
		t.Errorf("CRITICAL: All CPU dequantized weights are ZERO!")
	}
	
	ctx := NewContext()
	defer ctx.Free()
	
	input := ctx.NewTensorFP32(1, 256)
	defer input.Free()
	inputData := make([]float32, 256)
	for i := range inputData { inputData[i] = 1.0 }
	input.LoadFrom(inputData)
	
	weight := ctx.NewQ4KTensor(1, 256)
	defer weight.Free()
	weight.LoadRaw(blockData)
	
	output := ctx.NewTensorFP32(1, 1)
	defer output.Free()
	
	input.LinearInto(weight, output)
	
	gpuResult := output.ToHostF32()
	t.Logf("GPU Result: %f", gpuResult[0])
	
	diff := math.Abs(float64(gpuResult[0]-sumW))
	if diff > 0.01 {
		t.Errorf("GPU vs CPU mismatch: %f", diff)
	}
}

func TestEmbeddingQ4K_RealMistralBlock(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	blockData, err := os.ReadFile("../../token_the_q4k_block.bin")
	if err != nil {
		t.Skip("Real block file not found")
	}

	cpuOut := DequantizeQ4K_Reference(blockData)

	cols := 256
	weight := ctx.NewQ4KTensor(1, cols)
	defer weight.Free()
	weight.LoadRaw(blockData)
	
	res := weight.EmbeddingLookup(0) 
	defer res.Free()
	ctx.Synchronize()
	
	gpuOutF16 := res.ToHost()
	
	var diffSum float32
	for i := 0; i < 256; i++ {
		d := math.Abs(float64(gpuOutF16[i] - cpuOut[i]))
		diffSum += float32(d)
	}
	avgDiff := diffSum / 256.0
	t.Logf("Average Diff: %f", avgDiff)
	
	if avgDiff > 0.0001 { 
		t.Errorf("Validation failed! Average diff %f is too high", avgDiff)
	}
}

func TestLinearQ4K_RealMistralBlock(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	blockData, err := os.ReadFile("../../token_the_q4k_block.bin")
	if err != nil {
		t.Skip("Real block file not found")
	}

	inputVec := make([]float32, 256)
	for i := range inputVec {
		inputVec[i] = float32(i) * 0.01 
	}
	
	cpuWeights := DequantizeQ4K_Reference(blockData)
	expectedDot := float32(0)
	for i := 0; i < 256; i++ {
		expectedDot += cpuWeights[i] * inputVec[i]
	}
	t.Logf("Expected Dot Product: %f", expectedDot)

	weight := ctx.NewQ4KTensor(1, 256)
	defer weight.Free()
	weight.LoadRaw(blockData)
	
	input := ctx.NewTensorFP32(1, 256)
	defer input.Free()
	input.LoadFrom(inputVec)
	
	output := ctx.NewTensorFP32(1, 1)
	defer output.Free()
	
	input.LinearInto(weight, output)
	
	ctx.Synchronize()
	gpuResult := output.ToHostF32()
	
	t.Logf("GPU Result: %f", gpuResult[0])
	
	diff := math.Abs(float64(gpuResult[0] - expectedDot))
	t.Logf("Diff: %f", diff)
	
	if diff > 0.01 { 
		t.Errorf("Linear Kernel Mismatch! Expected %f, Got %f", expectedDot, gpuResult[0])
	}
}

func TestLinearQ4K_F16_RealMistralBlock(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	blockData, err := os.ReadFile("../../token_the_q4k_block.bin")
	if err != nil {
		t.Skip("Real block file not found")
	}

	inputVec := make([]float32, 256)
	for i := range inputVec {
		inputVec[i] = float32(i) * 0.01 
	}
	
	cpuWeights := DequantizeQ4K_Reference(blockData)
	expectedDot := float32(0)
	for i := 0; i < 256; i++ {
		expectedDot += cpuWeights[i] * inputVec[i]
	}
	t.Logf("Expected Dot Product: %f", expectedDot)

	weight := ctx.NewQ4KTensor(1, 256)
	defer weight.Free()
	weight.LoadRaw(blockData)
	
	input := ctx.NewTensor(1, 256) // Default F16
	defer input.Free()
	input.LoadFrom(inputVec)
	
	output := ctx.NewTensor(1, 1) // Default F16
	defer output.Free()
	
	input.LinearInto(weight, output)
	
	ctx.Synchronize()
	gpuResult := output.ToHost() // Read F16 as float32 slice
	
	t.Logf("GPU Result (F16): %f", gpuResult[0])
	
	diff := math.Abs(float64(gpuResult[0] - expectedDot))
	t.Logf("Diff: %f", diff)
	
	if diff > 0.02 { 
		t.Errorf("Linear F16 Kernel Mismatch! Expected %f, Got %f", expectedDot, gpuResult[0])
	}
}

// DequantizeQ6K_Reference implements Q6K dequantization
func DequantizeQ6K_Reference(block []byte) []float32 {
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
			
			hbyte1 := qh[idx/4] // k=0,2 -> idx=0,2 -> idx/4=0. byte 0 covers idx 0,1,2,3
			// idx % 4: 0, 2
			// hbyte1 >> 0 (for idx)
			hval1 := (hbyte1 >> ((idx % 4) * 2)) & 3
			
			hbyte2 := qh[(idx+1)/4] 
			// idx+1 % 4: 1, 3
			hval2 := (hbyte2 >> (((idx+1) % 4) * 2)) & 3
			
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

func TestLinearQ6K_RealMistralBlock(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	blockData, err := os.ReadFile("../../token_the_q6k_block.bin")
	if err != nil {
		t.Skip("Real Q6K block file not found")
	}

	rows := 1
	cols := 256
	inputVec := make([]float32, 256)
	for i := range inputVec {
		inputVec[i] = float32(i) * 0.01 
	}
	
	cpuWeights := DequantizeQ6K_Reference(blockData)
	expectedDot := float32(0)
	for i := 0; i < 256; i++ {
		expectedDot += cpuWeights[i] * inputVec[i]
	}
	
	t.Logf("Expected Dot Product (Q6K): %f", expectedDot)

	weight := ctx.NewQ6KTensor(rows, cols)
	defer weight.Free()
	weight.LoadRaw(blockData)
	
	// Use F16 output to trigger q6k_f16 kernel (check logic in metal.go: if type==Q6K && out==F16 -> Linear_Q6K_F16)
	// Input F16
	input := ctx.NewTensor(rows, cols)
	defer input.Free()
	input.LoadFrom(inputVec)
	
	output := ctx.NewTensor(1, 1)
	defer output.Free()
	
	input.LinearInto(weight, output)
	ctx.Synchronize()
	
	gpuResult := output.ToHost()
	t.Logf("GPU Result: %f", gpuResult[0])
	
	diff := math.Abs(float64(gpuResult[0] - expectedDot))
	t.Logf("Diff: %f", diff)
	
	if diff > 0.05 { // Tolerance for F16/Subnormal
		t.Errorf("Q6K Linear Mismatch! Expected %f, Got %f", expectedDot, gpuResult[0])
	}
}
