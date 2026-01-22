//go:build darwin && metal

package device

import (
	"encoding/binary"
	"fmt"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
	"math"
	"testing"
)

// DequantizeQ4K_Reference implements llama.cpp's Q4K dequantization in Go
func DequantizeQ4K_Reference(block []byte) []float32 {
	if len(block) != 144 {
		panic(fmt.Sprintf("Invalid Q4K block size: %d", len(block)))
	}

	out := make([]float32, 256)

	// Read d and dmin (FP16)
	d := Float16ToFloat32(binary.LittleEndian.Uint16(block[0:2]))
	dmin := Float16ToFloat32(binary.LittleEndian.Uint16(block[2:4]))

	fmt.Printf("DEBUG Q4K: d=%.9f, dmin=%.9f\n", d, dmin)

	// Read scales (12 bytes)
	scales := block[4:16]

	// Read quants (128 bytes)
	qs := block[16:144]

	// Decode scales (6-bit packed)
	// llama.cpp format: 8 scale values, each 6 bits, packed
	sc := make([]uint8, 8)
	m := make([]uint8, 8)

	// First 4 scales
	sc[0] = scales[0] & 0x3F
	sc[1] = scales[1] & 0x3F
	sc[2] = scales[2] & 0x3F
	sc[3] = scales[3] & 0x3F

	// First 4 mins
	m[0] = scales[4] & 0x3F
	m[1] = scales[5] & 0x3F
	m[2] = scales[6] & 0x3F
	m[3] = scales[7] & 0x3F

	// Next 4 scales (more complex packing)
	sc[4] = (scales[8] & 0x0F) | ((scales[0] & 0xC0) >> 2)
	sc[5] = (scales[9] & 0x0F) | ((scales[1] & 0xC0) >> 2)
	sc[6] = (scales[10] & 0x0F) | ((scales[2] & 0xC0) >> 2)
	sc[7] = (scales[11] & 0x0F) | ((scales[3] & 0xC0) >> 2)

	// Next 4 mins
	m[4] = (scales[8]&0xF0)>>4 | ((scales[4] & 0xC0) >> 2)
	m[5] = (scales[9]&0xF0)>>4 | ((scales[5] & 0xC0) >> 2)
	m[6] = (scales[10]&0xF0)>>4 | ((scales[6] & 0xC0) >> 2)
	m[7] = (scales[11]&0xF0)>>4 | ((scales[7] & 0xC0) >> 2)

	fmt.Printf("DEBUG Q4K: sc=%v, m=%v\n", sc, m)

	// Dequantize 256 values (8 groups of 32)
	for j := 0; j < 8; j++ {
		dVal := d * float32(sc[j])
		mVal := dmin * float32(m[j])

		fmt.Printf("DEBUG Q4K: group %d: dVal=%.9f, mVal=%.9f\n", j, dVal, mVal)

		// Each group has 32 values (16 bytes of packed 4-bit quants)
		for k := 0; k < 16; k++ {
			idx := j*16 + k
			q1 := qs[idx] & 0x0F
			q2 := (qs[idx] & 0xF0) >> 4

			out[j*32+k] = dVal*float32(q1) - mVal
			out[j*32+k+16] = dVal*float32(q2) - mVal
		}
	}

	return out
}

// TestQ4K_RealWeights_Mistral tests Q4K with actual Mistral weights
func TestQ4K_RealWeights_Mistral(t *testing.T) {
	// This test requires loading actual model weights
	// For now, create a realistic Q4K block based on observed values

	ctx := NewContext()
	defer ctx.Free()

	// Create Q4K block with realistic Mistral parameters
	// From logs: blk.0.attn_q.weight has d~0.000009, max scale~0.000093
	blockSize := 144
	block := make([]byte, blockSize)

	// d = 0.00001 (observed from Mistral). 1e-5 is subnormal in F16.
	// 0x00A8 represents approx 1.001e-5
	binary.LittleEndian.PutUint16(block[0:2], 0x00A8)

	// dmin = 0.000001. 1e-6.
	// 1e-6 * 2^14 * 1024 = 16.7 -> 17 -> 0x0011
	binary.LittleEndian.PutUint16(block[2:4], 0x0011)

	// Set scales to small values (like real Mistral)
	block[4] = 10 // sc[0] = 10 (~6-bit value)
	block[5] = 15 // sc[1] = 15
	block[6] = 8  // sc[2] = 8
	block[7] = 12 // sc[3] = 12

	// Fill quants with pattern
	for i := 16; i < 144; i++ {
		block[i] = uint8((i - 16) % 16) // Pattern 0-15
	}

	// CPU Reference dequantization
	cpuResult := DequantizeQ4K_Reference(block)

	// GPU dequantization via matmul
	// Create Q4K weight matrix (1 block = 256 values)
	weight, err := ctx.NewQ4KTensor(1, 256)
	if err != nil {
		t.Fatalf("Failed to create Q4K tensor: %v", err)
	}
	if err := weight.LoadRaw(block); err != nil {
		t.Fatalf("Failed to load Q4K data: %v", err)
	}

	// Create input (all ones)
	input := ctx.NewTensorPooled(1, 256)
	inputData := make([]float32, 256)
	for i := range inputData {
		inputData[i] = 1.0
	}
	input.LoadFrom(inputData)

	// Run GPU kernel
	output := ctx.NewTensorPooled(1, 1)
	input.LinearInto(weight, output, 1.0)
	ctx.Synchronize()

	gpuResult := output.ToHost()

	// Expected: dot product of cpuResult with all-ones input
	expectedDot := float32(0)
	for _, v := range cpuResult {
		expectedDot += v
	}

	t.Logf("CPU dequant sample [0:5]: %v", cpuResult[:5])
	t.Logf("CPU dequant sum: %.6f", expectedDot)
	t.Logf("GPU result: %.6f", gpuResult[0])
	t.Logf("Difference: %.6f", gpuResult[0]-expectedDot)

	// Check if they match within reasonable tolerance
	diff := math.Abs(float64(gpuResult[0] - expectedDot))
	relativeDiff := diff / math.Abs(float64(expectedDot))

	if relativeDiff > 0.01 { // 1% tolerance
		t.Errorf("GPU/CPU mismatch: GPU=%.6f, CPU=%.6f, relative diff=%.2f%%",
			gpuResult[0], expectedDot, relativeDiff*100)
	}
}

// TestQ6K_Structure checks if DequantizeQ6K produces non-zero output for known subnormal d
func TestQ6K_Structure(t *testing.T) {
	// Block size 210 bytes
	block := make([]byte, 210)

	// d = 0.00001 (subnormal approx 0x00A8)
	binary.LittleEndian.PutUint16(block[208:210], 0x00A8)

	// Set scales (16 bytes at 192)
	// Set scale[0] = 10
	block[192] = 10

	// Set quants
	// qs (128 bytes) at 0
	// qh (64 bytes) at 128

	// Set weight 0:
	// low 4 bits (qs[0] low nibble) = 7
	// high 2 bits (qh[0] bits 0,1) = 1
	// q = (1 << 4) | 7 = 16 + 7 = 23
	// val = d * sc * (q - 32)
	// val = 1e-5 * 10 * (23 - 32) = 1e-4 * (-9) = -0.0009

	block[0] = 7   // qs[0] low nibble
	block[128] = 1 // qh[0] low 2 bits

	data := gguf.DequantizeQ6K(block, 256)

	val0 := data[0]
	t.Logf("Q6K[0] = %e", val0)

	// Expect approx -0.0009
	if math.Abs(float64(val0)-(-0.0009)) > 0.0001 {
		t.Errorf("Q6K failed: got %e, want ~ -0.0009", val0)
	}
}
