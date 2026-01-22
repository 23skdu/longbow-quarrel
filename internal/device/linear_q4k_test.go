//go:build darwin && metal

package device

import (
	"math"
	"testing"
)

// TestLinear_Q4K_F16 verifies the Q4K -> F16 Linear kernel
// We will manually construct a small Q4K tensor and check projection.
func TestLinear_Q4K_F16_Verification(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	// Dimensions
	M := 1   // Batch
	N := 256 // Output Dim (1 row)
	K := 256 // Input Dim (1 block)

	// Access to private helpers or robust manual construction?
	// We need to create a valid Q4K blob.
	// 256 weights -> 1 block -> 144 bytes.

	// Create Q4K Tensor
	// We'll use NewQ4KTensor which allocates buffer.
	// We need to write raw bytes to it.
	weight, err := ctx.NewQ4KTensor(N, K)
	if err != nil {
		t.Fatalf("Failed to create Q4K tensor: %v", err)
	}

	// Create Block Data
	// d=1.0, dmin=0.0. Scales=0. Quants=0.
	// Effectively all weights = 0? No, decoding:
	// d_val = d * sc
	// w = d_val * q - m_val

	// Let's try to set all weights to 1.0.
	// d=1.0, dmin=0.0.
	// sc=1.0 (requires packing 6-bit scales).
	// q=1 (requires packing 4-bit quants).

	// Easier: set weights such that sum is predictable.
	// Block Bytes:
	// 0-1: d (FP16)
	// 2-3: dmin (FP16)
	// 4-15: scales (12 bytes)
	// 16-143: qs (128 bytes)

	blockSize := 144
	totalBytes := (N * K / 256) * blockSize
	rawData := make([]byte, totalBytes)

	// For each block (there are N blocks, since K=256, 1 block per row)
	for r := 0; r < N; r++ {
		offset := r * blockSize

		// d = 1.0
		binaryOps := make([]byte, 2)
		// 1.0 in FP16 is 0x3C00
		dF16 := uint16(0x3C00)
		binaryOps[0] = byte(dF16 & 0xFF)
		binaryOps[1] = byte(dF16 >> 8)
		copy(rawData[offset:offset+2], binaryOps)

		// dmin = 0.0 -> 0x0000

		// Scales: 12 bytes.
		// We want scale = 1.0 ?? No, scale is quantized 6-bit.
		// Layout: sc[j] & 63.
		// If we set all bytes to 0, scale = 0?
		// sc[j] = scales[j] & 63.
		// If scales[...] = 0, sc=0. d_val=0. w=0.

		// We want something non-zero.
		// If we put 1 in scales?
		// Set all scales to 1. (6-bit 1).
		// scales[0..11] = 0x01?
		// for j=0..3: sc=1. m=1.
		// Let's set bytes 4..15 to 0. except...
		// We want w = 1.0.

		// Alternative: Use a known pattern check?
		// Or just load random data and expect consistency?
		// Consistency is hard without reference impl in Go.
		//
		// Valid Q4K Check:
		// If we set K=256. Input x=[1,1,...].
		// Weight row 0: [1,1,...]. Dot = 256.
		//
		// How to encode 1.0 in Q4K?
		// d=1.0. dmin=0.0.
		// sc=1 (normalized?). No, sc is multiplier.
		// d_val = d * sc.
		// w = d_val * q.
		// If sc=1, q=1. w=1.
		// sc depends on scales encoding.
		// sc[j] = scales[j] & 63.
		// Set scales[0..11] such that all sc=1, m=0?
		// scales[0..3] = 1. (sc=1, m=?)
		// m[0] = scales[4] & 63.
		// Set scales[0..3] = 1. scales[4..7] = 0.
		// Then sc[0..3]=1, m[0..3]=0.
		// What about sc[4..7]?
		// sc[4] = (scales[8] & 0xF) | ...
		// This is complicated to pack manually.

		// Let's assume the kernel works if we get NON-ZERO output for NON-ZERO input.
		// And specifically check for NaNs/Infs.

		// Better: Construct ONE block with known values.
		// d=1.0 (0x3C00). dmin=0.0.
		// Scales: All 0 => sc=0?
		// If scales bytes are all 0:
		// sc[0..3] = 0. m[0..3]=0.
		// sc[4..7] = 0. m[4..7]=0.
		// Then w=0.
		//
		// Set scales[0] = 1.
		// sc[0]=1. m[0]=0.
		// This affects first 32 weights.
		// qs[0..15] (first 32 weights).
		// Set qs[0] = 0x11 (q=1 for w0, q=1 for w1).
		// w = 1.0 * 1.0 * 1.0 - 0 = 1.0.
		// So first 2 weights are 1.0.
		// Input first 2 elements = 1.0.
		// Sum should be 2.0.

		// Let's do this.
		// Row 0 only.
		if r == 0 {
			// d=1.0
			copy(rawData[offset:offset+2], binaryOps)

			// Scales[0] = 1. (sc[0]=1, m[0]=0)
			rawData[offset+4] = 1

			// qs[0] = 0x11 -> w[0]=1, w[16]=?. No, wait.
			// Kernel logic:
			// k loops 0..15.
			// idx0 = i*256 + j*32 + k.
			// idx1 = idx0 + 16.
			// b = qs[j*16 + k].
			// w0 uses b & 0xF. (idx0)
			// w1 uses b >> 4. (idx1)

			// So qs[0] (k=0) controls w[0] and w[0+16] = w[16].
			// We want w[1] to be 1.0.
			// w[1] corresponds to k=1. idx0=1. idx1=17.
			// So we need qs[1] = 0x11 to make w[1]=1 and w[17]=1.

			rawData[offset+16] = 0x11 // w[0]=1, w[16]=1
			rawData[offset+17] = 0x11 // w[1]=1, w[17]=1
		}
	}

	weight.LoadRaw(rawData)

	// Create Input
	// K=256.
	inData := make([]float32, K)
	// Set in[0]=1, in[1]=1. Others 0.
	inData[0] = 1.0
	inData[1] = 1.0

	input := ctx.NewTensor(M, K) // F16 input
	input.LoadFrom(inData)

	// Output
	output := ctx.NewTensor(M, N)

	// Run Linear
	// output = input * weight^T
	// [1, 256] * [256, 256]^T = [1, 256]
	input.LinearInto(weight, output, 1.0)

	res := output.ToHost()

	// Row 0, should comply with our manual injection.
	// w[0]=1, w[1]=1. input[0]=1, input[1]=1.
	// dot = 1+1 = 2.

	val := float64(res[0])
	if math.Abs(val-2.0) > 0.1 {
		t.Errorf("Linear Q4K Failed: Row 0 expected ~2.0, got %f", val)
	} else {
		t.Logf("Linear Q4K Success: Row 0 = %f", val)
	}

	// Check other rows should be 0 (since we left d=0 or scale=0 there?)
	// Actually we zeroed the buffer initially by make([]byte).
	// d=0 -> w=0. Sum=0.
	if math.Abs(float64(res[1])) > 1e-3 {
		t.Errorf("Linear Q4K Failed: Row 1 expected 0.0, got %f", res[1])
	}
}
