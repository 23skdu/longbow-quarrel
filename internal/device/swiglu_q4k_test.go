//go:build darwin && metal

package device

import (
	"math"
	"testing"
)

func TestSwiGLU_Q4K_F16_Verification(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	// Dimensions
	M := 1   // Batch
	N := 256 // Output Dim (1 row)
	K := 256 // Input Dim (1 block)

	weight, _ := ctx.NewQ4KTensor(N, K)
	gate := ctx.NewTensor(M, K)
	up := ctx.NewTensor(M, K)
	output := ctx.NewTensor(M, N)

	// Set weights to 1.0 (d=1, dmin=0, sc[0]=1, qs[0]=0x11)
	binaryOps := make([]byte, 2)
	dF16 := uint16(0x3C00) // 1.0
	binaryOps[0] = byte(dF16 & 0xFF)
	binaryOps[1] = byte(dF16 >> 8)

	blockSize := 144
	rawData := make([]byte, N*blockSize)
	for r := 0; r < N; r++ {
		offset := r * blockSize
		if r == 0 {
			copy(rawData[offset:offset+2], binaryOps)
			rawData[offset+4] = 1     // sc[0]=1, m[0]=0
			rawData[offset+16] = 0x11 // w[0]=1, w[16]=1
		}
	}
	weight.LoadRaw(rawData)

	// Set input such that SwiGLU(gate, up) is known
	// SwiGLU(g, u) = u * (g / (1 + exp(-g)))
	// If g=10.0, sigmoid(10.0) ~= 1.0. SwiGLU(10.0, u) ~= u.
	// If u=2.0, SwiGLU(10.0, 2.0) = 2.0.

	gateData := make([]float32, K)
	upData := make([]float32, K)
	// Only first 2 elements contribute (since w[0]=1, w[16]=1, others are 0)
	gateData[0] = 10.0
	upData[0] = 2.0
	gateData[16] = 10.0
	upData[16] = 2.0

	gate.LoadFrom(gateData)
	up.LoadFrom(upData)

	// Run SwiGLU Linear
	gate.SwiGLULinearIntoQ4K(up, weight, output, 1.0)

	res := output.ToHost()

	// dot = w[0]*SwiGLU(10,2) + w[16]*SwiGLU(10,2)
	// dot = 1 * 20.0 + 1 * 20.0 = 40.0

	val := float64(res[0])
	if math.Abs(val-40.0) > 1.0 {
		t.Errorf("SwiGLU Q4K Failed: Row 0 expected ~40.0, got %f", val)
	} else {
		t.Logf("SwiGLU Q4K Success: Row 0 = %f", val)
	}
}
