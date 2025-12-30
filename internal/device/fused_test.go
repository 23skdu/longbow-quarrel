//go:build darwin && metal

package device

import (
	"math"
	"testing"
)

func TestRMSNormLinearQ4K(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	M := 1
	K := 256 // Must be multiple of 256 for Q4K
	N := 1
	eps := float32(1e-5)
	scale := float32(1.0)

	// 1. Prepare Input (all 2s)
	inputData := make([]float32, M*K)
	for i := range inputData {
		inputData[i] = 2.0
	}
	tIn := ctx.NewTensorFP32(M, K)
	tIn.LoadFrom(inputData)

	// 2. Prepare Norm Weight (all 0.5s)
	// Normalized(2.0) = 1.0. 1.0 * 0.5 = 0.5.
	normWeightData := make([]float32, K)
	for i := range normWeightData {
		normWeightData[i] = 0.5
	}
	tNormW := ctx.NewTensorFP32(1, K) // F32
	tNormW.LoadFrom(normWeightData)

	// 3. Prepare Q4K Weights (Expected weight = 1.0)
	// d = 1.0, dmin = 0.0
	// sc = 1, m = 0
	// v = 1
	// weight = 1.0 * 1 - 0.0 * 0 = 1.0
	q4kBlock := make([]byte, 144)
	d_f16 := Float32ToFloat16(1.0)
	dmin_f16 := Float32ToFloat16(0.0)
	q4kBlock[0] = byte(d_f16 & 0xFF)
	q4kBlock[1] = byte(d_f16 >> 8)
	q4kBlock[2] = byte(dmin_f16 & 0xFF)
	q4kBlock[3] = byte(dmin_f16 >> 8)
	
	// scales (12 bytes)
	// sc[j] = scales[j] & 63. set to 1.
	for i := 4; i < 8; i++ {
		q4kBlock[i] = 1 
	}
	// m[j] = scales[j+4] & 63. set to 0.
	for i := 8; i < 12; i++ {
		q4kBlock[i] = 0
	}
	// bits for sc[4..7] and m[4..7] also in 12-15? 
	// For simplicity, let's just use first 4 sub-blocks or ensure sc[j]=1 m[j]=0 for all.
	// Actually let's just set all to 1 (sc=1, m=1) for simplicity if easier.
	for i := 4; i < 16; i++ {
		q4kBlock[i] = 1 // sc=1, m=1 (bits 6,7 of first 8 bytes and bytes 8-11)
	}
	// qs (128 bytes)
	// v0 = 1, v1 = 1 -> byte = 0x11
	for i := 16; i < 144; i++ {
		q4kBlock[i] = 0x11
	}

	tWeight := ctx.NewQ4KTensor(N, K)
	tWeight.LoadFromRaw(q4kBlock)

	// 4. Run Fused
	tFused := tIn.RMSNormLinearQ4K(tNormW, tWeight, eps, scale)
	fusedRes := tFused.ToHost()

	// 5. Run Sequential for Reference
	tNormed := ctx.NewTensorPooled(M, K)
	tIn.RMSNormFP32_ToF16_Into(tNormW, eps, tNormed)
	tSeq := tNormed.Linear(tWeight)
	seqRes := tSeq.ToHost()

	// Expected calculation:
	// RMS = sqrt(mean(input^2) + eps) = sqrt(4 + 1e-5) ~ 2.0000025
	// NormVal = (2.0 / 2.0000025) * 0.5 ~ 0.499999
	// Sum(NormVal * weight) = 256 * 0.499999 * 1.0 ~ 127.999
	
	t.Logf("Fused results: %v", fusedRes)
	t.Logf("Sequential results: %v", seqRes)

	if math.Abs(float64(fusedRes[0]-seqRes[0])) > 1e-1 {
		t.Errorf("Mismatch between fused and sequential: got %f, want %f", fusedRes[0], seqRes[0])
	}
}

func TestSwiGLULinearQ4K(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	M := 1
	K := 256
	N := 1
	scale := float32(1.0)

	// 1. Prepare Gate and Up inputs
	gateData := make([]float32, M*K)
	upData := make([]float32, M*K)
	for i := range gateData {
		gateData[i] = 1.0 // Silu(1.0) ~ 0.731
		upData[i] = 1.0
	}
	tGate := ctx.NewTensor(M, K)
	tGate.LoadFrom(gateData)
	tUp := ctx.NewTensor(M, K)
	tUp.LoadFrom(upData)

	// 2. Prepare Q4K Weights (Expected weight = 1.0)
	q4kBlock := make([]byte, 144)
	d_f16 := Float32ToFloat16(1.0)
	dmin_f16 := Float32ToFloat16(0.0)
	q4kBlock[0] = byte(d_f16 & 0xFF)
	q4kBlock[1] = byte(d_f16 >> 8)
	q4kBlock[2] = byte(dmin_f16 & 0xFF)
	q4kBlock[3] = byte(dmin_f16 >> 8)
	for i := 4; i < 16; i++ { q4kBlock[i] = 1 } // sc=1, m=1
	for i := 16; i < 144; i++ { q4kBlock[i] = 0x11 } // v=1

	tWeight := ctx.NewQ4KTensor(N, K)
	tWeight.LoadFromRaw(q4kBlock)

	// 3. Run Fused
	tFused := tGate.SwiGLULinearQ4K(tUp, tWeight, scale)
	fusedRes := tFused.ToHost()

	// 4. Run Sequential
	tSwi := tGate.SwiGLU(tUp)
	tSeq := tSwi.Linear(tWeight)
	seqRes := tSeq.ToHost()

	t.Logf("Fused results: %v", fusedRes)
	t.Logf("Sequential results: %v", seqRes)

	if math.Abs(float64(fusedRes[0]-seqRes[0])) > 1e-1 {
		t.Errorf("Mismatch between fused and sequential: got %f, want %f", fusedRes[0], seqRes[0])
	}
}
