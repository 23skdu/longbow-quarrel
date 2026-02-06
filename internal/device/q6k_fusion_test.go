//go:build darwin && metal

package device

import (
	"fmt"
	"math"
	"testing"
)

func TestQ6K_Fusion_Correctness(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	// Dimensions
	dimIn := 256
	dimOut := 128
	eps := float32(1e-5)
	scale := float32(1.0)

	// 1. Create Input Tensor (FP16)
	input := ctx.NewTensor(1, dimIn)
	inputData := make([]float32, dimIn)
	for i := 0; i < dimIn; i++ {
		inputData[i] = float32(i) / 100.0
	}
	input.LoadFrom(inputData)

	// 2. Create Weights (Q6_K)
	weightData := make([]float32, dimOut*dimIn)
	for i := 0; i < dimOut*dimIn; i++ {
		weightData[i] = float32(i%10) / 10.0
	}
	weights, err := ctx.NewQ6KTensor(dimOut, dimIn)
	if err != nil {
		t.Fatalf("failed to create Q6_K tensor: %v", err)
	}
	weights.LoadFrom(weightData)

	// 3. Create Norm Weights (FP32)
	normWeightData := make([]float32, dimIn)
	for i := 0; i < dimIn; i++ {
		normWeightData[i] = 1.0 + float32(i)/1000.0
	}
	normWeights := ctx.NewTensorFP32(1, dimIn)
	normWeights.LoadFromF32(normWeightData)

	t.Run("RMSNormLinear_Q6K", func(t *testing.T) {
		// Reference: Non-fused
		normed := ctx.NewTensor(1, dimIn)
		input.RMSNormFP32_ToF16_Into(normWeights, eps, normed)

		refRes := ctx.NewTensor(1, dimOut)
		normed.LinearInto(weights, refRes, scale)
		ctx.Synchronize()

		refData := refRes.ToHost()

		// Fused
		fusedRes := ctx.NewTensor(1, dimOut)
		input.RMSNormLinearIntoQ6K(normWeights, weights, fusedRes, eps, scale)
		ctx.Synchronize()

		fusedData := fusedRes.ToHost()

		// Compare
		for i := 0; i < dimOut; i++ {
			diff := math.Abs(float64(refData[i] - fusedData[i]))
			if diff > 1e-3 {
				t.Errorf("Mismatch at index %d: ref=%f, fused=%f, diff=%f", i, refData[i], fusedData[i], diff)
				break
			}
		}
		fmt.Printf("RMSNormLinear_Q6K ref[0]=%f, fused[0]=%f\n", refData[0], fusedData[0])
	})

	t.Run("SwiGLULinear_Q6K", func(t *testing.T) {
		// Dimensions for SwiGLU: up, gate inputs
		hiddenDim := 128

		gateIn := ctx.NewTensor(1, hiddenDim)
		upIn := ctx.NewTensor(1, hiddenDim)

		gData := make([]float32, hiddenDim)
		uData := make([]float32, hiddenDim)
		for i := 0; i < hiddenDim; i++ {
			gData[i] = float32(i) / 50.0
			uData[i] = float32(i) / 60.0
		}
		gateIn.LoadFrom(gData)
		upIn.LoadFrom(uData)

		// Down weights (Q6_K)
		downWeights, err := ctx.NewQ6KTensor(dimOut, hiddenDim)
		if err != nil {
			t.Fatalf("failed to create Q6_K tensor: %v", err)
		}
		dwData := make([]float32, dimOut*hiddenDim)
		for i := 0; i < dimOut*hiddenDim; i++ {
			dwData[i] = float32(i%7) / 10.0
		}
		downWeights.LoadFrom(dwData)

		// Reference: Non-fused
		swiOut, _ := upIn.SwiGLU(gateIn)
		refRes := ctx.NewTensor(1, dimOut)
		swiOut.LinearInto(downWeights, refRes, scale)
		ctx.Synchronize()

		refData := refRes.ToHost()

		// Fused
		fusedRes := ctx.NewTensor(1, dimOut)
		gateIn.SwiGLULinearIntoQ6K(upIn, downWeights, fusedRes, scale)
		ctx.Synchronize()

		fusedData := fusedRes.ToHost()

		// Compare
		for i := 0; i < dimOut; i++ {
			diff := math.Abs(float64(refData[i] - fusedData[i]))
			if diff > 1e-3 {
				t.Fatalf("Mismatch at index %d: ref=%f, fused=%f, diff=%f", i, refData[i], fusedData[i], diff)
			}
		}
		fmt.Printf("SwiGLULinear_Q6K ref[0]=%f, fused[0]=%f\n", refData[0], fusedData[0])
	})
}

func BenchmarkQ6K_Fusion(b *testing.B) {
	ctx := NewContext()
	defer ctx.Free()

	dimIn := 4096
	dimOut := 4096
	hiddenDim := 11008 // Standard Llama-2-7B size
	eps := float32(1e-5)
	scale := float32(1.0)

	input := ctx.NewTensor(1, dimIn)
	weights, _ := ctx.NewQ6KTensor(dimOut, dimIn)
	normWeights := ctx.NewTensorFP32(1, dimIn)

	gateIn := ctx.NewTensor(1, hiddenDim)
	upIn := ctx.NewTensor(1, hiddenDim)
	downWeights, _ := ctx.NewQ6KTensor(dimOut, hiddenDim)

	b.Run("RMSNormLinear_Q6K_NonFused", func(b *testing.B) {
		normed := ctx.NewTensor(1, dimIn)
		res := ctx.NewTensor(1, dimOut)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			input.RMSNormFP32_ToF16_Into(normWeights, eps, normed)
			normed.LinearInto(weights, res, scale)
			ctx.Synchronize()
		}
	})

	b.Run("RMSNormLinear_Q6K_Fused", func(b *testing.B) {
		res := ctx.NewTensor(1, dimOut)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			input.RMSNormLinearIntoQ6K(normWeights, weights, res, eps, scale)
			ctx.Synchronize()
		}
	})

	b.Run("SwiGLULinear_Q6K_NonFused", func(b *testing.B) {
		res := ctx.NewTensor(1, dimOut)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			swiOut, _ := upIn.SwiGLU(gateIn)
			swiOut.LinearInto(downWeights, res, scale)
			ctx.Synchronize()
			swiOut.ReturnToPool()
		}
	})

	b.Run("SwiGLULinear_Q6K_Fused", func(b *testing.B) {
		res := ctx.NewTensor(1, dimOut)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			gateIn.SwiGLULinearIntoQ6K(upIn, downWeights, res, scale)
			ctx.Synchronize()
		}
	})
}

func BenchmarkQ8_0(b *testing.B) {
	ctx := NewContext()
	defer ctx.Free()

	dimIn := 4096
	dimOut := 4096
	scale := float32(1.0)

	input := ctx.NewTensor(1, dimIn)
	weights, _ := ctx.NewQ8_0Tensor(dimOut, dimIn)
	res := ctx.NewTensor(1, dimOut)

	b.Run("Linear_Q8_0", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			input.LinearInto(weights, res, scale)
			ctx.Synchronize()
		}
	})
}
