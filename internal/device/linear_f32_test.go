//go:build darwin && metal

package device

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func TestLinearF16_F32_Kernel(t *testing.T) {
	ctx := NewContext()
	// if ctx == nil { t.Skip(...) } ? Assuming NewContext panics or returns valid.
	if ctx == nil {
		t.Skip("Metal context nil")
	}
	defer ctx.Free()

	// Dimensions matches SmolLM2 FFN: 576 -> 1536 (Gate/Up)
	M := 1
	N := 1536 // Out
	K := 576  // In

	// Data
	weightsF32 := make([]float32, N*K) // Transposed? [N, K] in Go wrapper
	inputF32 := make([]float32, M*K)

	// Init with determinism
	rnd := rand.New(rand.NewSource(42))
	for i := range weightsF32 {
		weightsF32[i] = (rnd.Float32() - 0.5) * 0.1
	}
	for i := range inputF32 {
		inputF32[i] = (rnd.Float32() - 0.5) * 2.0
	}

	// CPU Reference
	refOut := make([]float32, M*N)
	for r := 0; r < M; r++ {
		for c := 0; c < N; c++ {
			sum := float32(0)
			for k := 0; k < K; k++ {
				// Matrix Mul: Output[r, c] += Input[r, k] * Weight[c, k]
				// Wait. Weight shape in metal.go is [Rows, Cols].
				// Rows = OutputDim (N). Cols = InputDim (K).
				// So Weight[c, k].
				w := weightsF32[c*K+k]
				in := inputF32[r*K+k]
				sum += w * in
			}
			refOut[r*N+c] = sum
		}
	}

	// Metal Tensors
	// Weights F16
	wT := ctx.NewTensor(N, K)
	wT.LoadFrom(weightsF32) // Converts to F16

	// Input F32 (Use Scratch/NewTensorFP32Pooled logic simulation)
	// We need an F32 tensor.
	// NewTensor allocates F16 by default.
	// Need NewTensor with DataTypeF32.
	// We don't have exposed NewTensorFrom... we have NewLayerScratch which uses internal newT.
	// But `Tensor` struct has `datatype`.
	// We can manually create one? Or use `NewTensorFP32Pooled`?
	// `NewTensorFP32Pooled` is available on `Context`.

	inT := ctx.NewTensorFP32Pooled(M, K)
	inT.LoadFrom(inputF32) // LoadF32 to F32 keeps F32?
	// Use LoadFromRaw for exact load to F32 buffer?
	// LoadFrom checks DataType.

	// Create Output F32
	outT := ctx.NewTensorFP32Pooled(M, N)

	// Run LinearF32_Into (which calls Metal_MatMul_F16_F32)
	// func (t *Tensor) LinearF32_Into(weight *Tensor, out *Tensor)
	// t is Input.
	inT.LinearF32_Into(wT, outT, 1.0)
	ctx.Synchronize()

	gpuOut := outT.ToHost()

	// Verify
	errs := 0
	for i := 0; i < len(refOut); i++ {
		diff := math.Abs(float64(refOut[i] - gpuOut[i]))
		if diff > 0.1 { // F16 precision loss might be large for sums? 0.1 is generous.
			if errs < 10 {
				fmt.Printf("Mismatch [%d]: Ref %f vs GPU %f\n", i, refOut[i], gpuOut[i])
			}
			errs++
		}
	}

	if errs > 0 {
		t.Fatalf("Failed with %d mismatches", errs)
	}
	fmt.Printf("Passed with 0 mismatches. Max Ref: %f, Max GPU: %f\n", refOut[0], gpuOut[0])
}
