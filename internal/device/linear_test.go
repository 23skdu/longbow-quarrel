package device

import (
	"math"
	"math/rand"
	"testing"
)

// CPU reference for MatMul A * B^T
// A: [M, K], B: [N, K]. Out: [M, N]
func cpuMatMul(a []float32, b []float32, M, N, K int) []float32 {
	out := make([]float32, M*N)
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var sum float32 = 0
			for k := 0; k < K; k++ {
				valA := a[i*K+k]
				valB := b[j*K+k] // B is [N, K], row-major. B^T means dot product row(A) . row(B)
				sum += valA * valB
			}
			out[i*N+j] = sum
		}
	}
	return out
}

func TestLinearF16(t *testing.T) {
	ctx := NewContext()
	defer ctx.Free()

	M := 2 // Batch
	N := 4 // Out features
	K := 8 // In features

	// A: [M, K]
	aData := make([]float32, M*K)
	for i := range aData {
		aData[i] = rand.Float32() - 0.5
	}

	// B: [N, K] (Weights)
	bData := make([]float32, N*K)
	for i := range bData {
		bData[i] = rand.Float32() - 0.5
	}

	// CPU Result
	cpuOut := cpuMatMul(aData, bData, M, N, K)

	// Metal Result
	tensorA := ctx.NewTensor(M, K)
	tensorA.LoadFrom(aData)

	tensorB := ctx.NewTensor(N, K) // Weights
	tensorB.LoadFrom(bData)

	tensorOut := ctx.NewTensor(M, N)

	// Call LinearInto logic manually (emulating what LinearInto does)
	// We call the C function wrapper via Metal API or reuse LinearInto if possible?
	// LinearInto is on Tensor.
	// But LinearInto expects weights.dataType. NewTensor defaults to F16.
	// So tensorB.dataType == DataTypeF16.
	// tensorA.LinearInto(tensorB, tensorOut)
	
	// We need to implement LinearInto call or call it.
	// Since we are in `device` package, we can access Tensor methods.
	tensorA.LinearInto(tensorB, tensorOut)
	
	tensorA.ctx.Synchronize()
	
	metalOut := tensorOut.ToHost()
	
	// Compare
	for i, v := range metalOut {
		// FP16 precision tolerance
		if math.Abs(float64(v - cpuOut[i])) > 1e-2 {
			t.Errorf("Mismatch at %d: CPU %f, Metal %f (Diff: %f)", i, cpuOut[i], v, v-cpuOut[i])
			return 
		}
	}
}
