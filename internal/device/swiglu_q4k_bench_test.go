//go:build darwin && metal

package device

import (
	"testing"
)

func BenchmarkSwiGLULinear_Q4K_F16(b *testing.B) {
	ctx := NewContext()
	defer ctx.Free()

	// Typical FFN Layer Dimensions (Mistral 7B)
	// Hidden Dim: 4096
	// Inter Dim: 14336
	N := 14336
	K := 4096

	weight, _ := ctx.NewQ4KTensor(N, K)
	gate := ctx.NewTensor(1, K)
	up := ctx.NewTensor(1, K)
	output := ctx.NewTensor(1, N)

	// Minimal data
	inData := make([]float32, K)
	for i := range inData {
		inData[i] = 1.0
	}
	gate.LoadFrom(inData)
	up.LoadFrom(inData)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gate.SwiGLULinearIntoQ4K(up, weight, output, 1.0)
		ctx.Synchronize()
	}
}
