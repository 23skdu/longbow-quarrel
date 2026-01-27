//go:build darwin && metal

package device

import (
	"testing"
)

func BenchmarkLinear_Q4K_F16(b *testing.B) {
	ctx := NewContext()
	defer ctx.Free()

	// Typical Layer Dimensions (Mistral 7B)
	// Hidden Dim: 4096
	// Inter Dim: 14336
	N := 4096
	K := 4096

	weight, _ := ctx.NewQ4KTensor(N, K)
	input := ctx.NewTensor(1, K)
	output := ctx.NewTensor(1, N)

	// Minimal data to avoid NaNs
	inData := make([]float32, K)
	for i := range inData {
		inData[i] = 1.0
	}
	input.LoadFrom(inData)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		input.LinearInto(weight, output, 1.0)
		ctx.Synchronize()
	}
}
