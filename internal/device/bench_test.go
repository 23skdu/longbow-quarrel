//go:build darwin && metal

package device

import (
	"math/rand"
	"testing"
)

func BenchmarkMetalMatMul_128(b *testing.B)  { benchmarkMatMul(b, 128, 128, 128) }
func BenchmarkMetalMatMul_512(b *testing.B)  { benchmarkMatMul(b, 512, 512, 512) }
func BenchmarkMetalMatMul_1024(b *testing.B) { benchmarkMatMul(b, 1024, 1024, 1024) }
func BenchmarkMetalMatMul_2048(b *testing.B) { benchmarkMatMul(b, 2048, 2048, 2048) }

func benchmarkMatMul(b *testing.B, M, K, N int) {
	ctx := NewContext()
	defer ctx.Free()

	aData := make([]float32, M*K)
	bData := make([]float32, K*N)
	for i := range aData {
		aData[i] = rand.Float32()
	}
	for i := range bData {
		bData[i] = rand.Float32()
	}

	/*
	   tA := ctx.NewTensor(M, K)
	   tA.LoadFrom(aData)
	   tB := ctx.NewTensor(K, N)
	   tB.LoadFrom(bData)

	   tA_ := tA
	*/

	tA := ctx.NewTensor(M, K)
	tA.LoadFrom(aData)
	tB := ctx.NewTensor(K, N)
	tB.LoadFrom(bData)

	// Warmup
	_ = tA.MatMul(tB)
	ctx.Synchronize()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tC := tA.MatMul(tB)
		// To measure pure dispatch + GPU time, we can sync partially or lazily.
		// But to prevent OOM or command buffer queue filling too much, we should sync occasionally or just let Metal handle it.
		// For accurate timing of "inference step", we usually want to wait for result or at least commit.
		// MatMul returns a new tensor.
		// If we don't use tC, it might be optimized out if we are not careful? No, CGO calls have side effects.
		// However, we are allocating new result memory every time. This is expensive.
		// Ideally we should reuse result buffer.
		// The current API allocates new result every time.
		// This benchmark measures Allocation + Dispatch.
		// It represents current unoptimized state.

		_ = tC
		// Forcing sync every N steps might be good, but here we just loop.
	}
	ctx.Synchronize()
}
