package simd

import (
	"fmt"
	"math/rand"
	"testing"
)

func BenchmarkQJLTransformSIMD(b *testing.B) {
	rows := 1
	cols := 256
	residual := make([]float32, cols)
	sign := make([]float32, rows*cols)
	for i := range residual {
		residual[i] = rand.Float32()
	}
	for i := range sign {
		sign[i] = rand.Float32()
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = QJLTransformSIMD(residual, sign, rows, cols)
	}
}

func BenchmarkEndToEndInferenceStep(b *testing.B) {
	// Simulate an inference step with a small model
	// 32 layers, heads=8, dim=128
	layers := 32
	heads := 8
	dim := 128
	totalElements := heads * dim
	
	input := make([]float32, totalElements)
	rot := make([]float32, dim*dim)
	qjl := make([]float32, 64*dim)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Mock storing all layers
		for l := 0; l < layers; l++ {
			for h := 0; h < heads; h++ {
				headData := input[h*dim : (h+1)*dim]
				q, _, res := PolarQuantSIMD(headData, rot, dim, 4)
				_, _ = QJLTransformSIMD(res, qjl, 64, dim)
				_ = q
			}
		}
	}
	
	throughput := float64(b.N) * float64(layers) / b.Elapsed().Seconds()
	fmt.Printf("Throughput: %.2f layers/sec (single thread)\n", throughput)
}
