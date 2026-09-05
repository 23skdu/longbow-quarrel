package gguf

import (
	"fmt"
	"testing"
)

type BenchmarkResult struct {
	Name          string
	NumElements   int
	NumRuns       int
	TotalTimeNS   int64
	TimePerRunNS  int64
	OpsPerSec     float64
	MBPerSec      float64
}


func BenchmarkQuantizeQ4K_256(b *testing.B) {
	src := make([]float32, 256)
	for i := range src {
		src[i] = float32(i) * 0.1
	}

	b.ResetTimer()
	for b.Loop() {
		_, err := QuantizeWeightsToQ4K(src, len(src))
		if err != nil {
			b.Fatalf("QuantizeWeightsToQ4K failed: %v", err)
		}
	}
}

func BenchmarkQuantizeQ4K_512(b *testing.B) {
	src := make([]float32, 512)
	for i := range src {
		src[i] = float32(i) * 0.1
	}

	b.ResetTimer()
	for b.Loop() {
		_, err := QuantizeWeightsToQ4K(src, len(src))
		if err != nil {
			b.Fatalf("QuantizeWeightsToQ4K failed: %v", err)
		}
	}
}

func BenchmarkQuantizeQ4K_1024(b *testing.B) {
	src := make([]float32, 1024)
	for i := range src {
		src[i] = float32(i) * 0.1
	}

	b.ResetTimer()
	for b.Loop() {
		_, err := QuantizeWeightsToQ4K(src, len(src))
		if err != nil {
			b.Fatalf("QuantizeWeightsToQ4K failed: %v", err)
		}
	}
}

func BenchmarkQuantizeQ4K_2048(b *testing.B) {
	src := make([]float32, 2048)
	for i := range src {
		src[i] = float32(i) * 0.1
	}

	b.ResetTimer()
	for b.Loop() {
		_, err := QuantizeWeightsToQ4K(src, len(src))
		if err != nil {
			b.Fatalf("QuantizeWeightsToQ4K failed: %v", err)
		}
	}
}

func BenchmarkQuantizeQ4K_4096(b *testing.B) {
	src := make([]float32, 4096)
	for i := range src {
		src[i] = float32(i) * 0.1
	}

	b.ResetTimer()
	for b.Loop() {
		_, err := QuantizeWeightsToQ4K(src, len(src))
		if err != nil {
			b.Fatalf("QuantizeWeightsToQ4K failed: %v", err)
		}
	}
}

func BenchmarkDequantizeQ4K_256(b *testing.B) {
	src := make([]float32, 256)
	for i := range src {
		src[i] = float32(i) * 0.1
	}

	data, err := QuantizeWeightsToQ4K(src, len(src))
	if err != nil {
		b.Fatalf("QuantizeWeightsToQ4K failed: %v", err)
	}

	b.ResetTimer()
	for b.Loop() {
		_, err := DequantizeWeightsFromQ4K(data, 1, 256)
		if err != nil {
			b.Fatalf("DequantizeWeightsFromQ4K failed: %v", err)
		}
	}
}

func BenchmarkDequantizeQ4K_512(b *testing.B) {
	src := make([]float32, 512)
	for i := range src {
		src[i] = float32(i) * 0.1
	}

	data, err := QuantizeWeightsToQ4K(src, len(src))
	if err != nil {
		b.Fatalf("QuantizeWeightsToQ4K failed: %v", err)
	}

	b.ResetTimer()
	for b.Loop() {
		_, err := DequantizeWeightsFromQ4K(data, 1, 512)
		if err != nil {
			b.Fatalf("DequantizeWeightsFromQ4K failed: %v", err)
		}
	}
}

func BenchmarkDequantizeQ4K_1024(b *testing.B) {
	src := make([]float32, 1024)
	for i := range src {
		src[i] = float32(i) * 0.1
	}

	data, err := QuantizeWeightsToQ4K(src, len(src))
	if err != nil {
		b.Fatalf("QuantizeWeightsToQ4K failed: %v", err)
	}

	b.ResetTimer()
	for b.Loop() {
		_, err := DequantizeWeightsFromQ4K(data, 1, 1024)
		if err != nil {
			b.Fatalf("DequantizeWeightsFromQ4K failed: %v", err)
		}
	}
}

func BenchmarkDequantizeQ4K_2048(b *testing.B) {
	src := make([]float32, 2048)
	for i := range src {
		src[i] = float32(i) * 0.1
	}

	data, err := QuantizeWeightsToQ4K(src, len(src))
	if err != nil {
		b.Fatalf("QuantizeWeightsToQ4K failed: %v", err)
	}

	b.ResetTimer()
	for b.Loop() {
		_, err := DequantizeWeightsFromQ4K(data, 1, 2048)
		if err != nil {
			b.Fatalf("DequantizeWeightsFromQ4K failed: %v", err)
		}
	}
}

func BenchmarkDequantizeQ4K_4096(b *testing.B) {
	src := make([]float32, 4096)
	for i := range src {
		src[i] = float32(i) * 0.1
	}

	data, err := QuantizeWeightsToQ4K(src, len(src))
	if err != nil {
		b.Fatalf("QuantizeWeightsToQ4K failed: %v", err)
	}

	b.ResetTimer()
	for b.Loop() {
		_, err := DequantizeWeightsFromQ4K(data, 1, 4096)
		if err != nil {
			b.Fatalf("DequantizeWeightsFromQ4K failed: %v", err)
		}
	}
}

func BenchmarkQuantizeQ4K_RoundTrip(b *testing.B) {
	sizes := []int{256, 512, 1024, 2048, 4096}
	
	for _, size := range sizes {
		b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
			src := make([]float32, size)
			for i := range src {
				src[i] = float32(i) * 0.1
			}

			b.ResetTimer()
			for b.Loop() {
				data, err := QuantizeWeightsToQ4K(src, len(src))
				if err != nil {
					b.Fatalf("QuantizeWeightsToQ4K failed: %v", err)
				}
				_, err = DequantizeWeightsFromQ4K(data, 1, size)
				if err != nil {
					b.Fatalf("DequantizeWeightsFromQ4K failed: %v", err)
				}
			}
		})
	}
}

func BenchmarkTurboQuant_PolarQuant(b *testing.B) {
	blockSize := 256
	headDim := 256
	qjlRows := 64

	src := make([]float32, blockSize)
	for i := range src {
		src[i] = float32(i) * 0.1
	}

	// Identity rotation matrix
	rot := make([]float32, headDim*headDim)
	for i := 0; i < headDim; i++ {
		rot[i*headDim+i] = 1.0
	}

	// Simple sign matrix
	qjl := make([]float32, qjlRows*headDim)
	for i := range qjl {
		if i%2 == 0 {
			qjl[i] = 1.0
		} else {
			qjl[i] = -1.0
		}
	}

	b.ResetTimer()
	for b.Loop() {
		_, _, _, err := PolarQuant(src, rot, blockSize, 4)
		if err != nil {
			b.Fatalf("PolarQuant failed: %v", err)
		}
	}
}

func BenchmarkTurboQuant_QJLTransform(b *testing.B) {
	blockSize := 256
	qjlRows := 64

	residual := make([]float32, blockSize)
	for i := range residual {
		residual[i] = float32(i%10) * 0.1
	}

	qjl := make([]float32, qjlRows*blockSize)
	for i := range qjl {
		if i%2 == 0 {
			qjl[i] = 1.0
		} else {
			qjl[i] = -1.0
		}
	}

	b.ResetTimer()
	for b.Loop() {
		_, _, err := QJLTransform(residual, qjl, qjlRows, blockSize)
		if err != nil {
			b.Fatalf("QJLTransform failed: %v", err)
		}
	}
}

func BenchmarkTurboQuant_Full(b *testing.B) {
	blockSize := 256
	headDim := 256
	qjlRows := 64

	src := make([]float32, blockSize)
	for i := range src {
		src[i] = float32(i) * 0.1
	}

	rot := make([]float32, headDim*headDim)
	for i := 0; i < headDim; i++ {
		rot[i*headDim+i] = 1.0
	}

	qjl := make([]float32, qjlRows*headDim)
	for i := range qjl {
		if i%2 == 0 {
			qjl[i] = 1.0
		} else {
			qjl[i] = -1.0
		}
	}

	b.ResetTimer()
	for b.Loop() {
		_, err := QuantizeTurboQuant(src, rot, qjl, blockSize, 4)
		if err != nil {
			b.Fatalf("QuantizeTurboQuant failed: %v", err)
		}
	}
}
