package simd

import (
	"fmt"
	"testing"
	"time"
)

func BenchmarkTurboQuant_PolarQuant_AVX2(b *testing.B) {
	headDim := 256
	blockSize := 256

	src := make([]float32, blockSize)
	for i := range src {
		src[i] = float32(i) * 0.1
	}

	rot := make([]float32, headDim*headDim)
	for i := 0; i < headDim; i++ {
		rot[i*headDim+i] = 1.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _, err := TurboQuantPolarQuant(src, rot, blockSize)
		if err != nil {
			b.Fatalf("TurboQuantPolarQuant failed: %v", err)
		}
	}
}

func BenchmarkTurboQuant_PolarQuant_1D(b *testing.B) {
	sizes := []int{256, 512, 1024, 2048, 4096}
	headDim := 256

	for _, size := range sizes {
		b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
			src := make([]float32, size)
			for i := range src {
				src[i] = float32(i) * 0.1
			}

			rot := make([]float32, headDim*headDim)
			for i := 0; i < headDim; i++ {
				rot[i*headDim+i] = 1.0
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _, _, err := TurboQuantPolarQuant(src, rot, headDim)
				if err != nil {
					b.Fatalf("TurboQuantPolarQuant failed: %v", err)
				}
			}
		})
	}
}

func BenchmarkTurboQuant_QJL_AVX2(b *testing.B) {
	cols := 256
	rows := 64

	residual := make([]float32, cols)
	for i := range residual {
		residual[i] = float32(i%10) * 0.1
	}

	signs := make([]float32, rows*cols)
	for i := range signs {
		if i%2 == 0 {
			signs[i] = 1.0
		} else {
			signs[i] = -1.0
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, err := TurboQuantQJL(signs, residual, rows, cols)
		if err != nil {
			b.Fatalf("TurboQuantQJL failed: %v", err)
		}
	}
}

func BenchmarkTurboQuant_QJL_1D(b *testing.B) {
	colSizes := []int{256, 512, 1024, 2048}
	rowSizes := []int{16, 32, 64, 128}

	for _, cols := range colSizes {
		for _, rows := range rowSizes {
			b.Run(fmt.Sprintf("cols%d_rows%d", cols, rows), func(b *testing.B) {
				residual := make([]float32, cols)
				for i := range residual {
					residual[i] = float32(i%10) * 0.1
				}

				signs := make([]float32, rows*cols)
				for i := range signs {
					if i%2 == 0 {
						signs[i] = 1.0
					} else {
						signs[i] = -1.0
					}
				}

				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _, err := TurboQuantQJL(signs, residual, rows, cols)
					if err != nil {
						b.Fatalf("TurboQuantQJL failed: %v", err)
					}
				}
			})
		}
	}
}

func BenchmarkTurboQuant_Encode_1D(b *testing.B) {
	blockSizes := []int{256, 512, 1024, 2048}
	headDim := 256
	qjlRows := 64

	for _, blockSize := range blockSizes {
		b.Run(fmt.Sprintf("block_%d", blockSize), func(b *testing.B) {
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
			for i := 0; i < b.N; i++ {
				_, err := TurboQuantEncode(src, rot, qjl, blockSize, headDim)
				if err != nil {
					b.Fatalf("TurboQuantEncode failed: %v", err)
				}
			}
		})
	}
}

func BenchmarkTurboQuant_Decode_1D(b *testing.B) {
	blockSizes := []int{256, 512, 1024, 2048}
	headDim := 256
	qjlRows := 64

	for _, blockSize := range blockSizes {
		b.Run(fmt.Sprintf("block_%d", blockSize), func(b *testing.B) {
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

			data, err := TurboQuantEncode(src, rot, qjl, blockSize, headDim)
			if err != nil {
				b.Fatalf("TurboQuantEncode failed: %v", err)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := TurboQuantDecode(data, rot, qjl, blockSize, headDim)
				if err != nil {
					b.Fatalf("TurboQuantDecode failed: %v", err)
				}
			}
		})
	}
}

func BenchmarkTurboQuant_RoundTrip_1D(b *testing.B) {
	headDim := 256
	qjlRows := 64

	sizes := []int{256, 512, 1024, 2048, 4096}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
			src := make([]float32, size)
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
			for i := 0; i < b.N; i++ {
				data, err := TurboQuantEncode(src, rot, qjl, size, headDim)
				if err != nil {
					b.Fatalf("TurboQuantEncode failed: %v", err)
				}
				_, err = TurboQuantDecode(data, rot, qjl, size, headDim)
				if err != nil {
					b.Fatalf("TurboQuantDecode failed: %v", err)
				}
			}
		})
	}
}

func BenchmarkSIMD_MatMul_F32(b *testing.B) {
	matrixDims := []struct{ m, n, k int }{
		{512, 512, 512},
		{1024, 1024, 1024},
		{2048, 2048, 2048},
		{4096, 4096, 4096},
	}

	for _, dims := range matrixDims {
		b.Run(fmt.Sprintf("%dx%d_%dx%d", dims.m, dims.k, dims.k, dims.n), func(b *testing.B) {
			a := make([]float32, dims.m*dims.k)
			b := make([]float32, dims.k*dims.n)
			c := make([]float32, dims.m*dims.n)

			for i := range a {
				a[i] = float32(i%100) * 0.01
			}
			for i := range b {
				b[i] = float32(i%100) * 0.01
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				MatMul(a, b, c, dims.m, dims.n, dims.k)
			}
		})
	}
}

func BenchmarkBatch_MatMul_F32(b *testing.B) {
	batchSizes := []int{1, 2, 4, 8, 16, 32}
	dim := 512

	for _, batch := range batchSizes {
		b.Run(fmt.Sprintf("batch_%d", batch), func(b *testing.B) {
			// batch x dim x dim matrix multiplication
			a := make([]float32, batch*dim*dim)
			b := make([]float32, batch*dim*dim)
			c := make([]float32, batch*dim*dim)

			for i := range a {
				a[i] = float32(i%100) * 0.01
			}
			for i := range b {
				b[i] = float32(i%100) * 0.01
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				BatchMatMul(a, b, c, batch, dim, dim, dim)
			}
		})
	}
}
