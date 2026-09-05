package gguf

import (
	"math"
	"math/rand"
	"testing"
)

func TestDequantizeQ4K_SIMD_Parity(t *testing.T) {
	numBlocks := 4
	numElements := numBlocks * BlockSizeQ4K
	dataLen := numBlocks * 144
	data := make([]byte, dataLen)

	// Fill with deterministic pseudo-random data
	rng := rand.New(rand.NewSource(42))
	_, _ = rng.Read(data)

	// Set valid float16 scale
	for b := 0; b < numBlocks; b++ {
		offset := b * 144
		data[offset] = 0x00
		data[offset+1] = 0x3C // 1.0 in FP16
	}

	scalarOut := DequantizeQ4K(data, numElements)
	simdOut := DequantizeQ4K_SIMD(data, numElements)

	if len(scalarOut) != len(simdOut) {
		t.Fatalf("length mismatch: scalar %d vs simd %d", len(scalarOut), len(simdOut))
	}

	for i := 0; i < numElements; i++ {
		diff := math.Abs(float64(scalarOut[i] - simdOut[i]))
		if diff > 1e-6 {
			t.Fatalf("Q4_K mismatch at %d: scalar %v vs simd %v", i, scalarOut[i], simdOut[i])
		}
	}
}

func TestDequantizeQ6K_SIMD_Parity(t *testing.T) {
	numBlocks := 4
	numElements := numBlocks * BlockSizeQ6K
	dataLen := numBlocks * 210
	data := make([]byte, dataLen)

	rng := rand.New(rand.NewSource(99))
	_, _ = rng.Read(data)

	// Set valid float16 scale at offset 208
	for b := 0; b < numBlocks; b++ {
		offset := b * 210
		data[offset+208] = 0x00
		data[offset+209] = 0x38 // 0.5 in FP16
	}

	scalarOut := DequantizeQ6K(data, numElements)
	simdOut := DequantizeQ6K_SIMD(data, numElements)

	if len(scalarOut) != len(simdOut) {
		t.Fatalf("length mismatch: scalar %d vs simd %d", len(scalarOut), len(simdOut))
	}

	for i := 0; i < numElements; i++ {
		diff := math.Abs(float64(scalarOut[i] - simdOut[i]))
		if diff > 1e-6 {
			t.Fatalf("Q6_K mismatch at %d: scalar %v vs simd %v", i, scalarOut[i], simdOut[i])
		}
	}
}

func TestMatVecMulQ4_K(t *testing.T) {
	rows := 4
	cols := 256
	data := make([]byte, rows*144)
	rng := rand.New(rand.NewSource(123))
	_, _ = rng.Read(data)

	vec := make([]float32, cols)
	for i := range vec {
		vec[i] = rng.Float32()
	}

	// Reference result by dequantizing and multiplying
	deq := DequantizeQ4K(data, rows*cols)
	ref := make([]float32, rows)
	for r := 0; r < rows; r++ {
		var sum float32
		for c := 0; c < cols; c++ {
			sum += deq[r*cols+c] * vec[c]
		}
		ref[r] = sum
	}

	res := MatVecMulQ4_K(data, vec, rows, cols)
	for r := 0; r < rows; r++ {
		diff := math.Abs(float64(res[r] - ref[r]))
		if diff > 1e-3 {
			t.Errorf("MatVecMulQ4_K row %d got %v, want %v", r, res[r], ref[r])
		}
	}
}

func TestMatVecMulQ6_K(t *testing.T) {
	rows := 4
	cols := 256
	data := make([]byte, rows*210)
	rng := rand.New(rand.NewSource(456))
	_, _ = rng.Read(data)

	vec := make([]float32, cols)
	for i := range vec {
		vec[i] = rng.Float32()
	}

	deq := DequantizeQ6K(data, rows*cols)
	ref := make([]float32, rows)
	for r := 0; r < rows; r++ {
		var sum float32
		for c := 0; c < cols; c++ {
			sum += deq[r*cols+c] * vec[c]
		}
		ref[r] = sum
	}

	res := MatVecMulQ6_K(data, vec, rows, cols)
	for r := 0; r < rows; r++ {
		diff := math.Abs(float64(res[r] - ref[r]))
		if diff > 1e-3 {
			t.Errorf("MatVecMulQ6_K row %d got %v, want %v", r, res[r], ref[r])
		}
	}
}

func FuzzDequantizeQ4K_SIMD(f *testing.F) {
	f.Add(make([]byte, 144))
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) < 144 {
			return
		}
		numBlocks := len(data) / 144
		numElements := numBlocks * 256
		_ = DequantizeQ4K_SIMD(data[:numBlocks*144], numElements)
	})
}

func FuzzDequantizeQ6K_SIMD(f *testing.F) {
	f.Add(make([]byte, 210))
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) < 210 {
			return
		}
		numBlocks := len(data) / 210
		numElements := numBlocks * 256
		_ = DequantizeQ6K_SIMD(data[:numBlocks*210], numElements)
	})
}

func BenchmarkDequantizeQ4K_Comparison(b *testing.B) {
	numBlocks := 64
	numElements := numBlocks * BlockSizeQ4K
	data := make([]byte, numBlocks*144)

	b.Run("Scalar", func(b *testing.B) {
		for b.Loop() {
			_ = DequantizeQ4K(data, numElements)
		}
	})

	b.Run("SIMD", func(b *testing.B) {
		for b.Loop() {
			_ = DequantizeQ4K_SIMD(data, numElements)
		}
	})
}

func BenchmarkDequantizeQ6K_Comparison(b *testing.B) {
	numBlocks := 64
	numElements := numBlocks * BlockSizeQ6K
	data := make([]byte, numBlocks*210)

	b.Run("Scalar", func(b *testing.B) {
		for b.Loop() {
			_ = DequantizeQ6K(data, numElements)
		}
	})

	b.Run("SIMD", func(b *testing.B) {
		for b.Loop() {
			_ = DequantizeQ6K_SIMD(data, numElements)
		}
	})
}
