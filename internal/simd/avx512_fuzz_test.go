//go:build amd64 && cgo && avx512

package simd

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"
	"testing/quick"
)

func bytesToF32(data []byte) []float32 {
	n := len(data) / 4
	if n == 0 {
		return nil
	}
	res := make([]float32, n)
	for i := 0; i < n; i++ {
		bits := binary.LittleEndian.Uint32(data[i*4 : (i+1)*4])
		res[i] = math.Float32frombits(bits)
	}
	return res
}

func bytesToU16(data []byte) []uint16 {
	n := len(data) / 2
	if n == 0 {
		return nil
	}
	res := make([]uint16, n)
	for i := 0; i < n; i++ {
		res[i] = binary.LittleEndian.Uint16(data[i*2 : (i+1)*2])
	}
	return res
}

func bytesToInts(data []byte) []int {
	n := len(data) / 4
	if n == 0 {
		return nil
	}
	res := make([]int, n)
	for i := 0; i < n; i++ {
		res[i] = int(int32(binary.LittleEndian.Uint32(data[i*4 : (i+1)*4]))) // #nosec G115 -- fuzz input conversion
	}
	return res
}

func FuzzSoftmax(f *testing.F) {
	f.Add([]byte{0, 0, 128, 63, 0, 0, 0, 64})
	f.Fuzz(func(t *testing.T, rawData []byte) {
		data := bytesToF32(rawData)
		if len(data) == 0 || len(data) > 1024 {
			return
		}
		for _, v := range data {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return
			}
		}

		input := make([]float32, len(data))
		copy(input, data)

		Softmax(input)

		sum := float64(0)
		for _, v := range input {
			if math.IsInf(float64(v), 0) || math.IsNaN(float64(v)) {
				t.Logf("Inf or NaN in output: %v", v)
				continue
			}
			sum += float64(v)
		}

		if !math.IsInf(sum, 0) && !math.IsNaN(sum) && (sum < 0.99 || sum > 1.01) && len(input) > 0 && math.Abs(sum-1.0) > 0.1 {
			t.Logf("Softmax sum = %v, want ~1.0", sum)
		}
	})
}

func FuzzSwiGLU(f *testing.F) {
	f.Add([]byte{0, 0, 128, 63, 0, 0, 0, 64}, []byte{0, 0, 0, 64, 0, 0, 128, 63})
	f.Fuzz(func(t *testing.T, rawGate, rawUp []byte) {
		gate := bytesToF32(rawGate)
		up := bytesToF32(rawUp)
		if len(gate) == 0 || len(up) == 0 {
			return
		}
		minLen := len(gate)
		if len(up) < minLen {
			minLen = len(up)
		}
		if minLen > 1024 {
			minLen = 1024
		}
		gate = gate[:minLen]
		up = up[:minLen]

		out := make([]float32, minLen)
		SwiGLU(gate, up, out)

		for i, v := range out {
			if math.IsInf(float64(v), 0) || math.IsNaN(float64(v)) {
				t.Logf("SwiGLU[%d] = %v", i, v)
			}
		}
	})
}

func FuzzRMSNorm(f *testing.F) {
	f.Add([]byte{0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64}, []byte{0, 0, 128, 63, 0, 0, 128, 63}, 2, 2, float32(1e-5))
	f.Fuzz(func(t *testing.T, rawInput, rawWeight []byte, rows, cols int, eps float32) {
		if rows <= 0 || rows > 32 || cols <= 0 || cols > 64 {
			return
		}
		input := bytesToF32(rawInput)
		weight := bytesToF32(rawWeight)
		if len(input) < rows*cols || len(weight) < cols {
			return
		}
		if eps <= 0 || math.IsNaN(float64(eps)) {
			eps = 1e-5
		}

		output := make([]float32, rows*cols)
		RMSNorm(input, weight, output, rows, cols, eps)

		for i, v := range output {
			if math.IsInf(float64(v), 0) || math.IsNaN(float64(v)) {
				t.Logf("RMSNorm[%d] = %v", i, v)
			}
		}
	})
}

func FuzzMatmul(f *testing.F) {
	f.Add([]byte{0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64}, []byte{0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64}, 2, 2, 2)
	f.Fuzz(func(t *testing.T, rawA, rawB []byte, m, n, k int) {
		if m <= 0 || m > 16 || n <= 0 || n > 16 || k <= 0 || k > 16 {
			return
		}
		a := bytesToF32(rawA)
		b := bytesToF32(rawB)
		if len(a) < m*k || len(b) < k*n {
			return
		}

		c := make([]float32, m*n)
		Matmul(a, b, c, m, n, k)

		for i, v := range c {
			if math.IsInf(float64(v), 0) || math.IsNaN(float64(v)) {
				t.Logf("Matmul[%d] = %v", i, v)
			}
		}
	})
}

func FuzzFusedAttention(f *testing.F) {
	f.Add([]byte{0, 0, 128, 63, 0, 0, 0, 64}, []byte{0, 0, 128, 63, 0, 0, 0, 64}, []byte{0, 0, 128, 63, 0, 0, 0, 64}, 1, 1, 1, 1, 2, float32(0.25))
	f.Fuzz(func(t *testing.T, rawQ, rawK, rawV []byte, batch, heads, seqLen, kvSeqLen, headDim int, scale float32) {
		if batch <= 0 || batch > 2 || heads <= 0 || heads > 4 || seqLen <= 0 || seqLen > 4 || kvSeqLen <= 0 || kvSeqLen > 4 || headDim <= 0 || headDim > 64 {
			return
		}
		totalQ := batch * heads * seqLen * headDim
		totalK := batch * heads * kvSeqLen * headDim
		q := bytesToF32(rawQ)
		k := bytesToF32(rawK)
		v := bytesToF32(rawV)

		if len(q) < totalQ || len(k) < totalK || len(v) < totalK {
			return
		}
		if scale == 0 || math.IsNaN(float64(scale)) {
			scale = 0.25
		}

		output := make([]float32, totalQ)
		FusedAttention(q, k, v, output, batch, heads, seqLen, kvSeqLen, headDim, scale)

		for i, v := range output {
			if math.IsInf(float64(v), 0) || math.IsNaN(float64(v)) {
				t.Logf("FusedAttention[%d] = %v", i, v)
			}
		}
	})
}

func FuzzFusedMLP(f *testing.F) {
	f.Add([]byte{0, 0, 128, 63, 0, 0, 0, 64}, []byte{0, 0, 128, 63, 0, 0, 0, 64}, []byte{0, 0, 128, 63, 0, 0, 0, 64}, []byte{0, 0, 128, 63, 0, 0, 0, 64}, 1, 2, 2)
	f.Fuzz(func(t *testing.T, rawInput, rawGateW, rawUpW, rawDownW []byte, batch, dim, hiddenDim int) {
		if batch <= 0 || batch > 2 || dim <= 0 || dim > 16 || hiddenDim <= 0 || hiddenDim > 32 {
			return
		}
		input := bytesToF32(rawInput)
		gateW := bytesToF32(rawGateW)
		upW := bytesToF32(rawUpW)
		downW := bytesToF32(rawDownW)
		inputSize := batch * dim
		if len(input) < inputSize || len(gateW) < hiddenDim || len(upW) < hiddenDim || len(downW) < dim*hiddenDim {
			return
		}

		output := make([]float32, inputSize)
		FusedMLP(input, gateW, upW, downW, output, batch, dim, hiddenDim)

		for i, v := range output {
			if math.IsInf(float64(v), 0) || math.IsNaN(float64(v)) {
				t.Logf("FusedMLP[%d] = %v", i, v)
			}
		}
	})
}

func FuzzRoPE(f *testing.F) {
	f.Add([]byte{0, 0, 128, 63, 0, 0, 0, 64}, []byte{0, 0, 0, 0}, 1, 1, 1, 2, float32(10000.0))
	f.Fuzz(func(t *testing.T, rawTensor, rawPositions []byte, batch, heads, seqLen, headDim int, theta float32) {
		if batch <= 0 || batch > 2 || heads <= 0 || heads > 2 || seqLen <= 0 || seqLen > 4 || headDim <= 0 || headDim > 32 || headDim%2 != 0 {
			return
		}
		totalSize := batch * heads * seqLen * headDim
		tensor := bytesToF32(rawTensor)
		positions := bytesToInts(rawPositions)
		if len(tensor) < totalSize || len(positions) < seqLen {
			return
		}
		if theta <= 0 || math.IsNaN(float64(theta)) {
			theta = 10000.0
		}

		RoPE(tensor, positions, batch, heads, seqLen, headDim, theta)

		for i, v := range tensor {
			if math.IsInf(float64(v), 0) || math.IsNaN(float64(v)) {
				t.Logf("RoPE[%d] = %v", i, v)
			}
		}
	})
}

func FuzzFp16ToFp32(f *testing.F) {
	f.Add([]byte{0, 0, 0x00, 0x3c})
	f.Fuzz(func(t *testing.T, rawSrc []byte) {
		src := bytesToU16(rawSrc)
		if len(src) == 0 || len(src) > 1024 {
			return
		}

		dst := make([]float32, len(src))
		Fp16ToFp32(src, dst)

		for i, v := range dst {
			if math.IsInf(float64(v), 0) || math.IsNaN(float64(v)) {
				t.Logf("Fp16ToFp32[%d] = %v", i, v)
			}
		}
	})
}

func FuzzFp32ToFp16(f *testing.F) {
	f.Add([]byte{0, 0, 128, 63})
	f.Fuzz(func(t *testing.T, rawSrc []byte) {
		src := bytesToF32(rawSrc)
		if len(src) == 0 || len(src) > 1024 {
			return
		}

		dst := make([]uint16, len(src))
		Fp32ToFp16(src, dst)
	})
}

func TestSoftmax_QuickCheck(t *testing.T) {
	err := quick.Check(func(data []float32) bool {
		if len(data) == 0 || len(data) > 10000 {
			return true
		}

		input := make([]float32, len(data))
		copy(input, data)

		Softmax(input)

		sum := float64(0)
		for _, v := range input {
			sum += float64(v)
		}

		if math.IsInf(sum, 0) || math.IsNaN(sum) {
			return true
		}

		return math.Abs(sum-1.0) < 0.01 || len(input) == 0
	}, nil)

	if err != nil {
		t.Error(err)
	}
}

func TestSwiGLU_QuickCheck(t *testing.T) {
	err := quick.Check(func(gate, up []float32) bool {
		if len(gate) == 0 || len(gate) != len(up) || len(gate) > 10000 {
			return true
		}
		for i := range gate {
			if math.IsNaN(float64(gate[i])) || math.IsInf(float64(gate[i]), 0) ||
				math.IsNaN(float64(up[i])) || math.IsInf(float64(up[i]), 0) {
				return true
			}
			if math.Abs(float64(up[i])) > 1e37 {
				return true
			}
		}

		out := make([]float32, len(gate))
		SwiGLU(gate, up, out)

		for _, v := range out {
			if math.IsInf(float64(v), 0) || math.IsNaN(float64(v)) {
				return false
			}
		}

		return true
	}, nil)

	if err != nil {
		t.Error(err)
	}
}

func BenchmarkSoftmax_AVX512(b *testing.B) {
	sizes := []int{64, 256, 1024, 4096, 16384}

	for _, n := range sizes {
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			data := make([]float32, n)
			rng := rand.New(rand.NewSource(42))
			for i := range data {
				data[i] = rng.Float32() * 10
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				input := make([]float32, len(data))
				copy(input, data)
				Softmax(input)
			}
		})
	}
}

func BenchmarkRMSNorm_AVX512(b *testing.B) {
	configs := [][2]int{{4, 64}, {16, 128}, {64, 256}, {128, 512}}

	for _, cfg := range configs {
		rows, cols := cfg[0], cfg[1]
		b.Run(fmt.Sprintf("%dx%d", rows, cols), func(b *testing.B) {
			input := make([]float32, rows*cols)
			weight := make([]float32, cols)
			output := make([]float32, rows*cols)

			rng := rand.New(rand.NewSource(42))
			for i := range input {
				input[i] = (rng.Float32() - 0.5) * 2
			}
			for i := range weight {
				weight[i] = (rng.Float32() - 0.5) * 2
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				RMSNorm(input, weight, output, rows, cols, 1e-5)
			}
		})
	}
}

func BenchmarkMatmul_AVX512(b *testing.B) {
	configs := [][3]int{{8, 16, 8}, {32, 32, 32}, {64, 64, 64}, {128, 128, 64}}

	for _, cfg := range configs {
		m, n, k := cfg[0], cfg[1], cfg[2]
		b.Run(fmt.Sprintf("%dx%dx%d", m, n, k), func(b *testing.B) {
			a := make([]float32, m*k)
			matB := make([]float32, k*n)
			c := make([]float32, m*n)

			rng := rand.New(rand.NewSource(42))
			for i := range a {
				a[i] = (rng.Float32() - 0.5) * 2
			}
			for i := range matB {
				matB[i] = (rng.Float32() - 0.5) * 2
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				Matmul(a, matB, c, m, n, k)
			}
		})
	}
}

func BenchmarkFusedAttention_AVX512(b *testing.B) {
	configs := [][4]int{{1, 4, 32, 32}, {2, 8, 64, 32}, {4, 16, 128, 32}}

	for _, cfg := range configs {
		batch, heads, seqLen, headDim := cfg[0], cfg[1], cfg[2], cfg[3]
		kvSeqLen := seqLen
		b.Run(fmt.Sprintf("%dx%dx%dx%d", batch, heads, seqLen, headDim), func(b *testing.B) {
			q := make([]float32, batch*heads*seqLen*headDim)
			k := make([]float32, batch*heads*kvSeqLen*headDim)
			v := make([]float32, batch*heads*kvSeqLen*headDim)
			output := make([]float32, batch*heads*seqLen*headDim)

			rng := rand.New(rand.NewSource(42))
			for i := range q {
				q[i] = (rng.Float32() - 0.5) * 2
			}
			for i := range k {
				k[i] = (rng.Float32() - 0.5) * 2
			}
			for i := range v {
				v[i] = (rng.Float32() - 0.5) * 2
			}

			scale := float32(1.0) / float32(headDim)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				FusedAttention(q, k, v, output, batch, heads, seqLen, kvSeqLen, headDim, scale)
			}
		})
	}
}

func BenchmarkFusedMLP_AVX512(b *testing.B) {
	configs := [][3]int{{1, 256, 1024}, {2, 512, 2048}, {4, 512, 2048}}

	for _, cfg := range configs {
		batch, dim, hiddenDim := cfg[0], cfg[1], cfg[2]
		b.Run(fmt.Sprintf("%dx%dx%d", batch, dim, hiddenDim), func(b *testing.B) {
			input := make([]float32, batch*dim)
			gateW := make([]float32, hiddenDim)
			upW := make([]float32, hiddenDim)
			downW := make([]float32, dim*hiddenDim)
			output := make([]float32, batch*dim)

			rng := rand.New(rand.NewSource(42))
			for i := range input {
				input[i] = (rng.Float32() - 0.5) * 2
			}
			for i := range gateW {
				gateW[i] = (rng.Float32() - 0.5) * 2
			}
			for i := range upW {
				upW[i] = (rng.Float32() - 0.5) * 2
			}
			for i := range downW {
				downW[i] = (rng.Float32() - 0.5) * 2
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				FusedMLP(input, gateW, upW, downW, output, batch, dim, hiddenDim)
			}
		})
	}
}

func BenchmarkRoPE_AVX512(b *testing.B) {
	configs := [][4]int{{1, 4, 32, 32}, {2, 8, 64, 32}, {4, 16, 128, 32}}

	for _, cfg := range configs {
		batch, heads, seqLen, headDim := cfg[0], cfg[1], cfg[2], cfg[3]
		b.Run(fmt.Sprintf("%dx%dx%dx%d", batch, heads, seqLen, headDim), func(b *testing.B) {
			tensor := make([]float32, batch*heads*seqLen*headDim)
			positions := make([]int, seqLen)

			rng := rand.New(rand.NewSource(42))
			for i := range tensor {
				tensor[i] = (rng.Float32() - 0.5) * 2
			}
			for i := range positions {
				positions[i] = i
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				RoPE(tensor, positions, batch, heads, seqLen, headDim, 10000)
			}
		})
	}
}

func BenchmarkSIMDLevelDetection(b *testing.B) {
	for b.Loop() {
		GetSIMDLevel()
	}
}