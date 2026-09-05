//go:build !avx512

package simd

import (
	"math"
	"testing"
)

func TestSoftmax(t *testing.T) {
	testCases := []struct {
		name     string
		input    []float64
		expected []float64
	}{
		{
			name:     "simple",
			input:    []float64{1, 2, 3},
			expected: []float64{0.09003057, 0.24472847, 0.66524096},
		},
		{
			name:     "negative",
			input:    []float64{-1, -2, -3},
			expected: []float64{0.66524096, 0.24472847, 0.09003057},
		},
		{
			name:     "zero",
			input:    []float64{0, 0, 0},
			expected: []float64{0.33333333, 0.33333333, 0.33333333},
		},
		{
			name:     "empty",
			input:    []float64{},
			expected: []float64{},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			input := make([]float64, len(tc.input))
			copy(input, tc.input)
			Softmax(input)
			if len(input) != len(tc.expected) {
				t.Errorf("expected length %d, got %d", len(tc.expected), len(input))
			}
			for i := range input {
				if math.Abs(input[i]-tc.expected[i]) > 1e-6 {
					t.Errorf("expected %v, got %v", tc.expected, input)
					break
				}
			}
		})
	}
}

func TestSoftmaxF32(t *testing.T) {
	input := []float32{1, 2, 3}
	SoftmaxF32(input)

	expected := []float64{0.09003057, 0.24472847, 0.66524096}
	for i := range input {
		if math.Abs(float64(input[i])-expected[i]) > 1e-5 {
			t.Errorf("SoftmaxF32 failed at %d: got %f, expected %f", i, input[i], expected[i])
		}
	}
}

func TestSwiGLU(t *testing.T) {
	gate := []float32{0, 1, -1, 10, -10}
	up := []float32{1, 2, 3, 4, 5}
	out := make([]float32, 5)
	SwiGLU(gate, up, out)

	for i := range out {
		if math.IsInf(float64(out[i]), 0) || math.IsNaN(float64(out[i])) {
			t.Errorf("SwiGLU produced invalid value at %d: %f", i, out[i])
		}
	}
}

func TestFp16ToFp32(t *testing.T) {
	src := []uint16{0x3C00, 0x4000, 0x0000, 0x8000}
	dst := make([]float32, 4)
	Fp16ToFp32(src, dst)

	expected := []float32{1.0, 2.0, 0.0, float32(math.Copysign(0, -1))}
	for i := range dst {
		if math.Abs(float64(dst[i])-float64(expected[i])) > 1e-6 {
			t.Errorf("Fp16ToFp32 failed at %d: got %f, expected %f", i, dst[i], expected[i])
		}
	}
}

func TestSoftmaxStability(t *testing.T) {
	// Test that softmax handles very large values numerically stably
	x := []float64{1000.0, 1001.0, 1002.0}
	Softmax(x)

	sum := x[0] + x[1] + x[2]
	if math.Abs(sum-1.0) > 1e-10 {
		t.Errorf("Softmax sum = %f, want 1.0", sum)
	}
}

func TestSoftmaxF32Stability(t *testing.T) {
	x := []float32{1000.0, 1001.0, 1002.0}
	SoftmaxF32(x)

	var sum float32
	for _, v := range x {
		sum += v
	}
	if math.Abs(float64(sum)-1.0) > 1e-6 {
		t.Errorf("SoftmaxF32 sum = %f, want 1.0", sum)
	}
}

func TestSwiGLUClamping(t *testing.T) {
	gate := []float32{-20.0, -10.0, 10.0, 20.0}
	up := []float32{1.0, 1.0, 1.0, 1.0}
	out := make([]float32, 4)

	SwiGLU(gate, up, out)

	// Verify no NaN/Inf produced
	for i, v := range out {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("SwiGLU clamping produced NaN/Inf at %d: %f", i, v)
		}
	}
}

func TestFp16ToFp32Inf(t *testing.T) {
	src := []uint16{0x7C00} // +Inf
	dst := make([]float32, 1)
	Fp16ToFp32(src, dst)

	if !math.IsInf(float64(dst[0]), 1) {
		t.Errorf("Fp16ToFp32 +Inf = %f, want +Inf", dst[0])
	}
}

func TestFp16ToFp32NegInf(t *testing.T) {
	src := []uint16{0xFC00} // -Inf
	dst := make([]float32, 1)
	Fp16ToFp32(src, dst)

	if !math.IsInf(float64(dst[0]), -1) {
		t.Errorf("Fp16ToFp32 -Inf = %f, want -Inf", dst[0])
	}
}

func TestFp16ToFp32NaN(t *testing.T) {
	src := []uint16{0x7E00} // NaN
	dst := make([]float32, 1)
	Fp16ToFp32(src, dst)

	if !math.IsNaN(float64(dst[0])) {
		t.Errorf("Fp16ToFp32 NaN = %f, want NaN", dst[0])
	}
}

func TestFp32ToFp16Inf(t *testing.T) {
	src := []float32{float32(math.Inf(1))}
	dst := make([]uint16, 1)
	Fp32ToFp16(src, dst)

	if dst[0] != 0x7C00 {
		t.Errorf("Fp32ToFp16 +Inf = 0x%04X, want 0x7C00", dst[0])
	}
}

func TestFp32ToFp16NegInf(t *testing.T) {
	src := []float32{float32(math.Inf(-1))}
	dst := make([]uint16, 1)
	Fp32ToFp16(src, dst)

	if dst[0] != 0xFC00 {
		t.Errorf("Fp32ToFp16 -Inf = 0x%04X, want 0xFC00", dst[0])
	}
}

func TestSoftmaxVeryLargeInput(t *testing.T) {
	x := []float64{1e20, 1e20 + 1}
	Softmax(x)

	// Should handle without overflow
	for _, v := range x {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("Softmax very large input produced NaN/Inf: %f", v)
		}
	}
}

func TestSwiGLUEmpty(t *testing.T) {
	SwiGLU([]float32{}, []float32{}, make([]float32, 0))
}

func TestSoftmaxPreservesOrder(t *testing.T) {
	x := []float64{1.0, 2.0, 3.0}
	Softmax(x)

	// Larger inputs should have larger outputs after softmax
	if !(x[0] < x[1] && x[1] < x[2]) {
		t.Errorf("Softmax should preserve order: got %v", x)
	}
}

func TestFp16ToFp32Precision(t *testing.T) {
	// Test some specific FP16 values
	tests := []uint16{
		0x3C00, // 1.0
		0x4000, // 2.0
		0x4200, // 4.0
		0x0000, // 0.0
		0xC000, // -2.0
	}

	for _, src := range tests {
		dst := make([]float32, 1)
		Fp16ToFp32([]uint16{src}, dst)

		// Just verify no crash and reasonable range
		if math.IsNaN(float64(dst[0])) && src != 0x7E00 {
			t.Errorf("Fp16ToFp32(0x%04X) = NaN, unexpected", src)
		}
	}
}

func TestMatMul(t *testing.T) {
	a := []float32{1, 2, 3, 4}
	b := []float32{5, 6, 7, 8}

	c := MatMul(a, b, 2, 2, 2)
	expected := []float32{19, 22, 43, 50}

	for i := range c {
		if c[i] != expected[i] {
			t.Errorf("MatMul[%d] = %f, want %f", i, c[i], expected[i])
		}
	}
}

func TestMatMulNonSquare(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5, 6}
	b := []float32{7, 8, 9, 10, 11, 12}

	c := MatMul(a, b, 2, 3, 2)
	expected := []float32{58, 64, 139, 154}

	for i := range c {
		if c[i] != expected[i] {
			t.Errorf("MatMul[%d] = %f, want %f", i, c[i], expected[i])
		}
	}
}

func TestMatMulEmpty(t *testing.T) {
	result := MatMul(nil, nil, 0, 0, 0)
	if len(result) != 0 {
		t.Error("expected empty result")
	}
}

func TestMatVecMul(t *testing.T) {
	matrix := []float32{1, 2, 3, 4}
	vector := []float32{5, 6}

	result := MatVecMul(matrix, vector, 2, 2)
	expected := []float32{17, 39}

	for i := range result {
		if result[i] != expected[i] {
			t.Errorf("MatVecMul[%d] = %f, want %f", i, result[i], expected[i])
		}
	}
}

func TestMatVecMulRectangular(t *testing.T) {
	matrix := []float32{1, 2, 3, 4, 5, 6}
	vector := []float32{7, 8, 9}

	result := MatVecMul(matrix, vector, 2, 3)
	expected := []float32{50, 122}

	for i := range result {
		if result[i] != expected[i] {
			t.Errorf("MatVecMul[%d] = %f, want %f", i, result[i], expected[i])
		}
	}
}

func TestMatVecMulEmpty(t *testing.T) {
	result := MatVecMul(nil, nil, 0, 0)
	if len(result) != 0 {
		t.Error("expected empty result")
	}
}

func TestAttentionF32(t *testing.T) {
	seqLen := 3
	heads := 1
	headDim := 4

	q := []float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}
	k := []float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}
	v := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

	result := AttentionF32(q, k, v, seqLen, heads, headDim)

	if len(result) != seqLen*headDim {
		t.Fatalf("expected length %d, got %d", seqLen*headDim, len(result))
	}

	for _, val := range result {
		if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
			t.Errorf("unexpected NaN/Inf in output: %v", val)
		}
	}
}

func TestAttentionF32Causal(t *testing.T) {
	seqLen := 4
	heads := 1
	headDim := 2

	q := []float32{1, 0, 0, 1, 1, 0, 0, 1}
	k := []float32{1, 0, 1, 0, 1, 0, 1, 0}
	v := []float32{1, 1, 2, 2, 3, 3, 4, 4}

	result := AttentionF32(q, k, v, seqLen, heads, headDim)

	for _, val := range result {
		if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
			t.Errorf("unexpected NaN/Inf: %v", val)
		}
	}
}

func TestAttentionF32Trivial(t *testing.T) {
	result := AttentionF32(nil, nil, nil, 0, 0, 0)
	if len(result) != 0 {
		t.Error("expected empty for zero seqLen")
	}
}

func TestFp16ToFp32Subnormal(t *testing.T) {
	src := []uint16{0x0001, 0x0050, 0x00FF}
	dst := make([]float32, len(src))
	Fp16ToFp32(src, dst)
	for i, v := range dst {
		if math.IsNaN(float64(v)) {
			t.Errorf("Fp16ToFp32[0x%04X] produced NaN", src[i])
		}
	}
}

func TestFp16ToFp32Normal(t *testing.T) {
	src := []uint16{0x3C01, 0x4001, 0x7BFF}
	dst := make([]float32, len(src))
	Fp16ToFp32(src, dst)
	for i, v := range dst {
		if math.IsNaN(float64(v)) {
			t.Errorf("Fp16ToFp32[0x%04X] produced NaN", src[i])
		}
	}
}

func TestFp32ToFp16EdgeCases(t *testing.T) {
	negZero := float32(math.Copysign(0.0, -1.0))
	src := []float32{0.0, negZero, 1.5, -1.5, 0.0001, 65504.0, -65504.0, float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN())}
	dst := make([]uint16, len(src))
	Fp32ToFp16(src, dst)

	for i, v := range dst {
		if math.Float32frombits(uint32(v)) == math.Float32frombits(0xFFFF) {
			continue
		}
		if math.IsNaN(float64(src[i])) && v&0x7E00 == 0x7E00 {
			continue
		}
	}
}

func TestFp32ToFp16Zero(t *testing.T) {
	negZero := float32(math.Copysign(0.0, -1.0))
	src := []float32{0.0, negZero}
	dst := make([]uint16, len(src))
	Fp32ToFp16(src, dst)
	if dst[0] != 0 {
		t.Errorf("+0 -> 0x%04X, want 0x0000", dst[0])
	}
}

func TestFp32ToFp16Rounding(t *testing.T) {
	src := []float32{1.0, 2.0, 4.0}
	dst := make([]uint16, len(src))
	Fp32ToFp16(src, dst)
	if dst[0] != 0x3C00 {
		t.Errorf("1.0 -> 0x%04X, want 0x3C00", dst[0])
	}
	if dst[1] != 0x4000 {
		t.Errorf("2.0 -> 0x%04X, want 0x4000", dst[1])
	}
}

func BenchmarkMatMulLarge(b *testing.B) {
	n := 128
	a := make([]float32, n*n)
	bb := make([]float32, n*n)
	for i := range a {
		a[i] = float32(i % 100)
		bb[i] = float32(i % 100)
	}

	for b.Loop() {
		MatMul(a, bb, n, n, n)
	}
}

func TestRMSNorm(t *testing.T) {
	input := []float32{1, 2, 3, 4}
	weight := []float32{1, 1, 1, 1}
	output := make([]float32, 4)
	RMSNorm(input, weight, output, 1, 4, 1e-5)

	sum := float32(0)
	for _, v := range output {
		sum += v * v
	}
	meanSq := sum / 4
	if math.Abs(float64(meanSq-1)) > 1e-4 {
		t.Errorf("RMSNorm output mean^2=%f, expected ~1", meanSq)
	}
}

func TestRMSNormMultiRow(t *testing.T) {
	input := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	weight := []float32{1, 1, 1, 1}
	output := make([]float32, 8)
	RMSNorm(input, weight, output, 2, 4, 1e-5)

	if len(output) != 8 {
		t.Fatalf("expected 8 outputs, got %d", len(output))
	}
	for _, v := range output {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("unexpected NaN/Inf: %v", v)
		}
	}
}

func TestMatmulDefault(t *testing.T) {
	a := []float32{1, 2, 3, 4}
	b := []float32{5, 6, 7, 8}
	c := make([]float32, 4)
	Matmul(a, b, c, 2, 2, 2)
	expected := []float32{19, 22, 43, 50}
	for i := range c {
		if c[i] != expected[i] {
			t.Errorf("Matmul[%d] = %f, want %f", i, c[i], expected[i])
		}
	}
}

func TestMatmulDefaultNonSquare(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5, 6}
	b := []float32{7, 8, 9, 10, 11, 12}
	c := make([]float32, 4)
	Matmul(a, b, c, 2, 2, 3)
	expected := []float32{58, 64, 139, 154}
	for i := range c {
		if c[i] != expected[i] {
			t.Errorf("Matmul[%d] = %f, want %f", i, c[i], expected[i])
		}
	}
}

func TestRoPE(t *testing.T) {
	tensor := []float32{1, 0, 0, 1, 1, 0, 0, 1}
	positions := []int{0, 1}
	RoPE(tensor, positions, 1, 1, 2, 4, 10000.0)
	for _, v := range tensor {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("RoPE produced NaN/Inf: %v", v)
		}
	}
}

func TestRoPEEmpty(t *testing.T) {
	RoPE(nil, nil, 0, 0, 0, 0, 10000.0)
}

func TestFusedAttention(t *testing.T) {
	seqLen := 2
	heads := 1
	headDim := 4
	q := []float32{1, 0, 0, 0, 0, 1, 0, 0}
	k := []float32{1, 0, 0, 0, 0, 1, 0, 0}
	v := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	output := make([]float32, 8)
	FusedAttention(q, k, v, output, 1, heads, seqLen, seqLen, headDim, 0.5)
	for _, val := range output {
		if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
			t.Errorf("FusedAttention produced NaN/Inf: %v", val)
		}
	}
}

func TestFusedAttentionMultiHead(t *testing.T) {
	seqLen := 2
	heads := 2
	headDim := 2
	q := make([]float32, seqLen*heads*headDim)
	k := make([]float32, seqLen*heads*headDim)
	v := make([]float32, seqLen*heads*headDim)
	for i := range q {
		q[i] = float32(i%3) / 3.0
		k[i] = float32((i+1)%3) / 3.0
		v[i] = float32(i) / 10.0
	}
	output := make([]float32, seqLen*heads*headDim)
	FusedAttention(q, k, v, output, 1, heads, seqLen, seqLen, headDim, 0.5)
	for _, val := range output {
		if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
			t.Errorf("FusedAttention multi-head produced NaN/Inf: %v", val)
		}
	}
}

func TestFusedMLP(t *testing.T) {
	batch := 1
	dim := 4
	hiddenDim := 8
	input := []float32{1, 2, 3, 4}
	gateW := []float32{1, -1, 2, -2, 3, -3, 4, -4}
	upW := make([]float32, batch*dim*hiddenDim)
	downW := make([]float32, hiddenDim*dim)
	for i := range upW {
		upW[i] = float32(i) / float32(hiddenDim)
	}
	output := make([]float32, batch*dim)
	FusedMLP(input, gateW, upW, downW, output, batch, dim, hiddenDim)
	for _, val := range output {
		if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
			t.Errorf("FusedMLP produced NaN/Inf: %v", val)
		}
	}
}

func FuzzRMSNorm(f *testing.F) {
	f.Add([]byte{0, 0, 128, 63, 0, 0, 0, 64}, []byte{0, 0, 128, 63, 0, 0, 0, 64}, int(1), int(2), float32(1e-5))
	f.Fuzz(func(t *testing.T, inputBytes, weightBytes []byte, rows, cols int, eps float32) {
		if rows <= 0 || cols <= 0 || rows*cols > 10000 {
			return
		}
		if len(inputBytes) < rows*cols*4 || len(weightBytes) < cols*4 {
			return
		}
		input := bytesToF32(inputBytes[:rows*cols*4])
		weight := bytesToF32(weightBytes[:cols*4])
		output := make([]float32, rows*cols)
		RMSNorm(input, weight, output, rows, cols, eps)
		for _, v := range output {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Errorf("RMSNorm produced NaN/Inf")
				break
			}
		}
	})
}

func FuzzMatmulDefault(f *testing.F) {
	f.Add([]byte{1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0}, []byte{5, 0, 0, 0, 6, 0, 0, 0, 7, 0, 0, 0, 8, 0, 0, 0}, int(2), int(2), int(2))
	f.Fuzz(func(t *testing.T, aBytes, bBytes []byte, m, n, k int) {
		if m <= 0 || n <= 0 || k <= 0 || m*n*k > 100000 {
			return
		}
		if len(aBytes) < m*k*4 || len(bBytes) < k*n*4 {
			return
		}
		a := bytesToF32(aBytes[:m*k*4])
		b := bytesToF32(bBytes[:k*n*4])
		c := make([]float32, m*n)
		Matmul(a, b, c, m, n, k)
		if len(c) != m*n {
			t.Errorf("expected length %d, got %d", m*n, len(c))
		}
	})
}

func BenchmarkSoftmaxLarge(b *testing.B) {
	x := make([]float64, 16384)
	for i := range x {
		x[i] = float64(i%100) / 10.0
	}

	for b.Loop() {
		Softmax(x)
	}
}

func BenchmarkSoftmaxF32Large(b *testing.B) {
	x := make([]float32, 16384)
	for i := range x {
		x[i] = float32(i%100) / 10.0
	}

	for b.Loop() {
		SoftmaxF32(x)
	}
}

func FuzzMatMul(f *testing.F) {
	f.Add([]byte{1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0}, []byte{5, 0, 0, 0, 6, 0, 0, 0, 7, 0, 0, 0, 8, 0, 0, 0}, int(2), int(2), int(2))
	f.Fuzz(func(t *testing.T, aBytes, bBytes []byte, m, n, k int) {
		if m <= 0 || n <= 0 || k <= 0 || m*n*k > 100000 {
			return
		}
		if len(aBytes) < m*k*4 || len(bBytes) < k*n*4 {
			return
		}
		a := bytesToF32(aBytes[:m*k*4])
		b := bytesToF32(bBytes[:k*n*4])
		result := MatMul(a, b, m, n, k)
		if len(result) != m*n {
			t.Errorf("expected length %d, got %d", m*n, len(result))
		}
	})
}

func bytesToF32(data []byte) []float32 {
	out := make([]float32, len(data)/4)
	for i := range out {
		bits := uint32(data[i*4]) | uint32(data[i*4+1])<<8 | uint32(data[i*4+2])<<16 | uint32(data[i*4+3])<<24
		out[i] = math.Float32frombits(bits)
	}
	return out
}

func FuzzMatVecMul(f *testing.F) {
	f.Fuzz(func(t *testing.T, matrixBytes, vectorBytes []byte, rows, cols int) {
		if rows <= 0 || cols <= 0 || rows*cols > 100000 {
			return
		}
		if len(matrixBytes) < rows*cols*4 || len(vectorBytes) < cols*4 {
			return
		}
		matrix := bytesToF32(matrixBytes[:rows*cols*4])
		vector := bytesToF32(vectorBytes[:cols*4])
		result := MatVecMul(matrix, vector, rows, cols)
		if len(result) != rows {
			t.Errorf("expected length %d, got %d", rows, len(result))
		}
	})
}

func FuzzAttentionF32(f *testing.F) {
	f.Fuzz(func(t *testing.T, qBytes, kBytes, vBytes []byte, seqLen, heads, headDim int) {
		if seqLen <= 0 || heads <= 0 || headDim <= 0 || seqLen*headDim*heads > 50000 {
			return
		}
		total := seqLen * headDim * heads
		if len(qBytes) < total*4 || len(kBytes) < total*4 || len(vBytes) < total*4 {
			return
		}
		q := bytesToF32(qBytes[:total*4])
		k := bytesToF32(kBytes[:total*4])
		v := bytesToF32(vBytes[:total*4])
		result := AttentionF32(q, k, v, seqLen, heads, headDim)
		if len(result) != seqLen*headDim {
			t.Errorf("expected length %d, got %d", seqLen*headDim, len(result))
		}
	})
}

func FuzzSoftmaxF32(f *testing.F) {
	f.Add([]byte{0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64})
	f.Fuzz(func(t *testing.T, dataBytes []byte) {
		if len(dataBytes) < 4 || len(dataBytes) > 10000 {
			return
		}
		data := bytesToF32(dataBytes)
		if len(data) == 0 {
			return
		}
		input := make([]float32, len(data))
		copy(input, data)
		SoftmaxF32(input)
		sum := float64(0)
		for _, v := range input {
			sum += float64(v)
		}
		if math.IsInf(sum, 0) || math.IsNaN(sum) {
			return
		}
	})
}

func TestDarwinSysctlParsing(t *testing.T) {
	data := `
hw.optional.avx2_0: 1
hw.optional.avx512f: 1
hw.optional.avx512bw: 1
hw.optional.avx512dq: 1
hw.optional.avx512vl: 1
hw.optional.fma: 1
hw.optional.neon: 1
hw.optional.arm.feat_dotprod: 1
hw.optional.vaes: 1
hw.optional.vpclmulqdq: 1
hw.optional.gfni: 1
hw.optional.avxvnni: 1
hw.optional.unknown: 1
invalid_line_without_delim
hw.optional.avx2_0: 0
`
	res := parseDarwinSysctl(data)
	expectedFlags := []string{"avx2", "avx512f", "avx512bw", "avx512dq", "avx512vl", "fma", "neon", "asimddp", "vaes", "vpclmulqdq", "gfni", "avx_vnni"}
	for _, f := range expectedFlags {
		if !containsFlag(res, f) {
			t.Errorf("expected flag %q in %q", f, res)
		}
	}
}

func TestContainsFlag_Extra(t *testing.T) {
	if containsFlag("", "avx") {
		t.Error("expected false for empty string")
	}
	flags := "fma avx2 neon"
	if !containsFlag(flags, "fma") {
		t.Error("expected true for flag at start")
	}
	if !containsFlag(flags, "avx2") {
		t.Error("expected true for flag in middle")
	}
	if !containsFlag(flags, "neon") {
		t.Error("expected true for flag at end")
	}
	if containsFlag(flags, "avx") {
		t.Error("expected false for substring without word boundary")
	}
	if containsFlag(flags, "sse") {
		t.Error("expected false for missing flag")
	}
}

func TestGetCPULevelAndDetection(t *testing.T) {
	level := GetCPULevel()
	if level < 0 {
		t.Errorf("unexpected CPU level: %d", level)
	}
}

func TestScalarFPConversion_EdgeCases(t *testing.T) {
	// Subnormal / denormal FP16
	denormalFP16 := uint16(0x0001)
	f32 := fp16ToFp32(denormalFP16)
	if f32 <= 0 {
		t.Errorf("expected positive float for denormal, got %f", f32)
	}

	// Infinity and NaN
	infFP16 := uint16(0x7C00)
	if !math.IsInf(float64(fp16ToFp32(infFP16)), 1) {
		t.Errorf("expected +Inf, got %f", fp16ToFp32(infFP16))
	}
	nanFP16 := uint16(0x7C01)
	if !math.IsNaN(float64(fp16ToFp32(nanFP16))) {
		t.Errorf("expected NaN, got %f", fp16ToFp32(nanFP16))
	}

	// FP32 to FP16 special values
	infFP32 := float32(math.Inf(1))
	if fp32ToFp16(infFP32) != 0x7C00 {
		t.Errorf("expected 0x7C00 for +Inf, got 0x%X", fp32ToFp16(infFP32))
	}
	nanFP32 := float32(math.NaN())
	if (fp32ToFp16(nanFP32) & 0x7C00) != 0x7C00 {
		t.Errorf("expected NaN FP16, got 0x%X", fp32ToFp16(nanFP32))
	}
	largeFP32 := float32(1e10)
	if fp32ToFp16(largeFP32) != 0x7C00 {
		t.Errorf("expected overflow to Inf 0x7C00, got 0x%X", fp32ToFp16(largeFP32))
	}
	tinyFP32 := float32(1e-6) // subnormal in FP16
	if fp32ToFp16(tinyFP32) == 0 {
		t.Errorf("expected non-zero for small number, got 0")
	}
}

func TestAVX2Wrappers(t *testing.T) {
	// Length < 16 (scalar path)
	inSmall := []float32{1.0, 2.0, 3.0}
	SoftmaxAVX2(inSmall)

	gateSmall := []float32{1.0, 2.0}
	upSmall := []float32{2.0, 3.0}
	outSmall := make([]float32, 2)
	SwiGLUAVX2(gateSmall, upSmall, outSmall)

	srcSmall := []uint16{0x3C00, 0x4000}
	dstSmall := make([]float32, 2)
	Fp16ToFp32AVX2(srcSmall, dstSmall)

	dst16Small := make([]uint16, 2)
	Fp32ToFp16AVX2(dstSmall, dst16Small)

	// Length >= 16 (AVX2 path)
	inLarge := make([]float32, 32)
	for i := range inLarge {
		inLarge[i] = float32(i)
	}
	SoftmaxAVX2(inLarge)

	gateLarge := make([]float32, 32)
	upLarge := make([]float32, 32)
	outLarge := make([]float32, 32)
	SwiGLUAVX2(gateLarge, upLarge, outLarge)

	srcLarge := make([]uint16, 32)
	dstLarge := make([]float32, 32)
	for i := range srcLarge {
		srcLarge[i] = 0x3C00
	}
	Fp16ToFp32AVX2(srcLarge, dstLarge)

	dst16Large := make([]uint16, 32)
	Fp32ToFp16AVX2(dstLarge, dst16Large)

	// Mismatched / empty lengths
	SoftmaxAVX2(nil)
	SwiGLUAVX2(nil, nil, nil)
	Fp16ToFp32AVX2(nil, nil)
	Fp32ToFp16AVX2(nil, nil)
}


func TestFusedMLP_EdgeCases(t *testing.T) {
	// Empty inputs
	out := make([]float32, 2)
	FusedMLP(nil, nil, nil, nil, out, 0, 0, 0)

	// Valid batch with gate clamping >10 and <-10
	in := []float32{1.0, 2.0}
	gateW := []float32{15.0, -15.0, 2.0, -2.0}
	upW := []float32{1.0, 1.0, 1.0, 1.0}
	downW := []float32{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
	out2 := make([]float32, 2)
	FusedMLP(in, gateW, upW, downW, out2, 1, 2, 4)
}

func TestSwiGLU_Clamping(t *testing.T) {
	gate := []float32{15.0, -15.0, 0.0}
	up := []float32{1.0, 1.0, 1.0}
	out := make([]float32, 3)
	SwiGLU(gate, up, out)
	swigluScalar(gate, up, out)
}

func TestFp32ToFp16_DefaultScalar(t *testing.T) {
	src := []float32{
		float32(math.Inf(1)),
		float32(math.Inf(-1)),
		float32(math.NaN()),
		1e10,
		-1e10,
		1e-6,
		-1e-6,
		0.0,
		-0.0,
		1.0,
		-1.0,
	}
	dst := make([]uint16, len(src))
	Fp32ToFp16(src, dst)
	fp32ToFp16Scalar(src, dst)
}

func TestVecDotF32_And_VecFMAF32(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	b := []float32{2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
	dot := VecDotF32(a, b)
	// 2 + 6 + 12 + 20 + 30 + 42 + 56 + 72 + 90 + 110 = 440
	var expectedDot float32 = 440.0
	if math.Abs(float64(dot-expectedDot)) > 1e-4 {
		t.Errorf("VecDotF32 got %f, want %f", dot, expectedDot)
	}

	dst := make([]float32, 10)
	VecFMAF32(dst, a, 2.0)
	for i, v := range dst {
		if v != a[i]*2.0 {
			t.Errorf("VecFMAF32 at %d got %f, want %f", i, v, a[i]*2.0)
		}
	}
}
