//go:build amd64 && cgo && avx512

package simd

import (
	"math"
	"math/rand"
	"testing"
)

func TestSoftmax(t *testing.T) {
	tests := []struct {
		name   string
		input  []float32
		expect []float32
		tol    float64
	}{
		{
			name:   "simple",
			input:  []float32{1.0, 2.0, 3.0},
			expect: []float32{0.090030, 0.244728, 0.665241},
			tol:    1e-4,
		},
		{
			name:   "negative values",
			input:  []float32{-1.0, 0.0, 1.0},
			expect: []float32{0.090030, 0.244728, 0.665241},
			tol:    1e-4,
		},
		{
			name:   "large values",
			input:  []float32{1000.0, 1001.0, 1002.0},
			expect: []float32{0.090030, 0.244728, 0.665241},
			tol:    1e-4,
		},
		{
			name:   "single element",
			input:  []float32{5.0},
			expect: []float32{1.0},
			tol:    1e-6,
		},
		{
			name:   "empty",
			input:  []float32{},
			expect: []float32{},
			tol:    0,
		},
		{
			name:   "power of 2",
			input:  []float32{1, 2, 4, 8},
			expect: nil,
			tol:    1e-4,
		},
		{
			name:   "random",
			input:  nil,
			expect: nil,
			tol:    1e-4,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var input []float32
			if tt.input != nil {
				input = make([]float32, len(tt.input))
				copy(input, tt.input)
			} else {
				input = make([]float32, 4096)
				rand.New(rand.NewSource(42))
				for i := range input {
					input[i] = rand.Float32() * 10
				}
			}

			Softmax(input)

			if tt.expect != nil {
				for i, v := range input {
					if math.Abs(float64(v-tt.expect[i])) > tt.tol {
						t.Errorf("Softmax()[%d] = %v, want %v (diff=%v)", i, v, tt.expect[i], math.Abs(float64(v-tt.expect[i])))
					}
				}
			} else {
				sum := float64(0)
				for _, v := range input {
					sum += float64(v)
				}
				if math.Abs(sum-1.0) > tt.tol {
					t.Errorf("Softmax() sum = %v, want 1.0", sum)
				}
			}
		})
	}
}

func TestSwiGLU(t *testing.T) {
	gate := []float32{0.0, 1.0, -1.0, 10.0, -10.0}
	up := []float32{1.0, 2.0, 3.0, 4.0, 5.0}
	out := make([]float32, len(gate))
	SwiGLU(gate, up, out)

	sigmoid := func(x float32) float32 {
		return float32(1.0) / (float32(1.0) + float32(math.Exp(float64(-x))))
	}

	expected := []float32{
		0.0 * sigmoid(0.0) * 1.0,
		1.0 * sigmoid(1.0) * 2.0,
		-1.0 * sigmoid(-1.0) * 3.0,
		10.0 * sigmoid(10.0) * 4.0,
		-10.0 * sigmoid(-10.0) * 5.0,
	}

	for i, v := range out {
		if math.Abs(float64(v-expected[i])) > 1e-4 {
			t.Errorf("SwiGLU()[%d] = %v, want %v", i, v, expected[i])
		}
	}
}

func TestSwiGLU_Random(t *testing.T) {
	const N = 4096
	gate := make([]float32, N)
	up := make([]float32, N)
	out := make([]float32, N)

	rng := rand.New(rand.NewSource(42))
	for i := range gate {
		gate[i] = (rng.Float32() - 0.5) * 20
		up[i] = rng.Float32() * 10
	}

	SwiGLU(gate, up, out)

	for i, v := range out {
		if math.IsInf(float64(v), 0) || math.IsNaN(float64(v)) {
			t.Errorf("SwiGLU()[%d] = %v (Inf=%v, NaN=%v)", i, v, math.IsInf(float64(v), 0), math.IsNaN(float64(v)))
		}
		if v < -1e6 || v > 1e6 {
			t.Errorf("SwiGLU()[%d] out of range: %v", i, v)
		}
	}
}

func TestFp16ToFp32(t *testing.T) {
	tests := []struct {
		name  string
		input []uint16
	}{
		{
			name:  "zero",
			input: []uint16{0x0000},
		},
		{
			name:  "one",
			input: []uint16{0x3C00},
		},
		{
			name:  "negative",
			input: []uint16{0xBC00},
		},
		{
			name:  "subnormal",
			input: []uint16{0x0400},
		},
		{
			name:  "infinity",
			input: []uint16{0x7C00},
		},
		{
			name:  "neg infinity",
			input: []uint16{0xFC00},
		},
		{
			name:  "random",
			input: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var src []uint16
			if tt.input != nil {
				src = tt.input
			} else {
				src = make([]uint16, 4096)
				rng := rand.New(rand.NewSource(42))
				for i := range src {
					src[i] = uint16(rng.Uint32())
				}
			}

			dst := make([]float32, len(src))
			Fp16ToFp32(src, dst)

			for i, h := range src {
				if math.IsInf(float64(dst[i]), 0) && !math.IsInf(float64(fp16ToFp32(h)), 0) {
					t.Errorf("Fp16ToFp32()[%d] = Inf, want %v", i, fp16ToFp32(h))
				}
				if math.IsNaN(float64(dst[i])) && !math.IsNaN(float64(fp16ToFp32(h))) {
					t.Errorf("Fp16ToFp32()[%d] = NaN, want %v", i, fp16ToFp32(h))
				}
			}
		})
	}
}

func TestFp32ToFp16(t *testing.T) {
	tests := []struct {
		name  string
		input []float32
	}{
		{
			name:  "zero",
			input: []float32{0.0},
		},
		{
			name:  "one",
			input: []float32{1.0},
		},
		{
			name:  "negative one",
			input: []float32{-1.0},
		},
		{
			name:  "small pos",
			input: []float32{1e-8},
		},
		{
			name:  "random",
			input: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var src []float32
			if tt.input != nil {
				src = tt.input
			} else {
				src = make([]float32, 4096)
				rng := rand.New(rand.NewSource(42))
				for i := range src {
					src[i] = rng.Float32()*100 - 50
				}
			}

			dst := make([]uint16, len(src))
			Fp32ToFp16(src, dst)

			for i, f := range src {
				if f == 0 && dst[i] != 0 {
					t.Errorf("Fp32ToFp16()[%d] = 0x%04X, want 0 for zero", i, dst[i])
				}
			}
		})
	}
}

func TestRMSNorm(t *testing.T) {
	tests := []struct {
		name string
		rows int
		cols int
		eps  float32
	}{
		{
			name: "simple 4x8",
			rows: 4,
			cols: 8,
			eps:  1e-5,
		},
		{
			name: "medium 32x128",
			rows: 32,
			cols: 128,
			eps:  1e-5,
		},
		{
			name: "large 64x256",
			rows: 64,
			cols: 256,
			eps:  1e-5,
		},
		{
			name: "random",
			rows: 0,
			cols: 0,
			eps:  1e-5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rows, cols := tt.rows, tt.cols
			if rows == 0 {
				rng := rand.New(rand.NewSource(42))
				rows = 1 + rng.Intn(64)
				cols = 8 + rng.Intn(256)
			}

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

			RMSNorm(input, weight, output, rows, cols, tt.eps)

			for i := range output {
				if math.IsInf(float64(output[i]), 0) || math.IsNaN(float64(output[i])) {
					t.Errorf("RMSNorm()[%d] = %v (Inf=%v, NaN=%v)", i, output[i], math.IsInf(float64(output[i]), 0), math.IsNaN(float64(output[i])))
				}
			}
		})
	}
}

func TestGetSIMDLevel(t *testing.T) {
	level := GetSIMDLevel()
	if level < 0 || level > 2 {
		t.Errorf("GetSIMDLevel() = %d, want 0-2", level)
	}
	t.Logf("SIMD Level: %d", level)
}

func TestMatmul(t *testing.T) {
	tests := []struct {
		name    string
		m, n, k int
	}{
		{
			name: "small",
			m:    4,
			n:    8,
			k:    4,
		},
		{
			name: "medium",
			m:    16,
			n:    32,
			k:    16,
		},
		{
			name: "large",
			m:    32,
			n:    64,
			k:    32,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a := make([]float32, tt.m*tt.k)
			b := make([]float32, tt.k*tt.n)
			c := make([]float32, tt.m*tt.n)

			rng := rand.New(rand.NewSource(42))
			for i := range a {
				a[i] = (rng.Float32() - 0.5) * 2
			}
			for i := range b {
				b[i] = (rng.Float32() - 0.5) * 2
			}

			Matmul(a, b, c, tt.m, tt.n, tt.k)

			for i := range c {
				if math.IsInf(float64(c[i]), 0) || math.IsNaN(float64(c[i])) {
					t.Errorf("Matmul()[%d] = %v", i, c[i])
				}
			}
		})
	}
}

func TestFusedAttention(t *testing.T) {
	tests := []struct {
		name    string
		batch   int
		heads   int
		seqLen  int
		headDim int
		scale   float32
	}{
		{
			name:    "small",
			batch:   1,
			heads:   4,
			seqLen:  8,
			headDim: 16,
			scale:   0.25,
		},
		{
			name:    "medium",
			batch:   2,
			heads:   8,
			seqLen:  32,
			headDim: 32,
			scale:   0.125,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			kvSeqLen := tt.seqLen
			totalElems := tt.batch * tt.heads * tt.seqLen * tt.headDim
			q := make([]float32, totalElems)
			k := make([]float32, tt.batch*tt.heads*kvSeqLen*tt.headDim)
			v := make([]float32, tt.batch*tt.heads*kvSeqLen*tt.headDim)
			output := make([]float32, totalElems)

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

			FusedAttention(q, k, v, output, tt.batch, tt.heads, tt.seqLen, kvSeqLen, tt.headDim, tt.scale)

			for i, o := range output {
				if math.IsInf(float64(o), 0) || math.IsNaN(float64(o)) {
					t.Errorf("FusedAttention()[%d] = %v (Inf=%v, NaN=%v)", i, o, math.IsInf(float64(o), 0), math.IsNaN(float64(o)))
				}
			}
		})
	}
}

func TestFusedMLP(t *testing.T) {
	tests := []struct {
		name      string
		batch     int
		dim       int
		hiddenDim int
	}{
		{
			name:      "small",
			batch:     1,
			dim:       256,
			hiddenDim: 1024,
		},
		{
			name:      "medium",
			batch:     2,
			dim:       512,
			hiddenDim: 2048,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := make([]float32, tt.batch*tt.dim)
			gateW := make([]float32, tt.hiddenDim)
			upW := make([]float32, tt.hiddenDim)
			downW := make([]float32, tt.dim*tt.hiddenDim)
			output := make([]float32, tt.batch*tt.dim)

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

			FusedMLP(input, gateW, upW, downW, output, tt.batch, tt.dim, tt.hiddenDim)

			for i, o := range output {
				if math.IsInf(float64(o), 0) || math.IsNaN(float64(o)) {
					t.Errorf("FusedMLP()[%d] = %v", i, o)
				}
			}
		})
	}
}

func TestRoPE(t *testing.T) {
	tests := []struct {
		name    string
		batch   int
		heads   int
		seqLen  int
		headDim int
		theta   float32
	}{
		{
			name:    "small",
			batch:   1,
			heads:   4,
			seqLen:  8,
			headDim: 16,
			theta:   10000,
		},
		{
			name:    "medium",
			batch:   2,
			heads:   8,
			seqLen:  32,
			headDim: 32,
			theta:   10000,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			totalElems := tt.batch * tt.heads * tt.seqLen * tt.headDim
			tensor := make([]float32, totalElems)
			positions := make([]int, tt.seqLen)

			rng := rand.New(rand.NewSource(42))
			for i := range tensor {
				tensor[i] = (rng.Float32() - 0.5) * 2
			}
			for i := range positions {
				positions[i] = i
			}

			RoPE(tensor, positions, tt.batch, tt.heads, tt.seqLen, tt.headDim, tt.theta)

			for i, v := range tensor {
				if math.IsInf(float64(v), 0) || math.IsNaN(float64(v)) {
					t.Errorf("RoPE()[%d] = %v", i, v)
				}
			}
		})
	}
}

func TestAVX512_SoftmaxAVX2_Compatibility(t *testing.T) {
	input := []float32{1.0, 2.0, 3.0}
	SoftmaxAVX2(input)
	expected := []float32{0.090030, 0.244728, 0.665241}
	for i, v := range input {
		if math.Abs(float64(v-expected[i])) > 1e-4 {
			t.Errorf("SoftmaxAVX2()[%d] = %v, want %v", i, v, expected[i])
		}
	}
}

func TestAVX512_SwiGLUAVX2_Compatibility(t *testing.T) {
	gate := []float32{0.0, 1.0, -1.0}
	up := []float32{1.0, 2.0, 3.0}
	out := make([]float32, 3)
	SwiGLUAVX2(gate, up, out)
	for i, v := range out {
		if math.IsInf(float64(v), 0) || math.IsNaN(float64(v)) {
			t.Errorf("SwiGLUAVX2()[%d] = %v", i, v)
		}
	}
}

func TestAVX512_FusedAttentionScalar(t *testing.T) {
	const (
		batch    = 1
		heads    = 2
		seqLen   = 4
		kvSeqLen = 4
		headDim  = 8
		scale    = float32(0.3535)
	)
	total := batch * heads * seqLen * headDim
	q := make([]float32, total)
	k := make([]float32, total)
	v := make([]float32, total)
	out := make([]float32, total)
	for i := range q {
		q[i] = float32(i%7) * 0.1
		k[i] = float32(i%5) * 0.1
		v[i] = float32(i%3) * 0.1
	}

	fusedAttentionScalar(q, k, v, out, batch, heads, seqLen, kvSeqLen, headDim, scale)

	for i, val := range out {
		if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
			t.Errorf("fusedAttentionScalar out[%d] = %v", i, val)
		}
	}
}

func TestAVX512_FusedMLPScalar(t *testing.T) {
	const (
		batch     = 1
		dim       = 16
		hiddenDim = 32
	)
	input := make([]float32, batch*dim)
	gateW := make([]float32, hiddenDim)
	upW := make([]float32, hiddenDim)
	downW := make([]float32, dim*hiddenDim)
	out := make([]float32, batch*dim)
	for i := range input {
		input[i] = 0.5
	}
	for i := range gateW {
		gateW[i] = 0.2
		upW[i] = 0.3
	}
	for i := range downW {
		downW[i] = 0.1
	}

	fusedMLPScalar(input, gateW, upW, downW, out, batch, dim, hiddenDim)

	for i, val := range out {
		if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
			t.Errorf("fusedMLPScalar out[%d] = %v", i, val)
		}
	}
}

func TestAVX512_VNNIKernels(t *testing.T) {
	dataQ8 := make([]byte, 34) // 1 block of Q8_0: 2 bytes d (f16) + 32 bytes qs (int8)
	dataQ8[0] = 0x00
	dataQ8[1] = 0x3c // 1.0 in f16
	for i := 2; i < 34; i++ {
		dataQ8[i] = 1
	}
	vec := make([]float32, 32)
	for i := range vec {
		vec[i] = 1.0
	}

	resQ8 := VecDotQ8_0_VNNI(dataQ8, vec)
	if math.IsNaN(float64(resQ8)) {
		t.Errorf("VecDotQ8_0_VNNI returned NaN")
	}

	dataQ4 := make([]byte, 144) // 1 block of Q4_K
	dataQ4[0] = 0x00
	dataQ4[1] = 0x3c // d = 1.0
	resQ4 := VecDotQ4_K_VNNI(dataQ4, make([]float32, 256))
	if math.IsNaN(float64(resQ4)) {
		t.Errorf("VecDotQ4_K_VNNI returned NaN")
	}

	if HasAVXVNNI() {
		resC8 := dotQ8_0VNNIC(dataQ8, vec, 32)
		if math.IsNaN(float64(resC8)) {
			t.Errorf("dotQ8_0VNNIC returned NaN")
		}
		resC4 := dotQ4KVNNIC(dataQ4, make([]float32, 256), 256)
		if math.IsNaN(float64(resC4)) {
			t.Errorf("dotQ4KVNNIC returned NaN")
		}
	}
}

