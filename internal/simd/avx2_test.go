//go:build amd64 || arm64

package simd

import (
	"math"
	"testing"
)

func TestSoftmaxAVX2(t *testing.T) {
	tests := []struct {
		name   string
		input  []float32
		expect []float32
	}{
		{
			name:   "simple",
			input:  []float32{1.0, 2.0, 3.0},
			expect: []float32{0.090030, 0.244728, 0.665241},
		},
		{
			name:   "negative values",
			input:  []float32{-1.0, 0.0, 1.0},
			expect: []float32{0.090030, 0.244728, 0.665241},
		},
		{
			name:   "large values",
			input:  []float32{1000.0, 1001.0, 1002.0},
			expect: []float32{0.090030, 0.244728, 0.665241},
		},
		{
			name:   "single element",
			input:  []float32{5.0},
			expect: []float32{1.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := make([]float32, len(tt.input))
			copy(input, tt.input)
			SoftmaxAVX2(input)
			for i, v := range input {
				if math.Abs(float64(v-tt.expect[i])) > 1e-5 {
					t.Errorf("SoftmaxAVX2()[%d] = %v, want %v", i, v, tt.expect[i])
				}
			}
		})
	}
}

func TestSwiGLUAVX2(t *testing.T) {
	gate := []float32{0.0, 1.0, -1.0, 10.0, -10.0}
	up := []float32{1.0, 2.0, 3.0, 4.0, 5.0}
	out := make([]float32, len(gate))
	SwiGLUAVX2(gate, up, out)

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
		if math.Abs(float64(v-expected[i])) > 1e-5 {
			t.Errorf("SwiGLUAVX2()[%d] = %v, want %v", i, v, expected[i])
		}
	}
}

func TestFp16ToFp32AVX2(t *testing.T) {
	tests := []struct {
		name  string
		input []uint16
	}{
		{
			name:  "zero",
			input: []uint16{0x0000},
		},
		{
			name:  "positive normal",
			input: []uint16{0x3C00}, // 1.0
		},
		{
			name:  "negative",
			input: []uint16{0xBC00}, // -1.0
		},
		{
			name:  "subnormal",
			input: []uint16{0x0400}, // smallest subnormal
		},
		{
			name:  "infinity",
			input: []uint16{0x7C00},
		},
		{
			name:  "negative infinity",
			input: []uint16{0xFC00},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.input))
			Fp16ToFp32AVX2(tt.input, dst)

			for i, h := range tt.input {
				bits := uint32(h)
				sign := (bits >> 15) & 0x1
				exp := (bits >> 10) & 0x1F
				mant := bits & 0x3FF

				var expected float32
				switch exp {
				case 0:
					if mant == 0 {
						expected = math.Float32frombits(sign << 31)
					} else {
						expected = math.Float32frombits((sign << 31) | ((127 - 14 - 10 + 10) << 23) | (mant << 13))
					}
				case 31:
					if mant == 0 {
						expected = math.Float32frombits((sign << 31) | 0x7F800000)
					} else {
						expected = math.Float32frombits((sign << 31) | 0x7F800000 | (mant << 13))
					}
				default:
					newExp := exp - 15 + 127
					expected = math.Float32frombits((sign << 31) | (newExp << 23) | (mant << 13))
				}

				if math.Abs(float64(dst[i]-expected)) > 1e-6 {
					t.Errorf("Fp16ToFp32AVX2()[%d] = %v, want %v", i, dst[i], expected)
				}
			}
		})
	}
}

func TestFp32ToFp16AVX2(t *testing.T) {
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
			name:  "small positive",
			input: []float32{1e-8},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]uint16, len(tt.input))
			Fp32ToFp16AVX2(tt.input, dst)

			for i, f := range tt.input {
				bits := math.Float32bits(f)
				sign := bits >> 31
				exp := (bits >> 23) & 0xFF
				mant := bits & 0x7FFFFF

				var expected uint16
				switch exp {
				case 0:
					expected = 0
				case 255:
					expected = uint16(sign<<15) | 0x7C00 | uint16(mant>>9)
				default:
					newExp := int(exp) - 127 + 15
					if newExp >= 31 {
						expected = uint16(sign<<15) | 0x7C00
					} else if newExp <= 0 {
						shift := uint32(1 - newExp)
						m := mant | 0x800000
						expected = uint16(sign<<15) | uint16(m>>(9+shift))
					} else {
						expected = uint16(sign<<15) | uint16(newExp<<10) | uint16(mant>>13)
					}
				}

				if dst[i] != expected {
					t.Errorf("Fp32ToFp16AVX2()[%d] = 0x%04X, want 0x%04X", i, dst[i], expected)
				}
			}
		})
	}
}

func BenchmarkSoftmaxAVX2(b *testing.B) {
	data := make([]float32, 4096)
	for i := range data {
		data[i] = float32(i % 100)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		input := make([]float32, len(data))
		copy(input, data)
		SoftmaxAVX2(input)
	}
}

func BenchmarkSwiGLUAVX2(b *testing.B) {
	gate := make([]float32, 4096)
	up := make([]float32, 4096)
	out := make([]float32, 4096)
	for i := range gate {
		gate[i] = float32(i % 100)
		up[i] = float32(i % 100)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SwiGLUAVX2(gate, up, out)
	}
}
