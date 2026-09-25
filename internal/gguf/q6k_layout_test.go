package gguf

import (
	"math"
	"testing"
)

// dequantQ6KReference is an independent transcription of dequant_q6_k_kernel
// in internal/device/cuda_kernels.cu, which shares its layout with the
// linear_q6k_* kernels in internal/device/kernels.metal.
//
// CPU inference decodes Q6_K through DequantizeQ6K, so if the two ever
// disagree the same weights produce different outputs on CPU and GPU.
func dequantQ6KReference(block []byte) []float32 {
	ql := block[0:128]
	qh := block[128:192]
	scales := block[192:208]
	d := Float16ToFloat32(uint16(block[208]) | uint16(block[209])<<8)

	out := make([]float32, BlockSizeQ6K)
	for g := 0; g < 16; g++ {
		s := d * float32(int8(scales[g])) // #nosec G115 -- int8 scale read from the block
		for k := 0; k < 16; k++ {
			i := g*16 + k
			q4 := ql[i/2]
			if i%2 == 0 {
				q4 &= 0x0F
			} else {
				q4 >>= 4
			}
			q2 := (qh[i/4] >> uint((i%4)*2)) & 0x03
			out[i] = s * (float32(int8((q2<<4)|q4)) - 32) // #nosec G115 -- 6-bit value in [0,63]
		}
	}
	return out
}

func TestDequantizeQ6K_MatchesCUDAKernel(t *testing.T) {
	block := make([]byte, BlockSizeQ6KBytes)
	for i := 0; i < 208; i++ {
		block[i] = byte((i*37 + 11) & 0xFF) // #nosec G115 -- masked to byte range
	}
	block[208], block[209] = 0x3C, 0x38

	want := dequantQ6KReference(block)
	got := DequantizeQ6K(block, BlockSizeQ6K)
	if len(got) != BlockSizeQ6K {
		t.Fatalf("expected %d elements, got %d", BlockSizeQ6K, len(got))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("element %d: DequantizeQ6K=%v, CUDA kernel=%v", i, got[i], want[i])
		}
	}
}

func TestDequantizeQ6K_ZeroBlock(t *testing.T) {
	got := DequantizeQ6K(make([]byte, BlockSizeQ6KBytes), BlockSizeQ6K)
	for i, v := range got {
		if v != 0 {
			t.Fatalf("element %d of an all-zero block: got %v, want 0", i, v)
		}
	}
}

func TestQuantizeQ6K_RoundTrip(t *testing.T) {
	const numBlocks = 4
	src := make([]float32, numBlocks*BlockSizeQ6K)

	// Block 0: smooth deterministic signal in [-1, 1].
	for i := range src[:BlockSizeQ6K] {
		src[i] = float32(math.Sin(float64(i)*0.17)) * 0.9
	}

	// Block 1: left as all zeros (the all-zero / degenerate scale path).

	// Block 2: four orders of magnitude of dynamic range between groups, so
	// several groups have to fall back to the minimum int8 scale.
	for g := 0; g < 16; g++ {
		mag := float32(math.Pow(10, -float64(g)/4))
		for i := g * 16; i < g*16+16; i++ {
			src[2*BlockSizeQ6K+i] = mag * float32((i%7)-3) / 3
		}
	}

	// Block 3: single spike, everything else zero.
	src[3*BlockSizeQ6K+200] = -1000

	enc := QuantizeQ6K(src)
	if want := numBlocks * BlockSizeQ6KBytes; len(enc) != want {
		t.Fatalf("encoded size: got %d, want %d", len(enc), want)
	}

	got := DequantizeQ6K(enc, len(src))
	if len(got) != len(src) {
		t.Fatalf("decoded size: got %d, want %d", len(got), len(src))
	}

	for b := 0; b < numBlocks; b++ {
		lo, hi := b*BlockSizeQ6K, (b+1)*BlockSizeQ6K
		var amax float32
		for i := lo; i < hi; i++ {
			a := src[i]
			if a < 0 {
				a = -a
			}
			if a > amax {
				amax = a
			}
		}
		// The per-group step is amax/31 and the shared f16 scale rounds, so
		// the worst-case error is half a step plus the f16 rounding slack.
		tol := float64(amax) / 31 / 2 * 1.01
		for i := lo; i < hi; i++ {
			err := math.Abs(float64(got[i] - src[i]))
			if err > tol {
				t.Errorf("block %d element %d: got %v, want %v (|err|=%g > tol %g)",
					b, i%BlockSizeQ6K, got[i], src[i], err, tol)
			}
		}
	}
}

func TestQuantizeQ6K_TruncatesPartialBlock(t *testing.T) {
	src := make([]float32, BlockSizeQ6K+7)
	enc := QuantizeQ6K(src)
	if want := BlockSizeQ6KBytes; len(enc) != want {
		t.Fatalf("encoded size: got %d, want %d", len(enc), want)
	}
}
