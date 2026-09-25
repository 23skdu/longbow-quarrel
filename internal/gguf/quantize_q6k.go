package gguf

import (
	"encoding/binary"
	"math"
)

// BlockSizeQ6KBytes is the encoded size of one Q6_K block:
// 128 bytes of low nibbles + 64 bytes of high bits + 16 int8 scales + f16 scale.
const BlockSizeQ6KBytes = 210

// QuantizeQ6K encodes f32 values into Q6_K blocks.
//
// It is the exact inverse of DequantizeQ6K and uses the same block layout as
// dequant_q6_k_kernel (internal/device/cuda_kernels.cu) and the linear_q6k_*
// kernels (internal/device/kernels.metal), so a round trip through
// QuantizeQ6K and DequantizeQ6K reproduces the input to within the
// quantization step of the format.
//
// data is truncated to a whole number of 256-value blocks; use
// BlockSizeQ6KBytes*ceil(len(data)/256) for the required output size.
func QuantizeQ6K(data []float32) []byte {
	numBlocks := len(data) / BlockSizeQ6K
	out := make([]byte, numBlocks*BlockSizeQ6KBytes)

	for b := 0; b < numBlocks; b++ {
		encodeQ6KBlock(data[b*BlockSizeQ6K:(b+1)*BlockSizeQ6K], out[b*BlockSizeQ6KBytes:(b+1)*BlockSizeQ6KBytes])
	}
	return out
}

// encodeQ6KBlock encodes 256 f32 values into a 210-byte Q6_K block.
//
// Each of the 16 scales owns one group of 16 consecutive values. The per-group
// step is amax/31 so the signed 6-bit range [-32, 31] is used, and the shared
// f16 super-block scale d is chosen so the largest step maps to int8 127.
func encodeQ6KBlock(src []float32, dst []byte) {
	ql := dst[0:128]
	qh := dst[128:192]
	scales := dst[192:208]

	var steps [16]float32
	var maxStep float32
	for g := 0; g < 16; g++ {
		var amax float32
		for i := g * 16; i < g*16+16; i++ {
			a := src[i]
			if a < 0 {
				a = -a
			}
			if a > amax {
				amax = a
			}
		}
		step := amax / 31
		steps[g] = step
		if step > maxStep {
			maxStep = step
		}
	}

	var d float32
	if maxStep > 0 {
		d = maxStep / 127
	}
	dBits := Float32ToFloat16(d)
	dEff := Float16ToFloat32(dBits)
	binary.LittleEndian.PutUint16(dst[208:210], dBits)

	for g := 0; g < 16; g++ {
		sc := 1
		if dEff > 0 {
			sc = int(math.Round(float64(steps[g] / dEff)))
		}
		if sc < 1 {
			sc = 1
		} else if sc > 127 {
			sc = 127
		}
		scales[g] = byte(int8(sc)) // #nosec G115 -- sc is clamped to [1,127]

		step := dEff * float32(sc)
		for i := g * 16; i < g*16+16; i++ {
			q := 0
			if step > 0 {
				q = int(math.Round(float64(src[i] / step)))
			}
			if q < -32 {
				q = -32
			} else if q > 31 {
				q = 31
			}
			raw := uint8(q + 32) // #nosec G115 -- q is clamped to [-32,31] so raw is in [0,63]

			if i%2 == 0 {
				ql[i/2] = (ql[i/2] & 0xF0) | (raw & 0x0F)
			} else {
				ql[i/2] = (ql[i/2] & 0x0F) | ((raw & 0x0F) << 4)
			}
			qh[i/4] |= (raw >> 4) << uint((i%4)*2)
		}
	}
}
