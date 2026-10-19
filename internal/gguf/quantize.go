package gguf

import (
	"encoding/binary"
	"fmt"

	"github.com/23skdu/longbow-quarrel/internal/mathx"
)

const (
	Q4KBlockSize      = 256
	Q4KBlockSizeBytes = 144
)

func Float32ToFloat16(f float32) uint16 {
	bits := mathx.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits>>23)&0xFF) - 127 + 15
	frac := bits & 0x7FFFFF

	if exp <= 0 {
		exp = 0
		frac = 0
	} else if exp >= 31 {
		exp = 31
		frac = 0
	}

	return sign | uint16(exp<<10) | uint16(frac>>13) // #nosec G115 -- safe bit manipulation for float16
}

// QuantizeWeightsToQ4K encodes float32 values into Q4_K blocks.
//
// The block layout is the GGML one: a half-precision super-block scale `d`
// for the quantized values, a half-precision super-block scale `dmin` for
// the minimums, twelve bytes of packed six-bit scales and minimums, and
// 128 bytes of four-bit quantized values.
//
// The encoder used to write a different layout from the one the decoder
// read, so a quantized model could not be read back. It is also the layout
// the CUDA kernel reads, so a model quantized here is read the same way on
// the CPU and on the GPU.
func QuantizeWeightsToQ4K(weights []float32, numElements int) ([]byte, error) {
	if numElements%Q4KBlockSize != 0 {
		return nil, fmt.Errorf("QuantizeWeightsToQ4K: numElements %d must be multiple of %d", numElements, Q4KBlockSize)
	}
	if len(weights) < numElements {
		return nil, fmt.Errorf("QuantizeWeightsToQ4K: got %d weights, want %d", len(weights), numElements)
	}

	numBlocks := numElements / Q4KBlockSize
	out := make([]byte, numBlocks*Q4KBlockSizeBytes)

	for i := 0; i < numBlocks; i++ {
		block := weights[i*Q4KBlockSize : (i+1)*Q4KBlockSize]

		// Each group of 32 values is represented by
		//     d*sc[g]*q - dmin*m[g]
		// with q in [0, 15]. The group's lowest value is the offset and
		// its highest value is the offset plus fifteen steps.
		var lo, hi [8]float32
		var step [8]float32
		maxStep := float32(0)
		maxOffset := float32(0)

		for g := 0; g < 8; g++ {
			group := block[g*32 : (g+1)*32]
			low, high := group[0], group[0]
			for _, v := range group {
				if v < low {
					low = v
				}
				if v > high {
					high = v
				}
			}
			// The offset is always subtracted, so a group whose values
			// are all positive is represented from zero upwards.
			if low > 0 {
				low = 0
			}
			if high < 0 {
				high = 0
			}
			lo[g], hi[g] = low, high
			step[g] = (high - low) / 15.0
			if step[g] > maxStep {
				maxStep = step[g]
			}
			if -low > maxOffset {
				maxOffset = -low
			}
		}

		d := float32(1)
		if maxStep > 0 {
			d = maxStep / 63.0
		}
		dmin := float32(1)
		if maxOffset > 0 {
			dmin = maxOffset / 63.0
		}

		var sc, m [8]uint8
		for g := 0; g < 8; g++ {
			sc[g] = uint8(mathx.Round(float64(step[g] / d))) // #nosec G115 -- the ratio is at most 63
			if sc[g] > 63 {
				sc[g] = 63
			}
			m[g] = uint8(mathx.Round(float64(-lo[g] / dmin))) // #nosec G115 -- the ratio is at most 63
			if m[g] > 63 {
				m[g] = 63
			}
		}

		blockOffset := i * Q4KBlockSizeBytes
		binary.LittleEndian.PutUint16(out[blockOffset:blockOffset+2], Float32ToFloat16(d))
		binary.LittleEndian.PutUint16(out[blockOffset+2:blockOffset+4], Float32ToFloat16(dmin))

		// The scales and the minimums are packed six bits each: the
		// first four of each in the low six bits of their own byte, and
		// the last four split between the high bits of the first four
		// and the low and high nibbles of bytes eight to eleven.
		scaleBytes := out[blockOffset+4 : blockOffset+16]
		for j := 0; j < 4; j++ {
			scaleBytes[j] = (sc[j] & 0x3F) | ((sc[j+4] >> 4) << 6)
			scaleBytes[j+4] = (m[j] & 0x3F) | ((m[j+4] >> 4) << 6)
		}
		for j := 0; j < 4; j++ {
			scaleBytes[8+j] = (sc[j+4] & 0x0F) | ((m[j+4] & 0x0F) << 4)
		}

		// The four-bit values: the first sixteen of a group are the low
		// nibbles and the next sixteen the high nibbles of the same
		// bytes.
		quants := out[blockOffset+16 : blockOffset+Q4KBlockSizeBytes]
		for g := 0; g < 8; g++ {
			group := block[g*32 : (g+1)*32]
			for j := 0; j < 32; j++ {
				var q int
				if step[g] > 0 {
					q = int(mathx.Round(float64((group[j] - lo[g]) / step[g])))
				}
				if q < 0 {
					q = 0
				}
				if q > 15 {
					q = 15
				}
				qsOffset := g*16 + j
				if j < 16 {
					quants[qsOffset] = uint8(q) & 0x0F
				} else {
					quants[qsOffset-16] |= (uint8(q) & 0x0F) << 4
				}
			}
		}
	}

	return out, nil
}

// DequantizeWeightsFromQ4K decodes a Q4_K weight matrix.
func DequantizeWeightsFromQ4K(data []byte, rows, cols int) ([]float32, error) {
	numElements := rows * cols
	return DequantizeQ4K(data, numElements), nil
}
