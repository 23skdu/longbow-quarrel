package gguf

import (
	"encoding/binary"
	"fmt"

	"github.com/23skdu/longbow-quarrel/internal/mathx"
)

// Q8_0 is the block format of 32 int8 values with one half-precision scale.
// The repository could dequantize it but not produce it, so a model could
// not be converted to it.

const (
	// Q8_0BlockSize is the number of values in one Q8_0 block.
	Q8_0BlockSize = 32
	// Q8_0BlockSizeBytes is the size of one encoded Q8_0 block: a
	// half-precision scale followed by 32 int8 values.
	Q8_0BlockSizeBytes = 34
)

// QuantizeWeightsToQ8_0 encodes float32 values into Q8_0 blocks.
//
// It is the inverse of DequantizeQ8_0: each block of 32 values is scaled by
// the largest absolute value in it, and the scale is stored in half
// precision.
func QuantizeWeightsToQ8_0(weights []float32, numElements int) ([]byte, error) {
	if len(weights) != numElements {
		return nil, fmt.Errorf("weight length mismatch: got %d, want %d", len(weights), numElements)
	}
	if numElements%Q8_0BlockSize != 0 {
		return nil, fmt.Errorf("QuantizeWeightsToQ8_0: numElements %d must be a multiple of %d",
			numElements, Q8_0BlockSize)
	}

	numBlocks := numElements / Q8_0BlockSize
	out := make([]byte, numBlocks*Q8_0BlockSizeBytes)

	for b := 0; b < numBlocks; b++ {
		block := weights[b*Q8_0BlockSize : (b+1)*Q8_0BlockSize]

		var amax float32
		for _, v := range block {
			a := float32(mathx.Abs(float64(v)))
			if a > amax {
				amax = a
			}
		}

		// The scale maps the int8 range [-127, 127] onto the block.
		scale := float32(0)
		if amax > 0 {
			scale = amax / 127.0
		}
		invScale := float32(0)
		if scale > 0 {
			invScale = 1.0 / scale
		}

		offset := b * Q8_0BlockSizeBytes
		binary.LittleEndian.PutUint16(out[offset:offset+2], Float32ToFloat16(scale))

		for j, v := range block {
			q := int32(mathx.Round(float64(v * invScale)))
			if q > 127 {
				q = 127
			}
			if q < -127 {
				q = -127
			}
			out[offset+2+j] = byte(int8(q)) // #nosec G115 -- the value is clamped to the int8 range
		}
	}

	return out, nil
}
