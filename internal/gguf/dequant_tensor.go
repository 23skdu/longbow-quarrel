package gguf

import (
	"encoding/binary"
	"fmt"
)

// DequantizeTensor returns the float32 values of a tensor, whatever its
// stored type.
//
// The quantization functions in this package need it to requantize a
// model: a model that is already quantized has to be dequantized before it
// can be quantized into a different format.
func DequantizeTensor(t *TensorInfo) ([]float32, error) {
	if t == nil {
		return nil, fmt.Errorf("no tensor")
	}

	numElements := uint64(1)
	for _, d := range t.Dimensions {
		numElements *= d
	}
	n := int(numElements) // #nosec G115 -- the dimensions are bounded by the model

	switch t.Type {
	case GGMLTypeF32:
		if len(t.Data) < n*4 {
			return nil, fmt.Errorf("the tensor %q holds %d bytes, want %d", t.Name, len(t.Data), n*4)
		}
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			out[i] = math_Float32frombits(binary.LittleEndian.Uint32(t.Data[i*4:]))
		}
		return out, nil
	case GGMLTypeF16:
		return DequantizeF16(t.Data, n), nil
	case GGMLTypeBF16:
		return DequantizeBF16(t.Data, n), nil
	case GGMLTypeQ4_K:
		return DequantizeQ4K(t.Data, n), nil
	case GGMLTypeQ5_K:
		return DequantizeQ5K(t.Data, n), nil
	case GGMLTypeQ6_K:
		return DequantizeQ6K(t.Data, n), nil
	case GGMLTypeQ3_K:
		return DequantizeQ3K(t.Data, n), nil
	case GGMLTypeQ2_K:
		return DequantizeQ2K(t.Data, n), nil
	case GGMLTypeQ4_0:
		return DequantizeQ4_0(t.Data, n), nil
	case GGMLTypeQ5_0:
		return DequantizeQ5_0(t.Data, n), nil
	case GGMLTypeQ8_0:
		return DequantizeQ8_0(t.Data, n), nil
	case GGMLTypeIQ4_XS:
		return DequantizeIQ4XS(t.Data, n), nil
	case GGMLTypeIQ4_NL:
		return DequantizeIQ4NL(t.Data, n), nil
	default:
		return nil, fmt.Errorf("the tensor %q has the unsupported type %v", t.Name, t.Type)
	}
}
