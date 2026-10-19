package gguf

import (
	"encoding/binary"
	"fmt"

	"github.com/23skdu/longbow-quarrel/internal/mathx"
)

type FP8Type int

const (
	FP8E4M3 FP8Type = iota
	FP8E5M2
)

func QuantizeToFP8E4M3(data []float32) ([]byte, error) {
	result := make([]byte, len(data))
	for i, f := range data {
		result[i] = quantizeToE4M3(float64(f))
	}
	return result, nil
}

func QuantizeToFP8E5M2(data []float32) ([]byte, error) {
	result := make([]byte, len(data))
	for i, f := range data {
		result[i] = quantizeToE5M2(float64(f))
	}
	return result, nil
}

// quantizeToE4M3 encodes a value in the E4M3 format.
//
// The body used to be a second, incompatible implementation of the
// conversion: it used a different exponent bias and a different mantissa
// convention from the decoder next to it, so the pair did not round-trip.
func quantizeToE4M3(f float64) byte {
	return Float32ToFP8E4M3(float32(f))
}

// quantizeToE5M2 encodes a value in the E5M2 format.
func quantizeToE5M2(f float64) byte {
	return Float32ToFP8E5M2(float32(f))
}

func DequantizeFromFP8E4M3(data []byte) ([]float32, error) {
	result := make([]float32, len(data))
	for i, b := range data {
		result[i] = float32(dequantizeE4M3(b))
	}
	return result, nil
}

func DequantizeFromFP8E5M2(data []byte) ([]float32, error) {
	result := make([]float32, len(data))
	for i, b := range data {
		result[i] = float32(dequantizeE5M2(b))
	}
	return result, nil
}

// dequantizeE4M3 decodes an E4M3 value.
func dequantizeE4M3(b byte) float64 {
	return float64(FP8E4M3ToFloat32(b))
}

// dequantizeE5M2 decodes an E5M2 value.
func dequantizeE5M2(b byte) float64 {
	return float64(FP8E5M2ToFloat32(b))
}

func QuantizeWeightsToFP8(weights []float32, numElements int, fp8Type FP8Type) ([]byte, error) {
	if len(weights) != numElements {
		return nil, fmt.Errorf("weight length mismatch: got %d, want %d", len(weights), numElements)
	}

	maxVal := float32(0)
	for _, w := range weights {
		absW := mathx.Abs(float64(w))
		if absW > float64(maxVal) {
			maxVal = float32(absW)
		}
	}

	if maxVal == 0 {
		return make([]byte, numElements), nil
	}

	scale := maxVal / 127.0

	scaledWeights := make([]float32, numElements)
	for i, w := range weights {
		scaledWeights[i] = w / scale
	}

	switch fp8Type {
	case FP8E4M3:
		return QuantizeToFP8E4M3(scaledWeights)
	case FP8E5M2:
		return QuantizeToFP8E5M2(scaledWeights)
	default:
		return nil, fmt.Errorf("unknown FP8 type: %v", fp8Type)
	}
}

func DequantizeWeightsFromFP8(data []byte, rows, cols int, fp8Type FP8Type, scale float32) ([]float32, error) {
	numElements := rows * cols
	if len(data) != numElements {
		return nil, fmt.Errorf("data length mismatch: got %d, want %d", len(data), numElements)
	}

	var result []float32
	var err error

	switch fp8Type {
	case FP8E4M3:
		result, err = DequantizeFromFP8E4M3(data)
	case FP8E5M2:
		result, err = DequantizeFromFP8E5M2(data)
	default:
		return nil, fmt.Errorf("unknown FP8 type: %v", fp8Type)
	}
	_ = err

	for i := range result {
		result[i] *= scale
	}

	return result, nil
}

type FP8Tensor struct {
	Data     []byte
	Type     FP8Type
	Shape    []int
	Scale    float32
	Original []float32
}

func NewFP8Tensor(data []float32, shape []int, fp8Type FP8Type) (*FP8Tensor, error) {
	numElements := 1
	for _, s := range shape {
		numElements *= s
	}

	quantized, err := QuantizeWeightsToFP8(data, numElements, fp8Type)
	if err != nil {
		return nil, err
	}

	maxVal := float32(0)
	for _, d := range data {
		absD := mathx.Abs(float64(d))
		if absD > float64(maxVal) {
			maxVal = float32(absD)
		}
	}

	scale := float32(0)
	if maxVal > 0 {
		scale = maxVal / 127.0
	}

	return &FP8Tensor{
		Data:     quantized,
		Type:     fp8Type,
		Shape:    shape,
		Scale:    scale,
		Original: data,
	}, nil
}

func (t *FP8Tensor) ToFloat32() ([]float32, error) {
	numElements := 1
	for _, s := range t.Shape {
		numElements *= s
	}
	return DequantizeWeightsFromFP8(t.Data, 1, numElements, t.Type, t.Scale)
}

func (t *FP8Tensor) ByteSize() int {
	return len(t.Data)
}

type FP8Config struct {
	Type         FP8Type
	PerTensor    bool
	BlockSize    int
	ScaleFloat32 float32
	ScaleBytes   []byte
}

func NewFP8Config(fp8Type FP8Type) FP8Config {
	return FP8Config{
		Type:         fp8Type,
		PerTensor:    true,
		BlockSize:    1,
		ScaleFloat32: 1.0,
		ScaleBytes:   make([]byte, 4),
	}
}

func (c *FP8Config) MarshalBinary() ([]byte, error) {
	data := make([]byte, 8)
	data[0] = byte(c.Type) // #nosec G115 -- int to byte for FP8 config type
	if c.PerTensor {
		data[1] = 1
	}
	binary.LittleEndian.PutUint32(data[2:6], mathx.Float32bits(c.ScaleFloat32))
	binary.LittleEndian.PutUint16(data[6:8], uint16(c.BlockSize)) // #nosec G115 -- safe: BlockSize is always small
	return data, nil
}

func (c *FP8Config) UnmarshalBinary(data []byte) error {
	if len(data) < 8 {
		return fmt.Errorf("FP8Config data too short: %d bytes", len(data))
	}
	c.Type = FP8Type(data[0])
	c.PerTensor = data[1] != 0
	c.ScaleFloat32 = mathx.Float32frombits(binary.LittleEndian.Uint32(data[2:6]))
	c.BlockSize = int(binary.LittleEndian.Uint16(data[6:8]))
	return nil
}

func FP8E4M3SizeBytes(numElements int) int {
	return numElements
}

func FP8E5M2SizeBytes(numElements int) int {
	return numElements
}
