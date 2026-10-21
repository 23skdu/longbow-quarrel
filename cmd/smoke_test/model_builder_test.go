//go:build (darwin && metal) || (linux && amd64 && cuda && cgo) || (linux && !cuda)

package main

import (
	"math"

	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

// The mock models of the smoke tests are written with the GGUF writer.
//
// The generators used to write the file by hand: every tensor got the
// offset zero and the file ended with a block of dummy data, so it was
// never a valid GGUF. The engine now rejects such a file — it validates
// the tensor table before it touches a weight — and the tests reported
// that the model could not be loaded.

// buildTestModel writes a model with the given metadata and tensors.
func buildTestModel(path string, kv map[string]interface{}, tensors []gguf.TensorData) error {
	return gguf.WriteFile(path, kv, tensors, 32)
}

// f32Tensor builds a float32 tensor with a deterministic, non-zero
// weight.
//
// A zero weight makes every output zero, and a test that checks the
// output of a model with zero weights proves nothing.
func f32Tensor(name string, dims []uint64, seed int) gguf.TensorData {
	count := 1
	for _, d := range dims {
		count *= int(d)
	}
	if count < 1 {
		count = 1
		dims = []uint64{1}
	}

	data := make([]byte, count*4)
	state := uint32(seed*2654435761 + 1)
	for i := 0; i < count; i++ {
		state = state*1664525 + 1013904223
		// A multiple of a sixteenth, so that the value is exactly
		// representable in half precision.
		bits := math.Float32bits(float32(int32(state>>8)&0xF) / 16.0)
		data[i*4] = byte(bits)
		data[i*4+1] = byte(bits >> 8)
		data[i*4+2] = byte(bits >> 16)
		data[i*4+3] = byte(bits >> 24)
	}

	return gguf.TensorData{
		Name:       name,
		Dimensions: dims,
		Type:       gguf.GGMLTypeF32,
		Data:       data,
	}
}
