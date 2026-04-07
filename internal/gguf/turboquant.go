package gguf

import (
	"fmt"
	"unsafe"
)

// GetTurboQuantMatrices attempts to find the TQ rotation and QJL matrices in the GGUF file.
func (f *GGUFFile) GetTurboQuantMatrices() ([]float32, []float32, error) {
	var rotation []float32
	var qjl []float32

	for _, t := range f.Tensors {
		if t.Name == "turboquant.rotation_matrix" {
			if t.Type != GGMLTypeF32 {
				return nil, nil, fmt.Errorf("turboquant.rotation_matrix must be F32")
			}
			rotation = bytesToFloat32(t.Data, int(t.SizeBytes()/4))
		}
		if t.Name == "turboquant.qjl_matrix" {
			if t.Type != GGMLTypeF32 {
				return nil, nil, fmt.Errorf("turboquant.qjl_matrix must be F32")
			}
			qjl = bytesToFloat32(t.Data, int(t.SizeBytes()/4))
		}
	}

	if rotation == nil || qjl == nil {
		return nil, nil, fmt.Errorf("turboquant matrices not found in GGUF")
	}

	return rotation, qjl, nil
}

func bytesToFloat32(b []byte, n int) []float32 {
	res := make([]float32, n)
	for i := 0; i < n; i++ {
		res[i] = *(*float32)(unsafe.Pointer(&b[i*4]))
	}
	return res
}
