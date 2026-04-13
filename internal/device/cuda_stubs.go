//go:build !linux || !cuda

package device

import (
	"fmt"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

func CUDAAllocatedBytes() int64 {
	return 0
}

func (c *Context) Free() {}

func (c *Context) NewCUDAModel(f *gguf.GGUFFile, kvCache bool, maxSeqLen int) (*CUDAModel, error) {
	return nil, fmt.Errorf("CUDA not supported on this platform")
}

func (m *CUDAModel) Free() {}
