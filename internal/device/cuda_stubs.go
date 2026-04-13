//go:build !linux || !cuda

package device

import (
	"fmt"
	"github.com/23skdu/longbow-quarrel/internal/gguf"
)

type CUDAContext struct {
	device int
}

type CUDAModel struct {
	KCache []*Tensor
}

type Tensor struct{}
type CUDAWeight struct{}
type CUDALayerScratch struct{}

func NewCUDAContext() (*CUDAContext, error) {
	return nil, fmt.Errorf("CUDA not supported on this platform")
}

func CUDAAllocatedBytes() int64 {
	return 0
}

func (c *CUDAContext) Free() {}

func (c *CUDAContext) NewCUDAModel(f *gguf.GGUFFile, kvCache bool, maxSeqLen int) (*CUDAModel, error) {
	return nil, fmt.Errorf("CUDA not supported on this platform")
}

func (m *CUDAModel) Free() {}
