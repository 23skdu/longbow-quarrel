//go:build !linux || !amd64 || !cuda || !cgo

package device

import "errors"

// CUDAGraphManager stub for non-CUDA builds.
type CUDAGraphManager struct{}

func NewCUDAGraphManager(ctx *Context) *CUDAGraphManager {
	return &CUDAGraphManager{}
}

func (m *CUDAGraphManager) BeginCapture(batchSize int) error {
	return errors.New("cuda graph: CUDA not available in this build")
}

func (m *CUDAGraphManager) EndCapture(batchSize int) error {
	return errors.New("cuda graph: CUDA not available in this build")
}

func (m *CUDAGraphManager) Launch(batchSize int) error {
	return errors.New("cuda graph: CUDA not available in this build")
}

func (m *CUDAGraphManager) HasGraph(batchSize int) bool {
	return false
}

func (m *CUDAGraphManager) IsCapturing() bool {
	return false
}

func (m *CUDAGraphManager) Close() {}
