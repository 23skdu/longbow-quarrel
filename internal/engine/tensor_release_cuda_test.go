//go:build cuda

package engine

import "github.com/23skdu/longbow-quarrel/internal/device"

// releasedElementCount reports how many elements a tensor still holds after
// Free.
//
// The CUDA backend keeps the weights in device memory, so Free() releases
// the device pointer and Data() becomes nil. The CPU and Metal backends keep
// the weights in Go slices, which is checked in tensor_release_test.go.
func releasedElementCount(t *device.Tensor) int {
	if t == nil {
		return 0
	}
	if t.Data() == nil {
		return 0
	}
	return t.Rows() * t.Cols()
}
