//go:build !cuda

package engine

import "github.com/23skdu/longbow-quarrel/internal/device"

// releasedElementCount reports how many elements a tensor still holds after
// Free.
//
// The CPU and Metal backends keep the weights in Go slices, so Free() drops
// the slice and the element count goes to zero. The CUDA backend stores the
// weights in device memory and frees the device pointer instead, which is
// checked in tensor_release_cuda_test.go.
func releasedElementCount(t *device.Tensor) int {
	if t == nil {
		return 0
	}
	return len(t.Data())
}
