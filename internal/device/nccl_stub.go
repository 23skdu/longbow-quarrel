//go:build linux && cuda && !nccl

package device

import (
	"fmt"
	"unsafe"
)

// ncclCommHandle is a stub when NCCL is not available.
type ncclCommHandle struct{}

func ncclInit(_, _, _ int) (*ncclCommHandle, error) {
	return nil, fmt.Errorf("NCCL not available: build with nccl tag to enable")
}

func (h *ncclCommHandle) ncclAllReduce(_, _ unsafe.Pointer, _ int, _ unsafe.Pointer) error {
	return fmt.Errorf("NCCL not available")
}

func (h *ncclCommHandle) ncclBroadcast(_ unsafe.Pointer, _ int, _ int, _ unsafe.Pointer) error {
	return fmt.Errorf("NCCL not available")
}

func (h *ncclCommHandle) ncclAllGather(_, _ unsafe.Pointer, _ int, _ unsafe.Pointer) error {
	return fmt.Errorf("NCCL not available")
}

func (h *ncclCommHandle) destroy() {}
