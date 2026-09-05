//go:build linux && cuda && !nccl

package device

import (
	"fmt"
	"unsafe"
)

// ncclCommHandle is a stub when NCCL is not available.
type ncclCommHandle struct{}

func ncclInit(rank, worldSize, deviceID int) (*ncclCommHandle, error) {
	return nil, fmt.Errorf("NCCL not available: build with nccl tag to enable")
}

func (h *ncclCommHandle) ncclAllReduce(sendBuf, recvBuf unsafe.Pointer, count int, stream unsafe.Pointer) error {
	return fmt.Errorf("NCCL not available")
}

func (h *ncclCommHandle) ncclBroadcast(buff unsafe.Pointer, count int, root int, stream unsafe.Pointer) error {
	return fmt.Errorf("NCCL not available")
}

func (h *ncclCommHandle) ncclAllGather(sendBuf, recvBuf unsafe.Pointer, count int, stream unsafe.Pointer) error {
	return fmt.Errorf("NCCL not available")
}

func (h *ncclCommHandle) destroy() {}
