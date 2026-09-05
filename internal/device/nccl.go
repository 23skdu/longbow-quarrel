//go:build linux && cuda && nccl

package device

/*
#cgo LDFLAGS: -lnccl
#include <nccl.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

static ncclResult_t ncclInitComm(ncclComm_t* comm, int nranks, int rank, int device) {
    return ncclCommInitRank(comm, nranks, rank, device);
}

static ncclResult_t ncclCommAllReduce(ncclComm_t comm, const void* sendbuff, void* recvbuff,
    size_t count, ncclDataType_t datatype, ncclRedOp_t op, cudaStream_t stream) {
    return ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
}

static ncclResult_t ncclCommBroadcast(ncclComm_t comm, void* buff, size_t count,
    ncclDataType_t datatype, int root, cudaStream_t stream) {
    return ncclBroadcast(buff, buff, count, datatype, root, comm, stream);
}

static ncclResult_t ncclCommAllGather(ncclComm_t comm, const void* sendbuff, void* recvbuff,
    size_t count, ncclDataType_t datatype, cudaStream_t stream) {
    return ncclAllGather(sendbuff, recvbuff, count, datatype, comm, stream);
}

static void ncclCommDestroyWrapper(ncclComm_t comm) {
    ncclCommDestroy(comm);
}

static const char* ncclGetLastErrorString() {
    return ncclGetErrorString(ncclSystemError);
}
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// ncclCommHandle wraps an NCCL communicator for use across GPUs.
type ncclCommHandle struct {
	comm C.ncclComm_t
}

// ncclInit initializes an NCCL communicator for the given rank.
func ncclInit(rank, worldSize, deviceID int) (*ncclCommHandle, error) {
	var comm C.ncclComm_t
	ret := C.ncclInitComm(&comm, C.int(worldSize), C.int(rank), C.int(deviceID))
	if ret != C.ncclSuccess {
		return nil, fmt.Errorf("ncclCommInitRank failed for rank %d: %d", rank, int(ret))
	}
	return &ncclCommHandle{comm: comm}, nil
}

// ncclAllReduce performs an all-reduce operation across all ranks.
func (h *ncclCommHandle) ncclAllReduce(sendBuf, recvBuf unsafe.Pointer, count int, stream unsafe.Pointer) error {
	ret := C.ncclCommAllReduce(h.comm, sendBuf, recvBuf,
		C.size_t(count), C.ncclFloat, C.ncclSum, (C.cudaStream_t)(stream))
	if ret != C.ncclSuccess {
		return fmt.Errorf("ncclAllReduce failed: %d", int(ret))
	}
	return nil
}

// ncclBroadcast broadcasts data from root to all ranks.
func (h *ncclCommHandle) ncclBroadcast(buff unsafe.Pointer, count int, root int, stream unsafe.Pointer) error {
	ret := C.ncclCommBroadcast(h.comm, buff,
		C.size_t(count), C.ncclFloat, C.int(root), (C.cudaStream_t)(stream))
	if ret != C.ncclSuccess {
		return fmt.Errorf("ncclBroadcast failed: %d", int(ret))
	}
	return nil
}

// ncclAllGather gathers data from all ranks.
func (h *ncclCommHandle) ncclAllGather(sendBuf, recvBuf unsafe.Pointer, count int, stream unsafe.Pointer) error {
	ret := C.ncclCommAllGather(h.comm, sendBuf, recvBuf,
		C.size_t(count), C.ncclFloat, (C.cudaStream_t)(stream))
	if ret != C.ncclSuccess {
		return fmt.Errorf("ncclAllGather failed: %d", int(ret))
	}
	return nil
}

// destroy releases the NCCL communicator.
func (h *ncclCommHandle) destroy() {
	if h.comm != nil {
		C.ncclCommDestroyWrapper(h.comm)
		h.comm = nil
	}
}
