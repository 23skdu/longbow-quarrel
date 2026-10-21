//go:build linux

package vector

import (
	"syscall"
	"unsafe"
)

// mmapPopulate is Linux MAP_POPULATE: the mapping is pre-faulted at
// mmap time, so the first query touches no soft-fault storm (P0-05).
const mmapPopulate = 0x08000

// msyncSlice flushes one mapped chunk to the backing file. Go 1.27
// dropped the syscall.Msync/Madvise wrappers, so both shims go through
// the raw entry points.
func msyncSlice(b []byte) error {
	if len(b) == 0 {
		return nil
	}
	_, _, errno := syscall.Syscall(syscall.SYS_MSYNC,
		uintptr(unsafe.Pointer(&b[0])), uintptr(len(b)), uintptr(syscall.MS_SYNC))
	if errno != 0 {
		return errno
	}
	return nil
}

// madviseSlice issues the madvise hint for one chunk.
func madviseSlice(b []byte, advice int) error {
	if len(b) == 0 {
		return nil
	}
	_, _, errno := syscall.Syscall(syscall.SYS_MADVISE,
		uintptr(unsafe.Pointer(&b[0])), uintptr(len(b)), uintptr(advice))
	if errno != 0 {
		return errno
	}
	return nil
}
