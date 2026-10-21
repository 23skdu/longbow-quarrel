//go:build darwin

package vector

import (
	"syscall"
	"unsafe"
)

// mmapPopulate has no MAP_POPULATE on darwin; openMMap compensates
// with a MADV_WILLNEED pre-fault hint after the map is created.
const mmapPopulate = 0

// msyncSlice flushes one mapped chunk to the backing file. darwin has
// the msync system call but no syscall.Msync wrapper, so the shim goes
// through the raw entry point.
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

// madviseSlice issues the madvise hint through the raw entry point
// (darwin ships the call but no Go 1.27 wrapper).
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
