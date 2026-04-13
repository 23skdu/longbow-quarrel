//go:build linux && cuda

package device

import "C"

// Export package provides C-compatible exports for vLLM integration

// InitCUDA initializes the CUDA context for external use
func InitCUDA() (*Context, error) {
	return NewContext()
}

// GetCUDAContext returns the global CUDA context
func GetCUDAContext() *Context {
	return globalContext
}

// ExportDeviceCount returns the number of CUDA devices (exported)
func ExportDeviceCount() (int, error) {
	return GetDeviceCount()
}

// ExportDeviceName returns the device name (exported)
func ExportDeviceName(device int) string {
	return GetDeviceName(device)
}

// ExportDeviceMemory returns device memory info (exported)
func ExportDeviceMemory(device int) (int64, error) {
	return GetDeviceMemory(device)
}
