//go:build linux && amd64 && cuda && cgo

package device

import "testing"

// TestNewContextOnDevice checks that a context can be created on a chosen
// CUDA device.
//
// The -devices flag used to be ignored: the engine always created its
// context on the current device, and Context.DeviceID() was hardcoded to 0
// so nothing could report the device the engine was actually running on.
func TestNewContextOnDevice(t *testing.T) {
	count, err := GetDeviceCount()
	if err != nil {
		t.Skipf("no CUDA device available: %v", err)
	}
	if count < 1 {
		t.Skip("no CUDA devices on this host")
	}

	ctx, err := NewContextOnDevice(0)
	if err != nil {
		t.Fatalf("failed to create a context on device 0: %v", err)
	}
	defer ctx.Free()

	if got := ctx.DeviceID(); got != 0 {
		t.Errorf("the context reports device %d, want 0", got)
	}

	// A device that does not exist has to be an error rather than a
	// silent fallback to another device.
	if _, err := NewContextOnDevice(count + 10); err == nil {
		t.Errorf("expected an error for a device that does not exist (the host has %d)", count)
	}
}
