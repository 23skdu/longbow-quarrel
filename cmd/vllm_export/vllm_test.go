//go:build linux && cuda

package vllm

import (
	"testing"
)

func TestInit(t *testing.T) {
	if !IsInitialized() {
		if err := Init(0); err != nil {
			t.Errorf("Init() failed: %v", err)
		}
	}
}

func TestGetDeviceCount(t *testing.T) {
	count, err := GetDeviceCount()
	if err != nil {
		t.Skipf("Skipping: %v", err)
	}
	if count < 1 {
		t.Errorf("Expected at least 1 device, got %d", count)
	}
	t.Logf("Detected %d CUDA device(s)", count)
}

func TestGetDeviceName(t *testing.T) {
	count, err := GetDeviceCount()
	if err != nil || count < 1 {
		t.Skip("No CUDA devices available")
	}

	name := GetDeviceName(0)
	if name == "" {
		t.Error("GetDeviceName() returned empty string")
	}
	t.Logf("Device 0 name: %s", name)
}

func TestMemoryInfo(t *testing.T) {
	if !IsInitialized() {
		if err := Init(0); err != nil {
			t.Skipf("Skipping: CUDA not available: %v", err)
		}
	}

	free, total, err := GetMemoryInfo()
	if err != nil {
		t.Skipf("Skipping: %v", err)
	}
	if total == 0 {
		t.Error("Expected non-zero total memory")
	}
	t.Logf("GPU Memory: %d free / %d total bytes", free, total)
}
