//go:build !linux || !amd64 || !cuda || !cgo

package device

import "testing"

func TestCUDAGraphManager_NoCUDAStubs(t *testing.T) {
	mgr := NewCUDAGraphManager(nil)
	if mgr == nil {
		t.Fatal("expected non-nil manager")
	}
	if err := mgr.BeginCapture(1); err == nil {
		t.Error("expected error from BeginCapture")
	}
	if err := mgr.EndCapture(1); err == nil {
		t.Error("expected error from EndCapture")
	}
	if err := mgr.Launch(1); err == nil {
		t.Error("expected error from Launch")
	}
	if mgr.HasGraph(1) {
		t.Error("expected HasGraph to be false")
	}
	if mgr.IsCapturing() {
		t.Error("expected IsCapturing to be false")
	}
	mgr.Close()
}
