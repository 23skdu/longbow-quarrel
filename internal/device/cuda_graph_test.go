//go:build linux && amd64 && cuda && cgo

package device

import (
	"math"
	"testing"
)

func TestCUDA_GraphManager_Lifecycle(t *testing.T) {
	ctx := NewContext()

	gm := NewCUDAGraphManager(ctx)
	defer gm.Close()

	const size = 1024
	aHost := make([]float32, size)
	bHost := make([]float32, size)
	for i := 0; i < size; i++ {
		aHost[i] = float32(i) * 0.5
		bHost[i] = 2.0
	}

	tA, err := ctx.NewTensor(1, size, DataTypeF32)
	if err != nil {
		t.Fatalf("failed to create tensor A: %v", err)
	}
	defer tA.Free()

	tB, err := ctx.NewTensor(1, size, DataTypeF32)
	if err != nil {
		t.Fatalf("failed to create tensor B: %v", err)
	}
	defer tB.Free()

	tOut, err := ctx.NewTensor(1, size, DataTypeF32)
	if err != nil {
		t.Fatalf("failed to create tensor Out: %v", err)
	}
	defer tOut.Free()

	if err := tA.LoadFrom(aHost); err != nil {
		t.Fatalf("failed to load A: %v", err)
	}
	if err := tB.LoadFrom(bHost); err != nil {
		t.Fatalf("failed to load B: %v", err)
	}

	// 1. Capture graph for batch size 1
	if err := gm.BeginCapture(1); err != nil {
		t.Fatalf("BeginCapture failed: %v", err)
	}
	if !gm.IsCapturing() {
		t.Fatal("expected IsCapturing() == true")
	}

	// Enqueue kernel operations into stream during capture
	ctx.Add(tA, tB, tOut, size)


	if err := gm.EndCapture(1); err != nil {
		t.Fatalf("EndCapture failed: %v", err)
	}
	if gm.IsCapturing() {
		t.Fatal("expected IsCapturing() == false after EndCapture")
	}
	if !gm.HasGraph(1) {
		t.Fatal("expected HasGraph(1) == true")
	}

	// 2. Launch graph and verify output
	if err := gm.Launch(1); err != nil {
		t.Fatalf("Launch(1) failed: %v", err)
	}
	ctx.Synchronize()

	outHost := tOut.ToHostF32()
	for i := 0; i < size; i++ {
		expected := aHost[i] + bHost[i]
		if math.Abs(float64(outHost[i]-expected)) > 1e-4 {
			t.Fatalf("mismatch at index %d: got %f, want %f", i, outHost[i], expected)
		}
	}

	// 3. Update input values and launch graph again
	for i := 0; i < size; i++ {
		aHost[i] = float32(i) * 1.5
	}
	if err := tA.LoadFrom(aHost); err != nil {
		t.Fatalf("failed to reload A: %v", err)
	}

	if err := gm.Launch(1); err != nil {
		t.Fatalf("Launch(1) second run failed: %v", err)
	}
	ctx.Synchronize()

	outHost2 := tOut.ToHostF32()
	for i := 0; i < size; i++ {
		expected := aHost[i] + bHost[i]
		if math.Abs(float64(outHost2[i]-expected)) > 1e-4 {
			t.Fatalf("second run mismatch at index %d: got %f, want %f", i, outHost2[i], expected)
		}
	}

	// 4. Close graph manager
	gm.Close()
	if gm.HasGraph(1) {
		t.Fatal("expected HasGraph(1) == false after Close()")
	}
}

func TestCUDA_GraphManager_ErrorHandling(t *testing.T) {
	ctx := NewContext()

	gm := NewCUDAGraphManager(ctx)
	defer gm.Close()

	// Launch uncaptured batch size
	if err := gm.Launch(2); err == nil {
		t.Fatal("expected error launching uncaptured batch size 2")
	}

	// EndCapture without BeginCapture
	if err := gm.EndCapture(1); err == nil {
		t.Fatal("expected error on EndCapture without BeginCapture")
	}

	// Double BeginCapture
	if err := gm.BeginCapture(1); err != nil {
		t.Fatalf("BeginCapture failed: %v", err)
	}
	if err := gm.BeginCapture(2); err == nil {
		t.Fatal("expected error on double BeginCapture")
	}

	// Clean up capture
	_ = gm.EndCapture(1)
}
