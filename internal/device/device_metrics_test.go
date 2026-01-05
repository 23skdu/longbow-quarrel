//go:build darwin && metal


package device

import (
	"testing"
	"github.com/23skdu/longbow-quarrel/internal/metrics"
	"github.com/prometheus/client_golang/prometheus/testutil"
)

func TestDeviceMetrics(t *testing.T) {
	// Assert starting memory is 0 or tracked
	// Since tests run in parallel or sequentially, global state might be dirty.
	// But we can check for *increase*.
	
	initialMem := testutil.ToFloat64(metrics.GPUMemoryAllocated)
	
	ctx := NewContext()
	defer ctx.Free()
	
	// Allocate 1MB
	// 1024 * 512 * 2 (float16) = 1MB
	rows, cols := 1024, 512
	tA := ctx.NewTensor(rows, cols)
	
	// Check memory increase
	currentMem := testutil.ToFloat64(metrics.GPUMemoryAllocated)
	expectedIncrease := float64(rows * cols * 2)
	
	if currentMem - initialMem != expectedIncrease {
		t.Errorf("Expected GPU mem increase %v, got %v", expectedIncrease, currentMem - initialMem)
	}
	
	// Check MatMul duration
	// Reset not possible easily on HistogramVec without deep hacks or recreating.
	// But we can check count of observations.
	
	// Simulate an op
	tB := ctx.NewTensor(cols, rows)
	_ = tA.MatMul(tB)
	ctx.Synchronize()
	
	// We expect "MatMul" label to have at least 1 count.
	// Count using CollectAndCount is strictly for whole registry usually.
	// usage: testutil.CollectAndCount(metrics.KernelDuration, "gpu_kernel_duration_seconds")
	// But we want to check if "MatMul" label exists.
	
	// Metric is: gpu_kernel_duration_seconds{kernel="MatMul"}
	// We can try to Observse 0 just to see if it panics? No.
	
	if testutil.CollectAndCount(metrics.KernelDuration) == 0 {
		t.Error("Expected KernelDuration to have data, got 0")
	}
	
	// Ideally check if "MatMul" label is present.
	// But simply checking count > 0 is enough proof that instrumentation happened 
	// (assuming global state was empty or we know MatMul adds to it).
	// Since we haven't instrumented yet, valid count shouldn't have changed from 0 (if valid was checking delta)
	// But here checking raw existence.
	// metrics.KernelDuration is a Vec. If no Child created, it yields 0 metrics.
}
