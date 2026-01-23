package metrics

import (
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
)

func TestMetricsExistence(t *testing.T) {
	// Reset registry or assume fresh state
	// Verify our exported metrics exist

	// We expect these functions/variables to exist in package metrics:
	// RecordInference(tokens int, duration time.Duration)
	// RecordGPUMemory(bytes int64)
	// RecordKernelDuration(name string, duration time.Duration)

	// This test will fail to compile initially because these functions don't exist.
	// That is the "Red" in TDD.

	RecordInference(10, 100*time.Millisecond)
	RecordGPUMemory(1024 * 1024)
	RecordKernelDuration("MatMul", 5*time.Millisecond)

	// Verify values using testutil (basic check)
	// tokens counter should be 10
	if val := testutil.ToFloat64(InferenceTokensTotal); val != 10 {
		t.Errorf("Expected 10 tokens, got %v", val)
	}

	// GPU memory should be 1MB
	if val := testutil.ToFloat64(GPUMemoryAllocated); val != 1048576 {
		t.Errorf("Expected 1MB GPU mem, got %v", val)
	}
}
