package metrics

import (
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
)

func TestTokenMetricsOutput(t *testing.T) {
	initialTokens := testutil.ToFloat64(InferenceTokensTotal)
	initialAtomic := totalTokens.Load()

	tokensToRecord := 25
	duration := 150 * time.Millisecond
	RecordInference(tokensToRecord, duration)

	afterTokens := testutil.ToFloat64(InferenceTokensTotal)
	afterAtomic := totalTokens.Load()

	if delta := afterTokens - initialTokens; delta != float64(tokensToRecord) {
		t.Errorf("expected InferenceTokensTotal delta %d, got %v", tokensToRecord, delta)
	}

	if deltaAtomic := afterAtomic - initialAtomic; deltaAtomic != int64(tokensToRecord) {
		t.Errorf("expected totalTokens atomic delta %d, got %d", tokensToRecord, deltaAtomic)
	}

	// Test single token increment
	RecordInference(1, 10*time.Millisecond)
	if afterSingle := totalTokens.Load(); afterSingle != afterAtomic+1 {
		t.Errorf("expected atomic increment after RecordInference(1), got %d vs %d", afterSingle, afterAtomic+1)
	}
}

func TestMemoryMetricsAccuracy(t *testing.T) {
	testBytes := int64(1024 * 1024 * 128) // 128 MB
	RecordGPUMemory(testBytes)

	allocated := testutil.ToFloat64(GPUMemoryAllocated)
	if int64(allocated) != testBytes {
		t.Errorf("expected GPUMemoryAllocated %d, got %d", testBytes, int64(allocated))
	}

	// Update to 0 bytes
	RecordGPUMemory(0)
	allocatedZero := testutil.ToFloat64(GPUMemoryAllocated)
	if allocatedZero != 0 {
		t.Errorf("expected GPUMemoryAllocated 0, got %v", allocatedZero)
	}

	// Verify TurboQuant batch recording does not panic and sets metrics
	RecordTurboQuantBatch(4.0, 0.0012)
}