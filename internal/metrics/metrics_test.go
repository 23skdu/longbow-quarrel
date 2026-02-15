package metrics

import (
	"testing"
	"time"
)

func TestMetricsExistence(t *testing.T) {
	// Verify our exported metrics functions exist and don't panic
	RecordInference(10, 100*time.Millisecond)
	RecordGPUMemory(1024 * 1024)
	RecordKernelDuration("MatMul", 5*time.Millisecond)
	// Functions exist and work - no assertion needed
}

func TestRecordInferenceMultiple(t *testing.T) {
	RecordInference(5, 50*time.Millisecond)
	RecordInference(10, 100*time.Millisecond)
	RecordInference(3, 30*time.Millisecond)

	// Counter should accumulate - just verify no panic
}

func TestRecordGPUMemoryChanges(t *testing.T) {
	RecordGPUMemory(1024 * 1024 * 1024) // 1GB
	RecordGPUMemory(512 * 1024 * 1024)  // 512MB - gauge should update
	// Just verify no panic
}

func TestRecordKernelDurationHistogram(t *testing.T) {
	RecordKernelDuration("test_kernel", 10*time.Millisecond)
	RecordKernelDuration("test_kernel", 20*time.Millisecond)
	RecordKernelDuration("test_kernel", 30*time.Millisecond)

	// Histogram should have observations - just verify no panic
}

func TestRecordNumericalInstability(t *testing.T) {
	RecordNumericalInstability("tensor1", 5, 0) // 5 NaNs
	RecordNumericalInstability("tensor2", 0, 3) // 3 Infs

	// Just verify no panic
}

func TestRecordValidationError(t *testing.T) {
	RecordValidationError("decode", "bounds_check")
	RecordValidationError("decode", "dtype_mismatch")
}

func TestRecordContextLength(t *testing.T) {
	RecordContextLength(512)
	RecordContextLength(1024)
	RecordContextLength(2048)
	RecordContextLength(4096)
}

func TestRecordLogitAuditBasic(t *testing.T) {
	RecordLogitAudit(10.0, -5.0, 2.5, 3.0, false, false, false)
}

func TestRecordLogitAuditWithIssues(t *testing.T) {
	RecordLogitAudit(1000.0, -1000.0, 0.0, 500.0, true, true, true)
}

func TestRecordNaNPropagationAuditPatterns(t *testing.T) {
	RecordNaNPropagationAudit(2, 8, 256, "gradual")
	RecordNaNPropagationAudit(0, 4, 128, "sudden")
	RecordNaNPropagationAudit(1, 6, 64, "scattered")
}

func TestRecordRoPEDeviationAudit(t *testing.T) {
	RecordRoPEDeviationAudit(0.001, 1.0, true)
	RecordRoPEDeviationAudit(0.1, 1.5, false)
}

func TestRecordKVCacheSlidingWindow(t *testing.T) {
	RecordKVCacheSlidingWindow(4096, 100, false)
	RecordKVCacheSlidingWindow(4096, 4100, true) // wrapped
}

func TestRecordKVCacheOutOfBounds(t *testing.T) {
	RecordKVCacheOutOfBounds(5000, 4096)
}

func TestRecordKVCacheStats(t *testing.T) {
	RecordKVCacheStats(1024*1024*1024, 256*1024*1024)
}

func TestRecordActivationFlowAudit(t *testing.T) {
	RecordActivationFlowAudit(0, 0, 0) // healthy
	RecordActivationFlowAudit(3, 2, 5) // unhealthy
}

func TestRecordSamplingAudit(t *testing.T) {
	RecordSamplingAudit(map[string]interface{}{
		"temperature":    0.7,
		"top_k":          40,
		"top_p":          0.95,
		"top_token_prob": 0.25,
		"unique_samples": 12,
		"entropy":        2.0,
		"rep_penalty":    1.1,
	})
}

func TestRecordTokenizerMetrics(t *testing.T) {
	RecordTokenizerMetrics(100, 32000, 5*time.Millisecond)
}

func TestRecordActivationPrecision(t *testing.T) {
	RecordActivationPrecision("rmsnorm", 1.5)
	RecordActivationPrecision("swiglu", 50.0)
	RecordActivationPrecision("residual", 2.0)
}

func TestRecordMOELayerLatency(t *testing.T) {
	RecordMOELayerLatency(50 * time.Millisecond)
	RecordMOELayerLatency(100 * time.Millisecond)
}

func TestRecordMOERoutingLatency(t *testing.T) {
	RecordMOERoutingLatency(5 * time.Millisecond)
	RecordMOERoutingLatency(10 * time.Millisecond)
}

func TestRecordMOEExpertSelection(t *testing.T) {
	RecordMOEExpertSelection(2, []int32{0, 3, 5})
	RecordMOEExpertSelection(5, []int32{1, 7})
}

func TestRecordMOEExpertUtilization(t *testing.T) {
	RecordMOEExpertUtilization(0, 0, 0.5)
	RecordMOEExpertUtilization(1, 5, 0.75)
}

func TestTotalTokensAtomic(t *testing.T) {
	// Test atomic operations
	initial := totalTokens.Load()
	RecordInference(1, time.Millisecond)
	after := totalTokens.Load()
	if after != initial+1 {
		t.Errorf("Expected totalTokens to increment by 1, got %d -> %d", initial, after)
	}
}
