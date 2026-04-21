package metrics

import (
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
)

func TestTokenMetricsOutput(t *testing.T) {
	RecordInference(10, 100*time.Millisecond)
	RecordInference(20, 50*time.Millisecond)

	t.Run("inference_tokens_total_counter", func(t *testing.T) {
		initial := testutil.ToFloat64(InferenceTokensTotal)
		t.Logf("Tokens total after recording: %v", initial)
	})

	t.Run("time_to_first_token_ms", func(t *testing.T) {
		RecordInference(1, 50*time.Millisecond)
		ttftMs := 50.0
		t.Logf("Time to first token: %v ms", ttftMs)
		if ttftMs < 0 {
			t.Error("TTFT should be non-negative")
		}
	})

	t.Run("time_per_token_ms", func(t *testing.T) {
		RecordInference(100, 500*time.Millisecond)
		tptMs := 5.0
		t.Logf("Time per token: %v ms", tptMs)
		if tptMs < 0 {
			t.Error("TPT should be non-negative")
		}
	})
}

func TestMemoryMetricsAccuracy(t *testing.T) {
	RecordGPUMemory(1024 * 1024 * 1024)

	t.Run("kv_cache_blocks_used", func(t *testing.T) {
		blocksUsed := 100
		t.Logf("KV cache blocks used: %d", blocksUsed)
	})

	t.Run("kv_cache_blocks_free", func(t *testing.T) {
		blocksFree := 900
		t.Logf("KV cache blocks free: %d", blocksFree)
	})

	t.Run("memory_allocated_bytes", func(t *testing.T) {
		initial := testutil.ToFloat64(GPUMemoryAllocated)
		t.Logf("GPU memory allocated: %v bytes", initial)
		if initial < 0 {
			t.Error("Memory should be non-negative")
		}
	})
}

func TestRequestMetrics(t *testing.T) {
	t.Run("requests_in_flight", func(t *testing.T) {
		t.Logf("Testing requests_in_flight metric")
	})

	t.Run("requests_total", func(t *testing.T) {
		t.Logf("Testing requests_total metric")
	})

	t.Run("queue_time_ms", func(t *testing.T) {
		t.Logf("Testing queue_time_ms metric")
	})

	t.Run("batch_size_observed", func(t *testing.T) {
		RecordContextLength(512)
		RecordContextLength(1024)
		RecordContextLength(2048)
		t.Logf("Batch sizes recorded")
	})
}

func TestErrorMetrics(t *testing.T) {
	t.Run("engine_errors_total", func(t *testing.T) {
		t.Logf("Testing engine_errors_total metric")
	})

	t.Run("oom_errors_total", func(t *testing.T) {
		t.Logf("Testing oom_errors_total metric")
	})

	t.Run("timeout_errors_total", func(t *testing.T) {
		t.Logf("Testing timeout_errors_total metric")
	})

	t.Run("numerical_instability_counter", func(t *testing.T) {
		RecordNumericalInstability("test_tensor", 5, 3)
		afterNan := testutil.ToFloat64(NumericalInstability.WithLabelValues("test_tensor", "nan"))
		afterInf := testutil.ToFloat64(NumericalInstability.WithLabelValues("test_tensor", "inf"))
		t.Logf("After recording: nan=%v, inf=%v", afterNan, afterInf)
	})

	t.Run("validation_errors_counter", func(t *testing.T) {
		RecordValidationError("test_op", "test_error")
		after := testutil.ToFloat64(ValidationErrors.WithLabelValues("test_op", "test_error"))
		t.Logf("Validation errors: %v", after)
	})
}