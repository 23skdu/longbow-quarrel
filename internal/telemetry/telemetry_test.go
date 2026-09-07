package telemetry

import (
	"context"
	"os"
	"testing"
	"time"
)

func TestTelemetry_InitTracer_Stdout(t *testing.T) {
	shutdown := InitTracer()
	if shutdown == nil {
		t.Fatal("expected non-nil shutdown function")
	}
	defer func() {
		_ = shutdown(context.Background())
	}()

	tracer := GetTracer()
	if tracer == nil {
		t.Fatal("expected non-nil tracer")
	}

	ctx := context.Background()
	ctx, span := StartSpan(ctx, "test.operation")
	if span == nil {
		t.Fatal("expected non-nil span")
	}
	span.End()
}

func TestTelemetry_InitTracer_OTLP(t *testing.T) {
	_ = os.Setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "127.0.0.1:4317")
	defer func() {
		_ = os.Unsetenv("OTEL_EXPORTER_OTLP_ENDPOINT")
	}()

	shutdown := InitTracer()
	if shutdown == nil {
		t.Fatal("expected non-nil shutdown function")
	}
	ctxTimeout, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	defer func() {
		_ = shutdown(ctxTimeout)
	}()

	ctx := context.Background()
	ctx, span := StartRequestSpan(ctx, "req-123", "qwen3.5-0.8b")
	if span == nil {
		t.Fatal("expected non-nil request span")
	}

	RecordTTFT(span, 15*time.Millisecond)
	RecordInterTokenLatency(span, 5*time.Millisecond)
	RecordKVPageAlloc(span, 4, 128)
	RecordMemoryPressure(span, 0.88, 0.91, "defrag")

	span.End()
}

func TestTelemetry_NilSafety(t *testing.T) {
	// Test fallback initialization when tracer is nil
	tracer = nil
	tr := GetTracer()
	if tr == nil {
		t.Fatal("expected tracer to be created on demand")
	}

	// Should not panic on nil span
	RecordTTFT(nil, 10*time.Millisecond)
	RecordInterTokenLatency(nil, 10*time.Millisecond)
	RecordKVPageAlloc(nil, 1, 10)
	RecordMemoryPressure(nil, 0.5, 0.5, "none")
}
