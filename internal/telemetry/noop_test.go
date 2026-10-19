package telemetry

import (
	"context"
	"os"
	"testing"

	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

// TestInitTracer_NoExporterIsANoop checks the default configuration.
//
// InitTracer was never called by any binary, and its default branch set up a
// stdout exporter with AlwaysSample -- which would have printed every span
// of every request to stdout. That is unusable for a serving binary, so the
// default is now a no-op tracer and the console exporter is opt-in.
func TestInitTracer_NoExporterIsANoop(t *testing.T) {
	// Neither variable must be set for this test to be meaningful.
	if err := os.Unsetenv("OTEL_EXPORTER_OTLP_ENDPOINT"); err != nil {
		t.Fatalf("failed to unset %s: %v", "OTEL_EXPORTER_OTLP_ENDPOINT", err)
	}
	if err := os.Unsetenv("OTEL_TRACES_EXPORTER"); err != nil {
		t.Fatalf("failed to unset %s: %v", "OTEL_TRACES_EXPORTER", err)
	}

	shutdown := InitTracer()
	if shutdown == nil {
		t.Fatal("InitTracer returned a nil shutdown function")
	}
	defer func() { _ = shutdown(context.Background()) }()

	// Without an exporter the global tracer provider must not be an SDK
	// provider, i.e. it must be the no-op one.
	if _, ok := otel.GetTracerProvider().(*sdktrace.TracerProvider); ok {
		t.Error("the global tracer provider is an SDK provider; without a " +
			"configured exporter the spans would be exported somewhere unexpected")
	}

	// A span must still be creatable and usable.
	_, span := StartSpan(context.Background(), "test")
	span.SetAttributes()
	span.End()
}
