package telemetry

import (
	"context"
	"sync"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/trace"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.17.0"
)

var (
	tracer     trace.Tracer
	once       sync.Once
	tracerName = "github.com/23skdu/longbow-quarrel"
)

// InitTracer initializes the global tracer provider.
func InitTracer() func(context.Context) error {
	ctx := context.Background()

	res, err := resource.New(ctx,
		resource.WithAttributes(
			semconv.ServiceNameKey.String("longbow-quarrel"),
		),
	)
	if err != nil {
		return nil
	}

	// For now, we use a simple processor that exports to a mock or stdout if needed.
	// In production, this would use an OTLP exporter.
	tp := sdktrace.NewTracerProvider(
		sdktrace.WithResource(res),
		sdktrace.WithSampler(sdktrace.AlwaysSample()),
	)
	otel.SetTracerProvider(tp)
	
	tracer = tp.Tracer(tracerName)

	return tp.Shutdown
}

// StartSpan starts a new span from the given context.
func StartSpan(ctx context.Context, name string) (context.Context, trace.Span) {
	if tracer == nil {
		once.Do(func() {
			tracer = otel.Tracer(tracerName)
		})
	}
	return tracer.Start(ctx, name)
}
