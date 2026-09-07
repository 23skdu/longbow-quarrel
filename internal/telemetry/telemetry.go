package telemetry

import (
	"context"
	"errors"
	"os"
	"sync"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/otlp/otlpmetric/otlpmetricgrpc"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/exporters/stdout/stdouttrace"
	"go.opentelemetry.io/otel/sdk/resource"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.17.0"
	"go.opentelemetry.io/otel/trace"
)

var (
	tracer     trace.Tracer
	once       sync.Once
	tracerName = "github.com/23skdu/longbow-quarrel"
)

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

	var tp *sdktrace.TracerProvider
	var mp *sdkmetric.MeterProvider

	if endpoint := os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT"); endpoint != "" {
		traceExporter, err := otlptracegrpc.New(ctx,
			otlptracegrpc.WithEndpoint(endpoint),
			otlptracegrpc.WithInsecure(),
		)
		if err == nil {
			tp = sdktrace.NewTracerProvider(
				sdktrace.WithResource(res),
				sdktrace.WithSampler(sdktrace.AlwaysSample()),
				sdktrace.WithBatcher(traceExporter),
			)
		}

		metricExporter, err := otlpmetricgrpc.New(ctx,
			otlpmetricgrpc.WithEndpoint(endpoint),
			otlpmetricgrpc.WithInsecure(),
		)
		if err == nil {
			mp = sdkmetric.NewMeterProvider(
				sdkmetric.WithResource(res),
				sdkmetric.WithReader(sdkmetric.NewPeriodicReader(metricExporter)),
			)
		}
	} else {
		stdExporter, _ := stdouttrace.New(stdouttrace.WithPrettyPrint())
		tp = sdktrace.NewTracerProvider(
			sdktrace.WithResource(res),
			sdktrace.WithSampler(sdktrace.AlwaysSample()),
			sdktrace.WithBatcher(stdExporter),
		)
		mp = sdkmetric.NewMeterProvider()
	}

	otel.SetTracerProvider(tp)
	if mp != nil {
		otel.SetMeterProvider(mp)
	}

	tracer = tp.Tracer(tracerName)

	return func(ctx context.Context) error {
		var errs []error
		if tp != nil {
			if err := tp.Shutdown(ctx); err != nil {
				errs = append(errs, err)
			}
		}
		if mp != nil {
			if err := mp.Shutdown(ctx); err != nil {
				errs = append(errs, err)
			}
		}
		return errors.Join(errs...)
	}
}

// GetTracer returns the active OpenTelemetry tracer.
func GetTracer() trace.Tracer {
	if tracer == nil {
		once.Do(func() {
			tracer = otel.Tracer(tracerName)
		})
	}
	return tracer
}

// StartSpan starts a new span from the given context.
func StartSpan(ctx context.Context, name string) (context.Context, trace.Span) {
	return GetTracer().Start(ctx, name)
}

// StartRequestSpan starts an inference request span with standard metadata.
func StartRequestSpan(ctx context.Context, requestID, model string) (context.Context, trace.Span) {
	ctx, span := StartSpan(ctx, "llm.request")
	span.SetAttributes(
		semconv.ServiceNameKey.String("longbow-quarrel"),
	)
	span.AddEvent("request_started")
	return ctx, span
}

// RecordTTFT records Time To First Token on the given span.
func RecordTTFT(span trace.Span, d time.Duration) {
	if span != nil {
		span.AddEvent("first_token_emitted")
	}
}

// RecordInterTokenLatency records inter-token generation duration on the span.
func RecordInterTokenLatency(span trace.Span, d time.Duration) {
	if span != nil {
		span.AddEvent("token_generated")
	}
}

// RecordKVPageAlloc records KV page block allocation events on the span.
func RecordKVPageAlloc(span trace.Span, pagesAllocated, freePages int) {
	if span != nil {
		span.AddEvent("kv_page_allocated")
	}
}

// RecordMemoryPressure records high memory events and proactive actions on the span.
func RecordMemoryPressure(span trace.Span, ramPct, vramPct float64, action string) {
	if span != nil {
		span.AddEvent("memory_pressure_event")
	}
}

