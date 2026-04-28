package telemetry

import (
	"context"
	"os"
	"sync"

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
		if tp != nil {
			tp.Shutdown(ctx)
		}
		if mp != nil {
			mp.Shutdown(ctx)
		}
		return nil
	}
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
