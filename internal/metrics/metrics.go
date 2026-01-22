package metrics

import (
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	InferenceTokensTotal = promauto.NewCounter(prometheus.CounterOpts{
		Name: "inference_tokens_total",
		Help: "The total number of tokens generated",
	})

	InferenceDuration = promauto.NewSummary(prometheus.SummaryOpts{
		Name: "inference_duration_seconds",
		Help: "Duration of inference steps",
	})

	GPUMemoryAllocated = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "gpu_memory_allocated_bytes",
		Help: "Current bytes allocated on GPU",
	})

	KernelDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "gpu_kernel_duration_seconds",
		Help:    "Histogram of kernel execution times",
		Buckets: prometheus.DefBuckets,
	}, []string{"kernel"})

	NumericalInstability = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "numerical_instability_total",
		Help: "Total number of NaN/Inf values detected",
	}, []string{"tensor", "type"})

	ValidationErrors = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "validation_errors_total",
		Help: "Total number of validation errors",
	}, []string{"operation", "error_type"})

	ContextLengthHistogram = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "context_length_tokens",
		Help:    "Distribution of context lengths processed",
		Buckets: []float64{100, 500, 1000, 2000, 4000, 8000, 16000, 32000},
	})
)

func RecordInference(tokens int, duration time.Duration) {
	InferenceTokensTotal.Add(float64(tokens))
	InferenceDuration.Observe(duration.Seconds())
}

func RecordGPUMemory(bytes int64) {
	GPUMemoryAllocated.Set(float64(bytes))
}

func RecordKernelDuration(name string, duration time.Duration) {
	KernelDuration.WithLabelValues(name).Observe(duration.Seconds())
}

func RecordNumericalInstability(name string, nanCount, infCount int) {
	if nanCount > 0 {
		NumericalInstability.WithLabelValues(name, "nan").Add(float64(nanCount))
	}
	if infCount > 0 {
		NumericalInstability.WithLabelValues(name, "inf").Add(float64(infCount))
	}
}

func RecordValidationError(operation, errorType string) {
	ValidationErrors.WithLabelValues(operation, errorType).Inc()
}

func RecordContextLength(tokens int) {
	ContextLengthHistogram.Observe(float64(tokens))
}
