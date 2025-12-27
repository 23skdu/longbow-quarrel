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
		Name: "gpu_kernel_duration_seconds",
		Help: "Histogram of kernel execution times",
		Buckets: prometheus.DefBuckets,
	}, []string{"kernel"})
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
