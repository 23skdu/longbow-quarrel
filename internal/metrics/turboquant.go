package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// TurboQuantCompressionRatio tracks the effectively achieved compression ratio for KV caches using TurboQuant.
	TurboQuantCompressionRatio = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "longbow_turboquant_compression_ratio",
			Help: "Current compression ratio achieved by TurboQuant.",
		},
		[]string{"layer"},
	)

	// TurboQuantLatency counts the latency distribution for TurboQuant quantization (PolarQuant + QJL).
	TurboQuantLatency = promauto.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "longbow_turboquant_quantization_latency_seconds",
			Help:    "Latency of TurboQuant operations in seconds.",
			Buckets: prometheus.DefBuckets,
		},
	)
)

func RecordTurboQuantCompression(layer string, ratio float64) {
	TurboQuantCompressionRatio.WithLabelValues(layer).Set(ratio)
}

func RecordTurboQuantLatency(duration float64) {
	TurboQuantLatency.Observe(duration)
}

// RecordTurboQuantBatch records metrics for a batch of operations
func RecordTurboQuantBatch(compressionRatio float64, totalLatency float64) {
	TurboQuantCompressionRatio.WithLabelValues("batch").Set(compressionRatio)
	TurboQuantLatency.Observe(totalLatency)
}
