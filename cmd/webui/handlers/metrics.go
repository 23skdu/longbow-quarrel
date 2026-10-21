//go:build webui

package handlers

import (
	"net/http"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
	activeConnections = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "quarrel_webui_connections_active",
		Help: "Number of active WebSocket connections",
	})

	totalRequests = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "quarrel_webui_requests_total",
		Help: "Total number of requests",
	}, []string{"model", "endpoint"})

	inferenceDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "quarrel_webui_inference_duration_seconds",
		Help:    "Inference request duration",
		Buckets: prometheus.ExponentialBuckets(0.01, 2, 10),
	}, []string{"model"})

	totalTokens = promauto.NewCounter(prometheus.CounterOpts{
		Name: "quarrel_webui_tokens_total",
		Help: "Total tokens generated",
	})

	totalErrors = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "quarrel_webui_errors_total",
		Help: "Total errors by type",
	}, []string{"type"})
)

// MetricsHandler counts the scrape and serves the metrics.
//
// It used to only count: the /metrics route served the raw Prometheus
// handler, so the WebUI's own counters never had a child and were not
// even exported.
func MetricsHandler() http.HandlerFunc {
	promHandler := promhttp.Handler()

	return func(w http.ResponseWriter, r *http.Request) {
		activeConnections.Inc()
		defer activeConnections.Dec()

		totalRequests.WithLabelValues("default", r.URL.Path).Inc()

		promHandler.ServeHTTP(w, r)
	}
}

func RecordInference(model string, duration float64, tokens int) {
	inferenceDuration.WithLabelValues(model).Observe(duration)
	totalTokens.Add(float64(tokens))
}

func RecordError(errType string) {
	totalErrors.WithLabelValues(errType).Inc()
}
