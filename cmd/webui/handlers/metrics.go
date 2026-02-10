//go:build webui

package handlers

import (
	"net/http"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
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

func MetricsHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		activeConnections.Inc()
		defer activeConnections.Dec()

		totalRequests.WithLabelValues("default", r.URL.Path).Inc()
	}
}

func HealthHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"status": "healthy"}`))
	}
}

func RecordInference(model string, duration float64, tokens int) {
	inferenceDuration.WithLabelValues(model).Observe(duration)
	totalTokens.Add(float64(tokens))
}

func RecordError(errType string) {
	totalErrors.WithLabelValues(errType).Inc()
}
