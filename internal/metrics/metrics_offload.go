package metrics

import (
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// ===== Layer Offloading Metrics =====

var (
	GPULayersActive = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "quarrel_gpu_layers_active",
		Help: "Number of transformer layers offloaded to GPU",
	}, []string{"model"})

	CPULayersActive = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "quarrel_cpu_layers_active",
		Help: "Number of transformer layers running on CPU",
	}, []string{"model"})

	LayerOffloadTransfersTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "quarrel_layer_offload_transfers_total",
		Help: "Total number of GPU-to-CPU intermediate activation transfers",
	}, []string{"model"})

	LayerOffloadDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "quarrel_layer_offload_duration_seconds",
		Help:    "Duration of layer offload execution (transfer vs cpu)",
		Buckets: []float64{0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5},
	}, []string{"model", "phase"})
)

// RecordLayerOffload records active layer split between GPU and CPU
func RecordLayerOffload(model string, gpuLayers, cpuLayers int) {
	GPULayersActive.WithLabelValues(model).Set(float64(gpuLayers))
	CPULayersActive.WithLabelValues(model).Set(float64(cpuLayers))
}

// RecordLayerOffloadTransfer records a GPU-host transfer event and duration
func RecordLayerOffloadTransfer(model string, duration time.Duration) {
	LayerOffloadTransfersTotal.WithLabelValues(model).Inc()
	LayerOffloadDuration.WithLabelValues(model, "transfer").Observe(duration.Seconds())
}

// RecordLayerOffloadCPUDuration records the duration of CPU layer computation during offloading
func RecordLayerOffloadCPUDuration(model string, duration time.Duration) {
	LayerOffloadDuration.WithLabelValues(model, "cpu").Observe(duration.Seconds())
}
