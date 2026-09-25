package metrics

import (
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// =============================================================================
// SIMD Metrics - Phase 9
// =============================================================================

var (
	SIMDLevelDetected = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "simd_level_detected",
		Help: "SIMD level detected: 0=scalar, 1=AVX2, 2=AVX-512",
	})

	SIMDKernelDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "simd_kernel_duration_seconds",
		Help:    "Duration of SIMD kernel operations",
		Buckets: []float64{0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01},
	}, []string{"kernel", "size_class"})

	SIMDSoftmaxDuration = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "simd_softmax_duration_seconds",
		Help:    "Duration of softmax SIMD operation",
		Buckets: []float64{0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001},
	})

	SIMDRMSNormDuration = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "simd_rmsnorm_duration_seconds",
		Help:    "Duration of RMSNorm SIMD operation",
		Buckets: []float64{0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005},
	})

	SIMDMatmulDuration = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "simd_matmul_duration_seconds",
		Help:    "Duration of matmul SIMD operation",
		Buckets: []float64{0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0},
	})

	SIMDAttentionDuration = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "simd_attention_duration_seconds",
		Help:    "Duration of attention SIMD operation",
		Buckets: []float64{0.0001, 0.001, 0.01, 0.1, 1.0},
	})

	SIMDFusedMLPDuration = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "simd_fused_mlp_duration_seconds",
		Help:    "Duration of fused MLP SIMD operation",
		Buckets: []float64{0.0001, 0.001, 0.01, 0.1, 1.0},
	})

	SIMDRoPEDuration = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "simd_rope_duration_seconds",
		Help:    "Duration of RoPE SIMD operation",
		Buckets: []float64{0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005},
	})

	SIMDSwiGLUDuration = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "simd_swiglu_duration_seconds",
		Help:    "Duration of SwiGLU SIMD operation",
		Buckets: []float64{0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005},
	})

	SIMDFallbackCount = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "simd_fallback_count_total",
		Help: "Number of times SIMD fell back to lower implementation",
	}, []string{"from", "to"})

	SIMDNaNCount = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "simd_nan_count_total",
		Help: "Number of NaN values produced by SIMD operations",
	}, []string{"kernel"})

	SIMDInfCount = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "simd_inf_count_total",
		Help: "Number of Inf values produced by SIMD operations",
	}, []string{"kernel"})

	SIMDKernelErrors = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "simd_kernel_errors_total",
		Help: "Number of errors in SIMD kernel operations",
	}, []string{"kernel", "error_type"})

	SIMDDispatchTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "simd_dispatch_total",
		Help: "Total number of SIMD kernel dispatches by implementation",
	}, []string{"kernel"})
)

// RecordSIMDDispatch records a SIMD kernel dispatch
func RecordSIMDDispatch(kernel string) {
	SIMDDispatchTotal.WithLabelValues(kernel).Inc()
}

// RecordSIMDLevel records the detected SIMD level
func RecordSIMDLevel(level int) {
	SIMDLevelDetected.Set(float64(level))
}

// RecordSIMDKernelDuration records the duration of a SIMD kernel operation
func RecordSIMDKernelDuration(kernel string, duration time.Duration, sizeClass string) {
	SIMDKernelDuration.WithLabelValues(kernel, sizeClass).Observe(duration.Seconds())
}

// RecordSIMDSoftmaxDuration records softmax duration
func RecordSIMDSoftmaxDuration(duration time.Duration) {
	SIMDSoftmaxDuration.Observe(duration.Seconds())
}

// RecordSIMDRMSNormDuration records RMSNorm duration
func RecordSIMDRMSNormDuration(duration time.Duration) {
	SIMDRMSNormDuration.Observe(duration.Seconds())
}

// RecordSIMDMatmulDuration records matmul duration
func RecordSIMDMatmulDuration(duration time.Duration) {
	SIMDMatmulDuration.Observe(duration.Seconds())
}

// RecordSIMDAttentionDuration records attention duration
func RecordSIMDAttentionDuration(duration time.Duration) {
	SIMDAttentionDuration.Observe(duration.Seconds())
}

// RecordSIMDFusedMLPDuration records fused MLP duration
func RecordSIMDFusedMLPDuration(duration time.Duration) {
	SIMDFusedMLPDuration.Observe(duration.Seconds())
}

// RecordSIMDRoPEDuration records RoPE duration
func RecordSIMDRoPEDuration(duration time.Duration) {
	SIMDRoPEDuration.Observe(duration.Seconds())
}

// RecordSIMDSwiGLUDuration records SwiGLU duration
func RecordSIMDSwiGLUDuration(duration time.Duration) {
	SIMDSwiGLUDuration.Observe(duration.Seconds())
}

// RecordSIMDFallback records a SIMD fallback event
func RecordSIMDFallback(from, to string) {
	SIMDFallbackCount.WithLabelValues(from, to).Inc()
}

// RecordSIMDNaN records NaN in SIMD output
func RecordSIMDNaN(kernel string) {
	SIMDNaNCount.WithLabelValues(kernel).Inc()
}

// RecordSIMDInf records Inf in SIMD output
func RecordSIMDInf(kernel string) {
	SIMDInfCount.WithLabelValues(kernel).Inc()
}

// RecordSIMDKernelError records SIMD kernel error
func RecordSIMDKernelError(kernel, errorType string) {
	SIMDKernelErrors.WithLabelValues(kernel, errorType).Inc()
}
