//go:build amd64 && cgo

package simd

import (
	"math"
	"testing"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/metrics"
	"github.com/23skdu/longbow-quarrel/internal/promtest"
)

func TestClassifySize(t *testing.T) {
	tests := []struct {
		n    int
		want string
	}{
		{0, "tiny"},
		{1, "tiny"},
		{63, "tiny"},
		{64, "small"},
		{100, "small"},
		{255, "small"},
		{256, "medium"},
		{500, "medium"},
		{1023, "medium"},
		{1024, "large"},
		{2000, "large"},
		{4095, "large"},
		{4096, "xlarge"},
		{8000, "xlarge"},
		{16383, "xlarge"},
		{16384, "xxlarge"},
		{100000, "xxlarge"},
	}

	for _, tt := range tests {
		if got := classifySize(tt.n); got != tt.want {
			t.Errorf("classifySize(%d) = %q, want %q", tt.n, got, tt.want)
		}
	}
}

// kernelHistograms maps each kernel name to the per-kernel latency histogram
// that metricsWrapper.record() should feed.
func kernelHistograms() map[string]any {
	return map[string]any{
		"Softmax":   metrics.SIMDSoftmaxDuration,
		"RMSNorm":   metrics.SIMDRMSNormDuration,
		"Matmul":    metrics.SIMDMatmulDuration,
		"Attention": metrics.SIMDAttentionDuration,
		"FusedMLP":  metrics.SIMDFusedMLPDuration,
		"RoPE":      metrics.SIMDRoPEDuration,
		"SwiGLU":    metrics.SIMDSwiGLUDuration,
	}
}

func TestMetricsRecordCalls(t *testing.T) {
	metricsEnabled = true
	defer func() { metricsEnabled = true }()

	dispatch := metrics.SIMDKernelDuration.WithLabelValues("Softmax", "medium")
	softmax := kernelHistograms()["Softmax"]
	dispatchBefore := promtest.Value(t, "simd_kernel_duration_seconds", dispatch)
	softmaxBefore := promtest.Value(t, "simd_softmax_duration_seconds", softmax)

	m := &metricsWrapper{
		startTime: time.Now(),
		kernel:    "Softmax",
		size:      256,
	}
	m.record()

	promtest.RequireDelta(t, "simd_kernel_duration_seconds", dispatch, dispatchBefore, 1)
	promtest.RequireDelta(t, "simd_softmax_duration_seconds", softmax, softmaxBefore, 1)
}

func TestMetricsRecordAllKernels(t *testing.T) {
	metricsEnabled = true
	defer func() { metricsEnabled = true }()

	kernels := []string{"Softmax", "RMSNorm", "Matmul", "Attention", "FusedMLP", "RoPE", "SwiGLU", "Unknown"}
	perKernel := make([]float64, len(kernels))
	for i, k := range kernels {
		perKernel[i] = promtest.Value(t, "simd_kernel_duration_seconds",
			metrics.SIMDKernelDuration.WithLabelValues(k, "medium"))
	}
	specificBefore := make([]float64, 0, len(kernels))
	specific := make([]string, 0, len(kernels))
	for _, k := range kernels {
		if h, ok := kernelHistograms()[k]; ok {
			specific = append(specific, k)
			specificBefore = append(specificBefore, promtest.Value(t, "simd_kernel_latency", h))
		}
	}

	for _, k := range kernels {
		m := &metricsWrapper{
			startTime: time.Now(),
			kernel:    k,
			size:      1000,
		}
		m.record()
	}

	for i, k := range kernels {
		promtest.RequireDelta(t, "simd_kernel_duration_seconds["+k+"]",
			metrics.SIMDKernelDuration.WithLabelValues(k, "medium"), perKernel[i], 1)
	}
	for i, k := range specific {
		promtest.RequireDelta(t, "simd_kernel_latency["+k+"]", kernelHistograms()[k], specificBefore[i], 1)
	}
}

func TestMetricsDisabled(t *testing.T) {
	metricsEnabled = false
	defer func() { metricsEnabled = true }()

	dispatch := metrics.SIMDKernelDuration.WithLabelValues("Softmax", "medium")
	nan := metrics.SIMDNaNCount.WithLabelValues("Softmax")
	dispatchBefore := promtest.Value(t, "simd_kernel_duration_seconds", dispatch)
	nanBefore := promtest.Value(t, "simd_nan_count_total", nan)

	m := &metricsWrapper{
		startTime: time.Now(),
		kernel:    "Softmax",
		size:      256,
	}
	m.record()

	checkOutputNaNInf([]float32{1, float32(math.NaN()), 3}, "Softmax")

	promtest.RequireDelta(t, "simd_kernel_duration_seconds(disabled)", dispatch, dispatchBefore, 0)
	promtest.RequireDelta(t, "simd_nan_count_total(disabled)", nan, nanBefore, 0)
}

func TestCheckOutputNaN(t *testing.T) {
	metricsEnabled = true
	defer func() { metricsEnabled = true }()

	nan := metrics.SIMDNaNCount.WithLabelValues("Softmax")
	inf := metrics.SIMDInfCount.WithLabelValues("Softmax")
	errs := metrics.SIMDKernelErrors.WithLabelValues("Softmax", "output_invalid")
	nanBefore := promtest.Value(t, "simd_nan_count_total", nan)
	infBefore := promtest.Value(t, "simd_inf_count_total", inf)
	errsBefore := promtest.Value(t, "simd_kernel_errors_total", errs)

	data := []float32{1.0, float32(math.NaN()), 3.0}
	checkOutputNaNInf(data, "Softmax")

	promtest.RequireDelta(t, "simd_nan_count_total", nan, nanBefore, 1)
	promtest.RequireDelta(t, "simd_inf_count_total(no NaN-only input)", inf, infBefore, 0)
	promtest.RequireDelta(t, "simd_kernel_errors_total", errs, errsBefore, 1)
}

func TestCheckOutputInf(t *testing.T) {
	metricsEnabled = true
	defer func() { metricsEnabled = true }()

	nan := metrics.SIMDNaNCount.WithLabelValues("RMSNorm")
	inf := metrics.SIMDInfCount.WithLabelValues("RMSNorm")
	errs := metrics.SIMDKernelErrors.WithLabelValues("RMSNorm", "output_invalid")
	nanBefore := promtest.Value(t, "simd_nan_count_total", nan)
	infBefore := promtest.Value(t, "simd_inf_count_total", inf)
	errsBefore := promtest.Value(t, "simd_kernel_errors_total", errs)

	data := []float32{1.0, float32(math.Inf(1)), 3.0}
	checkOutputNaNInf(data, "RMSNorm")

	promtest.RequireDelta(t, "simd_inf_count_total", inf, infBefore, 1)
	promtest.RequireDelta(t, "simd_nan_count_total(no Inf-only input)", nan, nanBefore, 0)
	promtest.RequireDelta(t, "simd_kernel_errors_total", errs, errsBefore, 1)
}

func TestCheckOutputBothNaNInf(t *testing.T) {
	metricsEnabled = true
	defer func() { metricsEnabled = true }()

	nan := metrics.SIMDNaNCount.WithLabelValues("Matmul")
	inf := metrics.SIMDInfCount.WithLabelValues("Matmul")
	nanBefore := promtest.Value(t, "simd_nan_count_total", nan)
	infBefore := promtest.Value(t, "simd_inf_count_total", inf)

	data := []float32{float32(math.NaN()), float32(math.Inf(-1)), 3.0}
	checkOutputNaNInf(data, "Matmul")

	promtest.RequireDelta(t, "simd_nan_count_total", nan, nanBefore, 1)
	promtest.RequireDelta(t, "simd_inf_count_total", inf, infBefore, 1)
}

func TestCheckOutputClean(t *testing.T) {
	metricsEnabled = true
	defer func() { metricsEnabled = true }()

	nan := metrics.SIMDNaNCount.WithLabelValues("Softmax")
	errs := metrics.SIMDKernelErrors.WithLabelValues("Softmax", "output_invalid")
	nanBefore := promtest.Value(t, "simd_nan_count_total", nan)
	errsBefore := promtest.Value(t, "simd_kernel_errors_total", errs)

	data := []float32{1.0, 2.0, 3.0}
	checkOutputNaNInf(data, "Softmax")

	promtest.RequireDelta(t, "simd_nan_count_total(clean output)", nan, nanBefore, 0)
	promtest.RequireDelta(t, "simd_kernel_errors_total(clean output)", errs, errsBefore, 0)
}
