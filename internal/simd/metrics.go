package simd

import (
	"os"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/mathx"
	"github.com/23skdu/longbow-quarrel/internal/metrics"
)

// SIMD kernel instrumentation.
//
// The metrics wrapper existed but was never instantiated, so all 14 SIMD
// histograms and counters in internal/metrics/metrics_simd.go exported a
// permanent zero -- the observability was documented but not real.
//
// It is **off by default** on purpose. Instrumenting a kernel costs a
// timestamp pair plus a Prometheus histogram observation, and these kernels
// run thousands of times per generated token; a default-on cost in the
// inference hot path is not acceptable for metrics nobody asked for.
//
// Enable it with QUARREL_SIMD_METRICS=1 (or "true"/"yes"/"1").
//
// The switch also controls the NaN/Inf output scan, which is O(n) over the
// kernel output and would otherwise dominate the cost.
var simdMetricsEnabled = os.Getenv("QUARREL_SIMD_METRICS") == "1" ||
	os.Getenv("QUARREL_SIMD_METRICS") == "true" ||
	os.Getenv("QUARREL_SIMD_METRICS") == "yes"

// SIMDMetricsEnabled reports whether the SIMD kernel instrumentation is on.
func SIMDMetricsEnabled() bool {
	return simdMetricsEnabled
}

// metricsWrapper records one SIMD kernel invocation.
type metricsWrapper struct {
	startTime time.Time
	kernel    string
	size      int
}

func (m *metricsWrapper) record() {
	if !simdMetricsEnabled {
		return
	}

	metrics.RecordSIMDDispatch(m.kernel)
	duration := time.Since(m.startTime)
	sizeClass := classifySize(m.size)

	metrics.RecordSIMDKernelDuration(m.kernel, duration, sizeClass)

	switch m.kernel {
	case "Softmax":
		metrics.RecordSIMDSoftmaxDuration(duration)
	case "RMSNorm":
		metrics.RecordSIMDRMSNormDuration(duration)
	case "Matmul":
		metrics.RecordSIMDMatmulDuration(duration)
	case "Attention":
		metrics.RecordSIMDAttentionDuration(duration)
	case "FusedMLP":
		metrics.RecordSIMDFusedMLPDuration(duration)
	case "RoPE":
		metrics.RecordSIMDRoPEDuration(duration)
	case "SwiGLU":
		metrics.RecordSIMDSwiGLUDuration(duration)
	}
}

// newMetricsWrapper starts the timer for a kernel invocation. When the
// instrumentation is off it returns a wrapper whose record() is a no-op, so
// the cost is a single predictable branch.
func newMetricsWrapper(kernel string, size int) *metricsWrapper {
	if !simdMetricsEnabled {
		return &metricsWrapper{kernel: kernel, size: size}
	}
	return &metricsWrapper{startTime: time.Now(), kernel: kernel, size: size}
}

func classifySize(n int) string {
	switch {
	case n < 64:
		return "tiny"
	case n < 256:
		return "small"
	case n < 1024:
		return "medium"
	case n < 4096:
		return "large"
	case n < 16384:
		return "xlarge"
	default:
		return "xxlarge"
	}
}

// checkOutputNaNInf counts the NaN and Inf values in a kernel output and
// records them.
//
// The scan is O(n) over the output, so it only runs when the SIMD
// instrumentation is enabled.
func checkOutputNaNInf(data []float32, kernel string) {
	if !simdMetricsEnabled {
		return
	}

	nanCount := 0
	infCount := 0

	for _, v := range data {
		if mathx.IsNaN(float64(v)) {
			nanCount++
		}
		if mathx.IsInf(float64(v), 0) {
			infCount++
		}
	}

	if nanCount > 0 {
		metrics.RecordSIMDNaN(kernel)
	}
	if infCount > 0 {
		metrics.RecordSIMDInf(kernel)
	}

	if nanCount > 0 || infCount > 0 {
		metrics.RecordSIMDKernelError(kernel, "output_invalid")
	}
}
