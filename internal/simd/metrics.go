//go:build amd64 && cgo

package simd

import (
	"math"
	"time"

	"github.com/23skdu/longbow-quarrel/internal/metrics"
)

var metricsEnabled bool

func init() {
	metricsEnabled = true
}

type metricsWrapper struct {
	startTime time.Time
	kernel   string
	size     int
}

func (m *metricsWrapper) record() {
	if !metricsEnabled {
		return
	}

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

func checkOutputNaNInf(data []float32, kernel string) {
	if !metricsEnabled {
		return
	}

	nanCount := 0
	infCount := 0

	for _, v := range data {
		if math.IsNaN(float64(v)) {
			nanCount++
		}
		if math.IsInf(float64(v)) {
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

func initSIMDMetrics() {
	metrics.RecordSIMDLevel(GetSIMDLevel())
}
	if infCount > 0 {
		metrics.RecordSIMDInf(kernel)
	}
}

if nanCount > 0 || infCount > 0 {
		metrics.RecordSIMDKernelError(kernel, "output_invalid")
	}
}

func initSIMDMetrics() {
	metrics.RecordSIMDLevel(GetSIMDLevel())
}