//go:build amd64 && cgo

package simd

import (
	"math"
	"testing"
	"time"
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

func TestMetricsRecordCalls(t *testing.T) {
	metricsEnabled = true

	m := &metricsWrapper{
		startTime: time.Now(),
		kernel:   "Softmax",
		size:     256,
	}
	m.record()
}

func TestMetricsRecordAllKernels(t *testing.T) {
	metricsEnabled = true

	kernels := []string{"Softmax", "RMSNorm", "Matmul", "Attention", "FusedMLP", "RoPE", "SwiGLU", "Unknown"}
	for _, k := range kernels {
		m := &metricsWrapper{
			startTime: time.Now(),
			kernel:   k,
			size:     1000,
		}
		m.record()
	}
}

func TestMetricsDisabled(t *testing.T) {
	metricsEnabled = false

	m := &metricsWrapper{
		startTime: time.Now(),
		kernel:   "Softmax",
		size:     256,
	}
	m.record()

	checkOutputNaNInf([]float32{1, 2, 3}, "Softmax")
}

func TestCheckOutputNaN(t *testing.T) {
	metricsEnabled = true

	data := []float32{1.0, float32(math.NaN()), 3.0}
	checkOutputNaNInf(data, "Softmax")
}

func TestCheckOutputInf(t *testing.T) {
	metricsEnabled = true

	data := []float32{1.0, float32(math.Inf(1)), 3.0}
	checkOutputNaNInf(data, "RMSNorm")
}

func TestCheckOutputBothNaNInf(t *testing.T) {
	metricsEnabled = true

	data := []float32{float32(math.NaN()), float32(math.Inf(-1)), 3.0}
	checkOutputNaNInf(data, "Matmul")
}

func TestCheckOutputClean(t *testing.T) {
	metricsEnabled = true

	data := []float32{1.0, 2.0, 3.0}
	checkOutputNaNInf(data, "Softmax")
}
