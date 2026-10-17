// Package promtest provides small assertions over Prometheus metrics for use
// in tests: it reads a single metric (an unlabelled one, one child of a label
// vector, or a histogram Observer) and checks how it moved between two points
// in a test.
package promtest

import (
	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
)

// TestingT defines the subset of *testing.T used by promtest assertions.
type TestingT interface {
	Helper()
	Fatalf(format string, args ...any)
	Errorf(format string, args ...any)
}

// Value returns the current value of a single Prometheus metric: counters and
// gauges report their value, histograms and summaries their observation count.
//
// metric may be any prometheus.Metric (Counter, Gauge, unlabelled histogram)
// or a prometheus.Observer, which is what a HistogramVec child is.
func Value(t TestingT, name string, metric any) float64 {
	t.Helper()

	var m prometheus.Metric
	switch v := metric.(type) {
	case prometheus.Observer:
		obs, ok := v.(prometheus.Metric)
		if !ok {
			t.Fatalf("%s: histogram child %T does not expose its metric", name, v)
			return 0
		}
		m = obs
	case prometheus.Metric:
		m = v
	default:
		t.Fatalf("%s: unsupported metric source %T", name, metric)
		return 0
	}

	pb := &dto.Metric{}
	if err := m.Write(pb); err != nil {
		t.Fatalf("%s: write metric: %v", name, err)
		return 0
	}
	switch {
	case pb.Counter != nil:
		return pb.Counter.GetValue()
	case pb.Gauge != nil:
		return pb.Gauge.GetValue()
	case pb.Histogram != nil:
		return float64(pb.Histogram.GetSampleCount())
	case pb.Summary != nil:
		return float64(pb.Summary.GetSampleCount())
	default:
		t.Fatalf("%s: metric has no value", name)
		return 0
	}
}

// RequireDelta asserts the metric moved by exactly want since before was
// taken.
func RequireDelta(t TestingT, name string, metric any, before, want float64) {
	t.Helper()
	if got := Value(t, name, metric) - before; got != want {
		t.Errorf("%s: changed by %v, want %v", name, got, want)
	}
}

// RequireValue asserts the metric currently holds exactly want.
func RequireValue(t TestingT, name string, metric any, want float64) {
	t.Helper()
	if got := Value(t, name, metric); got != want {
		t.Errorf("%s: = %v, want %v", name, got, want)
	}
}
