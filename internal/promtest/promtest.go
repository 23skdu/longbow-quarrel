// Package promtest provides small assertions over Prometheus metrics for use
// in tests: it reads a single metric (an unlabelled one, one child of a label
// vector, or a histogram Observer) and checks how it moved between two points
// in a test.
package promtest

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
)

// Value returns the current value of a single Prometheus metric: counters and
// gauges report their value, histograms and summaries their observation count.
//
// metric may be any prometheus.Metric (Counter, Gauge, unlabelled histogram)
// or a prometheus.Observer, which is what a HistogramVec child is.
func Value(t *testing.T, name string, metric any) float64 {
	t.Helper()

	var m prometheus.Metric
	switch v := metric.(type) {
	case prometheus.Metric:
		m = v
	case prometheus.Observer:
		obs, ok := v.(prometheus.Metric)
		if !ok {
			t.Fatalf("%s: histogram child %T does not expose its metric", name, v)
		}
		m = obs
	default:
		t.Fatalf("%s: unsupported metric source %T", name, metric)
	}

	pb := &dto.Metric{}
	if err := m.Write(pb); err != nil {
		t.Fatalf("%s: write metric: %v", name, err)
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
func RequireDelta(t *testing.T, name string, metric any, before, want float64) {
	t.Helper()
	if got := Value(t, name, metric) - before; got != want {
		t.Errorf("%s: changed by %v, want %v", name, got, want)
	}
}

// RequireValue asserts the metric currently holds exactly want.
func RequireValue(t *testing.T, name string, metric any, want float64) {
	t.Helper()
	if got := Value(t, name, metric); got != want {
		t.Errorf("%s: = %v, want %v", name, got, want)
	}
}
