package promtest

import (
	"errors"
	"fmt"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
)

type mockT struct {
	fatals []string
	errors []string
}

func (m *mockT) Helper() {}

func (m *mockT) Fatalf(format string, args ...any) {
	m.fatals = append(m.fatals, fmt.Sprintf(format, args...))
}

func (m *mockT) Errorf(format string, args ...any) {
	m.errors = append(m.errors, fmt.Sprintf(format, args...))
}

type mockObserverOnly struct{}

func (m mockObserverOnly) Observe(float64) {}

type errorMetric struct{}

func (e errorMetric) Desc() *prometheus.Desc {
	return prometheus.NewDesc("error_metric", "help", nil, nil)
}

func (e errorMetric) Write(*dto.Metric) error {
	return errors.New("write failure")
}

type emptyMetric struct{}

func (e emptyMetric) Desc() *prometheus.Desc {
	return prometheus.NewDesc("empty_metric", "help", nil, nil)
}

func (e emptyMetric) Write(*dto.Metric) error {
	return nil
}

func TestPromtest_Coverage(t *testing.T) {
	// Standard types
	counter := prometheus.NewCounter(prometheus.CounterOpts{Name: "c", Help: "h"})
	counter.Add(5)
	RequireValue(t, "counter", counter, 5)
	RequireDelta(t, "counter", counter, 2, 3)

	gauge := prometheus.NewGauge(prometheus.GaugeOpts{Name: "g", Help: "h"})
	gauge.Set(42)
	RequireValue(t, "gauge", gauge, 42)
	RequireDelta(t, "gauge", gauge, 40, 2)

	hist := prometheus.NewHistogram(prometheus.HistogramOpts{Name: "h", Help: "h"})
	hist.Observe(1.0)
	RequireValue(t, "hist", hist, 1)

	histVec := prometheus.NewHistogramVec(prometheus.HistogramOpts{Name: "hv", Help: "h"}, []string{"k"})
	child := histVec.WithLabelValues("v")
	child.Observe(0.5)
	RequireValue(t, "hist_child", child, 1)

	summary := prometheus.NewSummary(prometheus.SummaryOpts{Name: "s", Help: "h"})
	summary.Observe(10)
	RequireValue(t, "summary", summary, 1)

	// Test Error paths using mockT
	mt := &mockT{}

	// 1. Observer that is not prometheus.Metric
	val := Value(mt, "obs_only", mockObserverOnly{})
	if len(mt.fatals) != 1 || val != 0 {
		t.Fatalf("expected fatal on non-metric observer, got: %v", mt.fatals)
	}

	// 2. Unsupported source
	mt = &mockT{}
	val = Value(mt, "string_source", "invalid")
	if len(mt.fatals) != 1 || val != 0 {
		t.Fatalf("expected fatal on invalid source, got: %v", mt.fatals)
	}

	// 3. Error on Write
	mt = &mockT{}
	val = Value(mt, "err_metric", errorMetric{})
	if len(mt.fatals) != 1 || val != 0 {
		t.Fatalf("expected fatal on write error, got: %v", mt.fatals)
	}

	// 4. Metric with no value
	mt = &mockT{}
	val = Value(mt, "empty_metric", emptyMetric{})
	if len(mt.fatals) != 1 || val != 0 {
		t.Fatalf("expected fatal on metric with no value, got: %v", mt.fatals)
	}

	// 5. RequireDelta mismatch
	mt = &mockT{}
	RequireDelta(mt, "counter", counter, 0, 100)
	if len(mt.errors) != 1 {
		t.Fatalf("expected error on RequireDelta mismatch, got: %v", mt.errors)
	}

	// 6. RequireValue mismatch
	mt = &mockT{}
	RequireValue(mt, "counter", counter, 999)
	if len(mt.errors) != 1 {
		t.Fatalf("expected error on RequireValue mismatch, got: %v", mt.errors)
	}
}
