package metrics

import (
	"os"
	"path/filepath"
	"regexp"
	"testing"
)

// The Grafana dashboard computes percentiles with
//
//	histogram_quantile(0.99, rate(<name>_bucket[5m]))
//
// which needs the _bucket series that only a Histogram exposes. A Summary
// without explicit Objectives exports no quantile series at all, so a panel
// pointed at one is permanently empty with no error anywhere.
//
// This test parses the metric declarations and the dashboard and asserts
// that every _bucket expression refers to a metric declared as a Histogram.
//
// The check is static rather than registry-based on purpose: a HistogramVec
// only appears in the default registry once a labelled observation has
// happened, so a registry snapshot would report false negatives for every
// vec metric that had not been used yet.

var (
	// promauto.NewHistogram(...) / NewHistogramVec(...) with a Name field.
	// The collector options are a struct literal, so the match has to span
	// braces.
	metricDeclRe = regexp.MustCompile(
		`promauto\.New(\w+)\([\s\S]{0,300}?Name:\s*"([a-zA-Z0-9_]+)"`)
	// histogram_quantile(<q>, rate(<name>_bucket[...]))
	bucketQueryRe = regexp.MustCompile(
		`histogram_quantile\([^,]+,\s*rate\(([a-zA-Z0-9_]+)_bucket`)
)

// TestInferenceDurationIsHistogram guards the metric type directly.
func TestInferenceDurationIsHistogram(t *testing.T) {
	decls := parseMetricDeclarations(t)

	kind, ok := decls["inference_duration_seconds"]
	if !ok {
		t.Fatal("inference_duration_seconds is not declared in internal/metrics")
	}
	if kind != "Histogram" && kind != "HistogramVec" {
		t.Fatalf("inference_duration_seconds is a %s, want a Histogram: the Grafana "+
			"latency panels query inference_duration_seconds_bucket, which only a "+
			"Histogram exposes", kind)
	}
}

// TestDashboardBucketQueriesHaveMatchingMetrics checks that every
// histogram_quantile() expression in the Grafana dashboard refers to a
// metric that is actually declared -- and declared as a Histogram.
func TestDashboardBucketQueriesHaveMatchingMetrics(t *testing.T) {
	dashPath := dashboardPath(t)
	if dashPath == "" {
		t.Skip("Grafana dashboard not found")
	}
	raw, err := os.ReadFile(dashPath)
	if err != nil {
		t.Fatalf("failed to read %s: %v", dashPath, err)
	}

	matches := bucketQueryRe.FindAllStringSubmatch(string(raw), -1)
	if len(matches) == 0 {
		t.Fatal("no histogram_quantile expressions found in the dashboard; the layout changed?")
	}

	decls := parseMetricDeclarations(t)
	for _, m := range matches {
		name := m[1]
		kind, ok := decls[name]
		if !ok {
			t.Errorf("dashboard queries %s_bucket but %s is not a declared metric", name, name)
			continue
		}
		if kind != "Histogram" && kind != "HistogramVec" {
			t.Errorf("dashboard queries %s_bucket but %s is a %s; only a Histogram exposes _bucket",
				name, name, kind)
		}
	}
}

// parseMetricDeclarations maps metric name -> collector kind, e.g.
// "inference_duration_seconds" -> "Histogram".
func parseMetricDeclarations(t *testing.T) map[string]string {
	t.Helper()

	out := map[string]string{}
	files := []string{
		filepath.Join("metrics.go"),
		filepath.Join("metrics_gemma4.go"),
		filepath.Join("metrics_offload.go"),
		filepath.Join("metrics_runtime.go"),
		filepath.Join("metrics_simd.go"),
		filepath.Join("turboquant.go"),
		filepath.Join("bg_flusher.go"),
	}
	for _, f := range files {
		raw, err := os.ReadFile(f)
		if err != nil {
			continue
		}
		for _, m := range metricDeclRe.FindAllStringSubmatch(string(raw), -1) {
			out[m[2]] = m[1]
		}
	}
	if len(out) == 0 {
		t.Fatal("no metric declarations parsed; the source layout changed?")
	}
	return out
}

func dashboardPath(t *testing.T) string {
	t.Helper()
	dir := "."
	for i := 0; i < 6; i++ {
		p := filepath.Join(dir, "grafana", "quarrel-dashboard.json")
		if _, err := os.Stat(p); err == nil {
			return p
		}
		dir = filepath.Join(dir, "..")
	}
	return ""
}
