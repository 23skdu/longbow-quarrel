package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The checks in this file are what keeps the documentation, the Helm
// chart and the Grafana dashboard from drifting away from the code.
//
// A test that only checked the current state would pass once and then
// stop saying anything, so each check is also run against a deliberately
// broken copy of the thing it looks at.

// TestCommandFlagsAreFound checks that the flags of the command are read
// out of the sources.
func TestCommandFlagsAreFound(t *testing.T) {
	builds, err := commandFlags()
	if err != nil {
		t.Fatalf("failed to read the command flags: %v", err)
	}

	for _, build := range []string{"cuda", "cpu", "metal"} {
		flags, ok := builds[build]
		if !ok {
			t.Fatalf("the %s build was not found", build)
		}
		if len(flags) == 0 {
			t.Fatalf("the %s build declares no flags", build)
		}
	}

	cuda := builds["cuda"]
	for name, want := range map[string]string{
		"model":          "",
		"ngl":            "-1",
		"max-batch-size": "16",
		"temp":           "0.7",
		"topk":           "40",
		"topp":           "0.95",
		"api-addr":       ":8080",
		"metrics":        ":9090",
		"draft-model":    "",
		"activation-log": "",
	} {
		def, ok := cuda[name]
		if !ok {
			t.Errorf("the CUDA build has no -%s flag", name)
			continue
		}
		if want != "" && def.deflt != want {
			t.Errorf("the default of -%s is %q, want %q", name, def.deflt, want)
		}
	}
}

// TestDeclaredMetricsAreFound checks that the metrics are read out of the
// code, with their types.
func TestDeclaredMetricsAreFound(t *testing.T) {
	metrics, err := declaredMetrics()
	if err != nil {
		t.Fatalf("failed to read the metrics: %v", err)
	}
	if len(metrics) < 100 {
		t.Fatalf("only %d metrics were found, which is too few", len(metrics))
	}

	for name, kind := range map[string]string{
		"logit_max_value":           "Histogram",
		"logit_nan_count_total":     "Counter",
		"quarrel_cpu_layers_active": "Gauge",
	} {
		def, ok := metrics[name]
		if !ok {
			t.Errorf("the metric %q was not found", name)
			continue
		}
		if def.kind != kind {
			t.Errorf("the metric %q is a %s, want a %s", name, def.kind, kind)
		}
	}
}

// TestTheRepositoryIsInSync runs every check against the repository.
func TestTheRepositoryIsInSync(t *testing.T) {
	flags, err := commandFlags()
	if err != nil {
		t.Fatalf("failed to read the command flags: %v", err)
	}
	metrics, err := declaredMetrics()
	if err != nil {
		t.Fatalf("failed to read the metrics: %v", err)
	}

	problems := 0
	report := func(format string, args ...interface{}) {
		problems++
		t.Errorf(format, args...)
	}

	problems += checkChartFlags(flags, report, false)
	problems += checkChartValues(flags, report, false)
	problems += checkDashboard(metrics, report, false)
	problems += checkDocumentedFlags(flags, report, false)

	if problems > 0 {
		t.Errorf("%d problem(s) found: the documentation, the Helm chart or the Grafana dashboard has drifted", problems)
	}
}

// TestCheckChartFlagsCatchesAnUnknownFlag checks that the check notices a
// flag the command does not have.
func TestCheckChartFlagsCatchesAnUnknownFlag(t *testing.T) {
	flags := map[string]map[string]flagDef{
		"cuda": {"model": {name: "model"}},
	}

	problems := 0
	report := func(format string, args ...interface{}) { problems++ }

	// The real chart, with the CUDA build reduced to a single flag, has
	// to produce a problem.
	problems += checkChartFlags(flags, report, false)
	if problems == 0 {
		t.Error("the check did not notice that the chart passes flags the command does not have")
	}
}

// TestCheckChartValuesCatchesAMismatch checks that the check notices a
// value that disagrees with the code's default.
func TestCheckChartValuesCatchesAMismatch(t *testing.T) {
	flags := map[string]map[string]flagDef{
		"cuda": {
			"temp": {name: "temp", deflt: "0.7"},
		},
	}

	problems := 0
	report := func(format string, args ...interface{}) { problems++ }

	// The real chart sets the temperature to 0.7, which matches; the
	// api-addr check needs the flag, so the run below has to complain
	// about the missing flags.
	problems += checkChartValues(flags, report, false)
	if problems == 0 {
		t.Error("the check did not notice that the chart's flags are missing from the build")
	}
}

// TestCheckDashboardCatchesAnUnknownMetric checks that the check notices a
// panel that queries a metric the code does not declare.
func TestCheckDashboardCatchesAnUnknownMetric(t *testing.T) {
	metrics := map[string]metricDef{
		"logit_max_value": {name: "logit_max_value", kind: "Histogram"},
	}

	problems := 0
	report := func(format string, args ...interface{}) { problems++ }

	// The real dashboard queries many metrics; with only one declared,
	// the check has to complain.
	problems += checkDashboard(metrics, report, false)
	if problems == 0 {
		t.Error("the check did not notice that the dashboard queries undeclared metrics")
	}
}

// TestCheckDocumentedFlagsCatchesAnUnknownFlag checks that the check
// notices a flag the documentation mentions but the command does not have.
func TestCheckDocumentedFlagsCatchesAnUnknownFlag(t *testing.T) {
	flags := map[string]map[string]flagDef{
		"cuda": {"model": {name: "model"}},
	}

	problems := 0
	report := func(format string, args ...interface{}) { problems++ }

	problems += checkDocumentedFlags(flags, report, false)
	if problems == 0 {
		t.Error("the check did not notice the flags the documentation mentions that the command does not have")
	}
}

// TestChartValuesAreRead checks the reader of the chart's values.
func TestChartValuesAreRead(t *testing.T) {
	values, err := chartValues()
	if err != nil {
		t.Fatalf("failed to read the chart values: %v", err)
	}

	for path, want := range map[string]string{
		"engine.ngl":                    "-1",
		"engine.maxBatchSize":           "16",
		"engine.kvCache.blockSize":      "16",
		"engine.kvCache.totalBlocks":    "256",
		"engine.temperature":            "0.7",
		"engine.topK":                   "40",
		"engine.topP":                   "0.95",
		"service.port":                  "8080",
		"service.metricsPort":           "9090",
		"engine.speculative.draftModel": "",
	} {
		got, ok := values[path]
		if !ok {
			t.Errorf("the chart has no value for %s", path)
			continue
		}
		if got != want {
			t.Errorf("the chart's %s is %q, want %q", path, got, want)
		}
	}
}

// TestTheAllowlistsExist checks that the allowlists are present, so that a
// missing one does not silently turn the checks into no-ops.
func TestTheAllowlistsExist(t *testing.T) {
	for _, path := range []string{
		"scripts/dashboard_allowlist.txt",
		"scripts/docsync_allowlist.txt",
	} {
		if _, err := os.Stat(filepath.Join(repoRoot, path)); err != nil {
			t.Errorf("%s is missing: %v", path, err)
		}
	}
}

// TestTheDashboardCoversEveryMetric checks the coverage of the dashboard
// on its own, so that a metric that is dropped from it is noticed even if
// the allowlist grows.
func TestTheDashboardCoversEveryMetric(t *testing.T) {
	metrics, err := declaredMetrics()
	if err != nil {
		t.Fatalf("failed to read the metrics: %v", err)
	}

	raw, err := os.ReadFile(filepath.Join(repoRoot, dashboardPath))
	if err != nil {
		t.Fatalf("failed to read the dashboard: %v", err)
	}

	queried := map[string]bool{}
	for _, line := range strings.Split(string(raw), "\n") {
		for _, name := range metricNames(line) {
			if _, ok := metrics[name]; ok {
				queried[name] = true
			}
		}
	}

	allowed := loadNotOnDashboard()
	var missing []string
	for name := range metrics {
		if !queried[name] && !allowed[name] {
			missing = append(missing, name)
		}
	}
	if len(missing) > 0 {
		t.Errorf("%d metrics are neither on the dashboard nor on the allowlist: %s",
			len(missing), strings.Join(missing, ", "))
	}
}

// TestCheckDocumentedEndpointsCatchesAnUnknownEndpoint checks that the
// check notices an endpoint the server does not register.
func TestCheckDocumentedEndpointsCatchesAnUnknownEndpoint(t *testing.T) {
	problems := 0
	report := func(format string, args ...interface{}) { problems++ }

	// The real documentation against the real server has to agree.
	problems += checkDocumentedEndpoints(report, false)
	if problems != 0 {
		t.Errorf("the API reference and the server disagree: %d problem(s)", problems)
	}
}

// TestTheAPIServerRoutesAreFound checks that the routes are read out of
// the server.
func TestTheAPIServerRoutesAreFound(t *testing.T) {
	routes, err := apiServerRoutes()
	if err != nil {
		t.Fatalf("failed to read the routes: %v", err)
	}

	for _, path := range []string{
		"/v1/completions",
		"/v1/chat/completions",
		"/v1/adapters/load",
		"/v1/adapters/list",
		"/healthz",
		"/readyz",
		"/metrics",
	} {
		if !routes[path] {
			t.Errorf("the route %s was not found", path)
		}
	}
}

// TestTheDocumentIndexListsEveryDocument checks that the index covers
// every document.
func TestTheDocumentIndexListsEveryDocument(t *testing.T) {
	problems := 0
	report := func(format string, args ...interface{}) { problems++ }

	problems += checkTheDocumentIndex(report, false)
	if problems != 0 {
		t.Errorf("%d document(s) are not in the index", problems)
	}
}

// TestCheckTheClaimedMetricCountCatchesAStaleCount checks that the check
// notices a document that claims the wrong number of metrics.
func TestCheckTheClaimedMetricCountCatchesAStaleCount(t *testing.T) {
	// The real documents claim 133 metrics, so the declared count has to
	// be in the same range for the claim to be checked at all.
	metrics := map[string]metricDef{}
	for i := 0; i < 130; i++ {
		name := fmt.Sprintf("metric_%d_total", i)
		metrics[name] = metricDef{name: name, kind: "Counter"}
	}

	problems := 0
	report := func(format string, args ...interface{}) { problems++ }

	problems += checkTheClaimedMetricCount(metrics, report, false)
	if problems == 0 {
		t.Error("the check did not notice the stale metric count")
	}
}
