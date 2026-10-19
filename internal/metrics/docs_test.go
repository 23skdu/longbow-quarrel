package metrics

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// metricRowRe matches a row of the metrics reference table:
//
//	| `some_metric_name` | Counter | Description. |
var metricRowRe = regexp.MustCompile("^\\|\\s*`([a-zA-Z0-9_]+)`\\s*\\|")

// TestDocsMetrics_OnlyDocumentsRealMetrics checks that every metric named in
// docs/metrics.md is actually declared in the code.
//
// Ten metrics were documented that did not exist, so the reference described
// an observability surface that was not there.
func TestDocsMetrics_OnlyDocumentsRealMetrics(t *testing.T) {
	docPath := docsMetricsPath(t)
	if docPath == "" {
		t.Skip("docs/metrics.md not found")
	}
	raw, err := os.ReadFile(docPath)
	if err != nil {
		t.Fatalf("failed to read %s: %v", docPath, err)
	}

	documented := map[string]bool{}
	for _, line := range strings.Split(string(raw), "\n") {
		if m := metricRowRe.FindStringSubmatch(line); m != nil {
			documented[m[1]] = true
		}
	}
	if len(documented) == 0 {
		t.Fatal("no metric rows were found in the reference; the table layout changed?")
	}

	declared := declaredMetricNames(t)

	for name := range documented {
		if !declared[name] {
			t.Errorf("docs/metrics.md documents %q, but no such metric is declared", name)
		}
	}
}

// TestDocsMetrics_CoversEveryDeclaredMetric is the other direction: a
// metric that exists but is not documented is invisible to anyone reading
// the reference.
//
// The Go runtime and process collectors (go_*, process_*, promhttp_*) are
// registered by the standard library, not by this project, so they are out
// of scope.
func TestDocsMetrics_CoversEveryDeclaredMetric(t *testing.T) {
	docPath := docsMetricsPath(t)
	if docPath == "" {
		t.Skip("docs/metrics.md not found")
	}
	raw, err := os.ReadFile(docPath)
	if err != nil {
		t.Fatalf("failed to read %s: %v", docPath, err)
	}

	documented := map[string]bool{}
	for _, line := range strings.Split(string(raw), "\n") {
		if m := metricRowRe.FindStringSubmatch(line); m != nil {
			documented[m[1]] = true
		}
	}

	for name := range declaredMetricNames(t) {
		if !isProjectMetric(name) {
			continue
		}
		if !documented[name] {
			t.Errorf("metric %q is declared but not documented in docs/metrics.md", name)
		}
	}
}

// isProjectMetric reports whether a metric belongs to this project rather
// than to the Go runtime or the process collectors.
func isProjectMetric(name string) bool {
	for _, prefix := range []string{"go_", "process_", "promhttp_"} {
		if strings.HasPrefix(name, prefix) {
			return false
		}
	}
	return true
}

// declaredMetricNames returns the full names of every Prometheus metric
// declared anywhere in the repository.
//
// The names are collected statically rather than from the default registry,
// because a metric declared in internal/device is not registered in a test
// binary that does not import that package. The Namespace and Subsystem
// options are folded into the name, the way the Prometheus client does.
func declaredMetricNames(t *testing.T) map[string]bool {
	t.Helper()

	root := repoRoot(t)
	if root == "" {
		t.Skip("repository root not found")
	}

	out := map[string]bool{}
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		if info.IsDir() {
			switch info.Name() {
			case ".git", "node_modules", "vendor":
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") {
			return nil
		}
		// Test files declare throwaway metrics that are not part of the
		// shipped observability surface.
		if strings.HasSuffix(path, "_test.go") {
			return nil
		}
		collectMetricNames(path, out)
		return nil
	})
	if err != nil {
		t.Fatalf("failed to walk the repository: %v", err)
	}

	if len(out) == 0 {
		t.Fatal("no metrics were found; the source layout changed?")
	}
	return out
}

// collectMetricNames parses one file and records the full name of every
// metric declared in it.
func collectMetricNames(path string, out map[string]bool) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, path, nil, 0)
	if err != nil {
		return
	}

	ast.Inspect(f, func(n ast.Node) bool {
		cl, ok := n.(*ast.CompositeLit)
		if !ok {
			return true
		}
		// Only the *Opts literals declare a metric.
		if !isOptsType(cl.Type) {
			return true
		}
		ns, sub, name := "", "", ""
		for _, elt := range cl.Elts {
			kv, ok := elt.(*ast.KeyValueExpr)
			if !ok {
				continue
			}
			key, ok := kv.Key.(*ast.Ident)
			if !ok {
				continue
			}
			lit, ok := kv.Value.(*ast.BasicLit)
			if !ok || lit.Kind != token.STRING {
				continue
			}
			v, err := strconv.Unquote(lit.Value)
			if err != nil {
				continue
			}
			switch key.Name {
			case "Namespace":
				ns = v
			case "Subsystem":
				sub = v
			case "Name":
				name = v
			}
		}
		if name == "" {
			return true
		}
		full := name
		if ns != "" {
			full = ns + "_" + full
		}
		if sub != "" {
			full = ns + "_" + sub + "_" + name
		}
		out[full] = true
		return true
	})
}

// isOptsType reports whether a composite literal type is one of the
// Prometheus *Opts structs.
func isOptsType(expr ast.Expr) bool {
	switch t := expr.(type) {
	case *ast.SelectorExpr:
		return strings.HasSuffix(t.Sel.Name, "Opts")
	case *ast.Ident:
		return strings.HasSuffix(t.Name, "Opts")
	}
	return false
}

// repoRoot locates the repository root by walking up until a go.mod is
// found.
func repoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		return ""
	}
	for i := 0; i < 8; i++ {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	return ""
}

// docsMetricsPath locates docs/metrics.md.
func docsMetricsPath(t *testing.T) string {
	t.Helper()
	root := repoRoot(t)
	if root == "" {
		return ""
	}
	p := filepath.Join(root, "docs", "metrics.md")
	if _, err := os.Stat(p); err != nil {
		return ""
	}
	return p
}
