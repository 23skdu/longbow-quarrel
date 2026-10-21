package metrics

import (
	"encoding/json"
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

// productionMetricAllowlist contains declared metrics that are intentionally not
// written by the standard engine execution path on this host (e.g. experimental backends).
var productionMetricAllowlist = map[string]string{}

// TestMetrics_WrittenByProductionCode verifies that every declared metric is written
// by production code or is on the explicit allowlist, and that no dashboard panel reads
// a metric that is never written.
func TestMetrics_WrittenByProductionCode(t *testing.T) {
	root := repoRoot(t)
	if root == "" {
		t.Skip("repository root not found")
	}

	type metricInfo struct {
		varName string
		file    string
	}
	metricDecls := map[string]metricInfo{}     // fullName -> info
	varToMetric := map[string]string{}         // varName -> fullName
	funcToVars := map[string]map[string]bool{} // funcName -> set of varNames written

	fset := token.NewFileSet()

	// 1. Collect metric declarations and helper functions across all non-test files
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			if info != nil && info.IsDir() {
				switch info.Name() {
				case ".git", "node_modules", "vendor":
					return filepath.SkipDir
				}
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
			return nil
		}

		f, err := parser.ParseFile(fset, path, nil, 0)
		if err != nil {
			return nil
		}

		// Look for variable declarations initialized with promauto.New*
		ast.Inspect(f, func(n ast.Node) bool {
			genDecl, ok := n.(*ast.GenDecl)
			if ok && genDecl.Tok == token.VAR {
				for _, spec := range genDecl.Specs {
					valSpec, ok := spec.(*ast.ValueSpec)
					if !ok {
						continue
					}
					for i, nameIdent := range valSpec.Names {
						if i >= len(valSpec.Values) {
							continue
						}
						val := valSpec.Values[i]
						// Find metric name insideOpts
						fullName := extractMetricNameFromExpr(val)
						if fullName != "" && isProjectMetric(fullName) {
							metricDecls[fullName] = metricInfo{varName: nameIdent.Name, file: path}
							varToMetric[nameIdent.Name] = fullName
						}
					}
				}
			}
			return true
		})

		// Look for function declarations and which metric vars they reference
		for _, decl := range f.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Body == nil {
				continue
			}
			fnName := fn.Name.Name
			if funcToVars[fnName] == nil {
				funcToVars[fnName] = map[string]bool{}
			}
			ast.Inspect(fn.Body, func(n ast.Node) bool {
				switch x := n.(type) {
				case *ast.Ident:
					funcToVars[fnName][x.Name] = true
				case *ast.SelectorExpr:
					funcToVars[fnName][x.Sel.Name] = true
				}
				return true
			})
		}

		return nil
	})
	if err != nil {
		t.Fatalf("failed scanning metric declarations: %v", err)
	}

	// 2. Collect which functions and variables are called/referenced in production code
	calledFuncs := map[string]bool{}
	referencedVars := map[string]bool{}

	err = filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			if info != nil && info.IsDir() {
				switch info.Name() {
				case ".git", "node_modules", "vendor":
					return filepath.SkipDir
				}
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
			return nil
		}

		f, err := parser.ParseFile(fset, path, nil, 0)
		if err != nil {
			return nil
		}

		ast.Inspect(f, func(n ast.Node) bool {
			// Skip the metric variable declaration itself
			if genDecl, ok := n.(*ast.GenDecl); ok && genDecl.Tok == token.VAR {
				for _, spec := range genDecl.Specs {
					if valSpec, ok := spec.(*ast.ValueSpec); ok {
						for i := range valSpec.Names {
							if i < len(valSpec.Values) && extractMetricNameFromExpr(valSpec.Values[i]) != "" {
								return false
							}
						}
					}
				}
			}
			switch expr := n.(type) {
			case *ast.CallExpr:
				switch fn := expr.Fun.(type) {
				case *ast.Ident:
					calledFuncs[fn.Name] = true
				case *ast.SelectorExpr:
					calledFuncs[fn.Sel.Name] = true
				}
			case *ast.SelectorExpr:
				referencedVars[expr.Sel.Name] = true
			case *ast.Ident:
				referencedVars[expr.Name] = true
			}
			return true
		})
		return nil
	})
	if err != nil {
		t.Fatalf("failed scanning production callers: %v", err)
	}

	// 3. Mark written metrics
	writtenMetrics := map[string]bool{}

	// If a helper function is called, mark all vars it references
	for fnName, referencedInFn := range funcToVars {
		if calledFuncs[fnName] {
			for varName := range referencedInFn {
				if metricName, ok := varToMetric[varName]; ok {
					writtenMetrics[metricName] = true
				}
			}
		}
	}

	// If a variable itself is referenced with method calls (e.g. .Inc, .Set, .Observe)
	for varName := range referencedVars {
		if metricName, ok := varToMetric[varName]; ok {
			writtenMetrics[metricName] = true
		}
	}

	// 4. Assert that every declared metric is written or on the allowlist
	for metricName, info := range metricDecls {
		if writtenMetrics[metricName] {
			continue
		}
		if _, ok := productionMetricAllowlist[metricName]; ok {
			continue
		}
		t.Errorf("metric %q (variable %s in %s) is declared but never written by production code",
			metricName, info.varName, filepath.Base(info.file))
	}

	// 5. Assert that every metric queried by the Grafana dashboard is written or allowlisted
	dashPath := dashboardPath(t)
	if dashPath != "" {
		dashRaw, err := os.ReadFile(dashPath)
		if err == nil {
			var dashboard struct {
				Panels []struct {
					Title   string `json:"title"`
					Targets []struct {
						Expr string `json:"expr"`
					} `json:"targets"`
				} `json:"panels"`
			}
			if err := json.Unmarshal(dashRaw, &dashboard); err == nil {
				for _, panel := range dashboard.Panels {
					for _, target := range panel.Targets {
						for _, word := range regexp.MustCompile(`\b([a-z][a-z0-9_]*)\b`).FindAllStringSubmatch(target.Expr, -1) {
							name := word[1]
							for _, suffix := range []string{"_bucket", "_sum", "_count"} {
								name = strings.TrimSuffix(name, suffix)
							}
							if _, isDecl := metricDecls[name]; isDecl {
								if !writtenMetrics[name] && productionMetricAllowlist[name] == "" {
									t.Errorf("dashboard panel %q queries metric %q which is never written by production code",
										panel.Title, name)
								}
							}
						}
					}
				}
			}
		}
	}
}

func extractMetricNameFromExpr(expr ast.Expr) string {
	cl, ok := expr.(*ast.CompositeLit)
	if !ok {
		// Might be CallExpr: promauto.NewCounter(prometheus.CounterOpts{...})
		if call, ok := expr.(*ast.CallExpr); ok && len(call.Args) > 0 {
			return extractMetricNameFromExpr(call.Args[0])
		}
		return ""
	}
	if !isOptsType(cl.Type) {
		return ""
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
		return ""
	}
	full := name
	if ns != "" {
		full = ns + "_" + full
	}
	if sub != "" {
		full = ns + "_" + sub + "_" + name
	}
	return full
}
