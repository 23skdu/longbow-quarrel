// Command docsync checks that the documentation, the Helm chart and the
// Grafana dashboard still describe the code.
//
// Three things had drifted:
//
//   - the Helm chart passed flags that the command line no longer has,
//     and it did not expose the options that were added since;
//   - the Grafana dashboard queried 27 of the 133 declared metrics, and
//     one of the panels it had queried a histogram as if it were a gauge;
//   - the reference documentation listed flags and metrics that had
//     changed.
//
// The checks are:
//
//  1. every flag the Helm chart passes must be a flag the command has;
//  2. every value the Helm chart sets must match the default the code
//     uses, so that the chart does not quietly change the behaviour;
//  3. every metric the Grafana dashboard queries must be declared, in a
//     query that matches the metric's type, and every declared metric
//     must be on the dashboard;
//  4. every flag the reference documentation mentions must exist.
//
// Usage:
//
//	go run ./scripts/docsync
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

const (
	chartDir        = "helm/quarrel"
	chartValuesPath = "helm/quarrel/values.yaml"
	chartDeployment = "helm/quarrel/templates/deployment.yaml"
	chartService    = "helm/quarrel/templates/service.yaml"
	dashboardPath   = "grafana/quarrel-dashboard.json"
	metricsDir      = "internal/metrics"
	commandDir      = "cmd/quarrel"
)

// The command line builds. The chart deploys the CUDA image, so the CUDA
// build is the one whose flags the chart has to agree with.
var buildFiles = map[string][]string{
	"cuda":  {"cmd/quarrel/main.go"},
	"cpu":   {"cmd/quarrel/main_cpu.go"},
	"metal": {"cmd/quarrel/main_metal.go"},
	"tpu":   {"cmd/quarrel/main_tpu.go"},
}

// commandDirs are the directories whose commands the documentation
// describes.
var commandDirs = []string{"cmd"}

// flagDef is a flag the command declares.
type flagDef struct {
	name  string
	kind  string
	deflt string
}

// metricDef is a metric the code declares.
type metricDef struct {
	name string
	kind string
}

func main() {
	verbose := flag.Bool("v", false, "print every check")
	flag.Parse()

	problems := 0
	report := func(format string, args ...interface{}) {
		problems++
		fmt.Printf("error: "+format+"\n", args...)
	}

	flags, err := commandFlags()
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to read the command line flags: %v\n", err)
		os.Exit(1)
	}
	metrics, err := declaredMetrics()
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to read the metrics: %v\n", err)
		os.Exit(1)
	}

	problems += checkChartFlags(flags, report, *verbose)
	problems += checkChartValues(flags, report, *verbose)
	problems += checkDashboard(metrics, report, *verbose)
	problems += checkDocumentedFlags(flags, report, *verbose)
	problems += checkDocumentedEndpoints(report, *verbose)
	problems += checkPinnedVersions(report, *verbose)
	problems += checkTheClaimedMetricCount(metrics, report, *verbose)
	problems += checkTheDocumentIndex(report, *verbose)

	if problems > 0 {
		fmt.Printf("\ndocsync: %d problem(s) found\n", problems)
		os.Exit(1)
	}
	fmt.Println("docsync: the documentation, the Helm chart and the Grafana dashboard agree with the code")
}

// repoRoot is the root of the repository, found by walking up to the
// go.mod. The tool is run from the repository root, but the test runs it
// from its own directory.
var repoRoot = findRepoRoot()

// repoFS scopes every file access to the repository. os.Root refuses a
// path that escapes the root - through "..", an absolute path, or a
// symbolic link that points outside - so a name that came from a
// directory listing or a glob can never reach a file the repository does
// not own.
var repoFS = openRepoRoot()

// openRepoRoot opens the repository root for scoped access.
func openRepoRoot() *os.Root {
	root, err := os.OpenRoot(repoRoot)
	if err != nil {
		// The root could not be opened; the checks that use it report the
		// error. An unopenable root is not a reason to walk the whole
		// filesystem with plain path joins.
		return nil
	}
	return root
}

// readRepoFile reads a file relative to the repository root.
func readRepoFile(name string) ([]byte, error) {
	if repoFS == nil {
		return nil, fmt.Errorf("the repository root %s could not be opened", repoRoot)
	}
	return repoFS.ReadFile(name)
}

// readRepoDir lists a directory relative to the repository root.
func readRepoDir(name string) ([]os.DirEntry, error) {
	if repoFS == nil {
		return nil, fmt.Errorf("the repository root %s could not be opened", repoRoot)
	}
	dir, err := repoFS.Open(name)
	if err != nil {
		return nil, err
	}
	defer dir.Close()
	return dir.ReadDir(-1)
}

// findRepoRoot walks up from the working directory to the directory that
// holds the go.mod.
func findRepoRoot() string {
	dir, err := os.Getwd()
	if err != nil {
		return "."
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "."
		}
		dir = parent
	}
}

// commandFlags reads the flags every build of the command declares.
func commandFlags() (map[string]map[string]flagDef, error) {
	builds := map[string]map[string]flagDef{}
	for build, files := range buildFiles {
		flags := map[string]flagDef{}
		for _, file := range files {
			defs, err := flagsInFile(filepath.Join(repoRoot, file))
			if err != nil {
				return nil, err
			}
			for name, def := range defs {
				flags[name] = def
			}
		}
		builds[build] = flags
	}
	return builds, nil
}

// flagsInFile reads the flag registrations of one file.
func flagsInFile(path string) (map[string]flagDef, error) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, path, nil, 0)
	if err != nil {
		return nil, err
	}

	flags := map[string]flagDef{}
	for _, decl := range file.Decls {
		gen, ok := decl.(*ast.GenDecl)
		if !ok || gen.Tok != token.VAR {
			continue
		}
		for _, spec := range gen.Specs {
			value, ok := spec.(*ast.ValueSpec)
			if !ok || len(value.Values) == 0 {
				continue
			}
			call, ok := value.Values[0].(*ast.CallExpr)
			if !ok {
				continue
			}
			sel, ok := call.Fun.(*ast.SelectorExpr)
			if !ok || sel.Sel == nil || sel.Sel.Name != "String" &&
				sel.Sel.Name != "Int" && sel.Sel.Name != "Int64" &&
				sel.Sel.Name != "Bool" && sel.Sel.Name != "Float64" &&
				sel.Sel.Name != "Uint" {
				continue
			}
			if len(call.Args) < 2 {
				continue
			}
			nameLit, ok := call.Args[0].(*ast.BasicLit)
			if !ok {
				continue
			}
			name := strings.Trim(nameLit.Value, "\"")
			def := flagDef{name: name, kind: sel.Sel.Name}
			if len(call.Args) >= 2 {
				def.deflt = literalText(call.Args[1])
			}
			flags[name] = def
		}
	}
	return flags, nil
}

// literalText returns the text of a literal, including a negative one.
func literalText(expr ast.Expr) string {
	switch e := expr.(type) {
	case *ast.BasicLit:
		return strings.Trim(e.Value, "\"")
	case *ast.UnaryExpr:
		if e.Op == token.SUB {
			return "-" + literalText(e.X)
		}
	}
	return ""
}

// chartFlagRe matches a flag the chart passes to the command.
var chartFlagRe = regexp.MustCompile(`^\s*-\s*"(-[a-z0-9-]+)"\s*$`)

// chartFlags reads the flags the chart passes.
func chartFlags() ([]string, error) {
	raw, err := readRepoFile(chartDeployment)
	if err != nil {
		return nil, err
	}

	var flags []string
	for _, line := range strings.Split(string(raw), "\n") {
		match := chartFlagRe.FindStringSubmatch(line)
		if match != nil {
			flags = append(flags, strings.TrimPrefix(match[1], "-"))
		}
	}
	return flags, nil
}

// checkChartFlags verifies that every flag the chart passes exists.
func checkChartFlags(flags map[string]map[string]flagDef, report func(string, ...interface{}), verbose bool) int {
	problems := 0

	chart, err := chartFlags()
	if err != nil {
		report("failed to read %s: %v", chartDeployment, err)
		return 1
	}
	if len(chart) == 0 {
		report("no flags were found in %s", chartDeployment)
		return 1
	}

	// The chart deploys the CUDA image.
	cuda := flags["cuda"]
	for _, name := range chart {
		if _, ok := cuda[name]; !ok {
			report("the Helm chart passes -%s, but the CUDA build of the command has no such flag", name)
			problems++
			continue
		}
		if verbose {
			fmt.Printf("the chart's -%s is a flag of the CUDA build\n", name)
		}
	}

	// A flag that only some builds have is a trap: the chart would work
	// with the CUDA image and fail with another one.
	for name := range cuda {
		inChart := false
		for _, chartFlag := range chart {
			if chartFlag == name {
				inChart = true
				break
			}
		}
		if !inChart && verbose {
			fmt.Printf("the CUDA build's -%s is not in the chart\n", name)
		}
	}

	return problems
}

// chartValues reads the scalar values of the chart.
func chartValues() (map[string]string, error) {
	raw, err := readRepoFile(chartValuesPath)
	if err != nil {
		return nil, err
	}

	values := map[string]string{}
	// The stack of the indentation levels and their keys.
	type level struct {
		indent int
		key    string
	}
	var stack []level

	for _, line := range strings.Split(string(raw), "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), "#") {
			continue
		}
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			continue
		}
		indent := len(line) - len(strings.TrimLeft(line, " "))
		if strings.HasPrefix(trimmed, "- ") {
			// A list item.
			continue
		}
		colon := strings.Index(trimmed, ":")
		if colon < 0 {
			continue
		}
		key := strings.TrimSpace(trimmed[:colon])
		value := strings.TrimSpace(trimmed[colon+1:])
		if value == "" || value == "{}" || value == "[]" {
			// A section or an empty collection.
			for len(stack) > 0 && stack[len(stack)-1].indent >= indent {
				stack = stack[:len(stack)-1]
			}
			stack = append(stack, level{indent: indent, key: key})
			continue
		}
		value = strings.Trim(value, "\"")

		for len(stack) > 0 && stack[len(stack)-1].indent >= indent {
			stack = stack[:len(stack)-1]
		}
		parts := make([]string, 0, len(stack)+1)
		for _, l := range stack {
			parts = append(parts, l.key)
		}
		parts = append(parts, key)
		values[strings.Join(parts, ".")] = value
	}
	return values, nil
}

// chartFlagValue maps a value in the chart to the flag it feeds.
var chartFlagValue = []struct {
	path string
	flag string
}{
	{"engine.ngl", "ngl"},
	{"engine.maxBatchSize", "max-batch-size"},
	{"engine.kvCache.blockSize", "block-size"},
	{"engine.kvCache.totalBlocks", "total-blocks"},
	{"engine.temperature", "temp"},
	{"engine.topK", "topk"},
	{"engine.topP", "topp"},
	{"engine.minP", "min-p"},
	{"engine.repPenalty", "rep-penalty"},
	{"engine.speculative.draftLength", "draft-length"},
	{"engine.speculative.draftPaths", "draft-paths"},
}

// checkChartValues verifies that the chart's values match the code's
// defaults.
func checkChartValues(flags map[string]map[string]flagDef, report func(string, ...interface{}), verbose bool) int {
	problems := 0

	values, err := chartValues()
	if err != nil {
		report("failed to read %s: %v", chartValuesPath, err)
		return 1
	}

	cuda := flags["cuda"]
	for _, mapping := range chartFlagValue {
		value, ok := values[mapping.path]
		if !ok {
			report("the chart has no value for %s", mapping.path)
			problems++
			continue
		}
		def, ok := cuda[mapping.flag]
		if !ok {
			report("the chart sets %s, which feeds -%s, and the CUDA build has no such flag",
				mapping.path, mapping.flag)
			problems++
			continue
		}
		if !sameNumber(value, def.deflt) {
			report("the chart sets %s to %q, but the default of -%s is %q",
				mapping.path, value, mapping.flag, def.deflt)
			problems++
			continue
		}
		if verbose {
			fmt.Printf("the chart's %s matches the default of -%s\n", mapping.path, mapping.flag)
		}
	}

	// The API address is built from the service port.
	port, ok := values["service.port"]
	if !ok {
		report("the chart has no service.port")
		problems++
	} else if def, ok := cuda["api-addr"]; ok {
		if !strings.HasSuffix(def.deflt, port) {
			report("the chart serves the API on port %s, but the default of -api-addr is %q",
				port, def.deflt)
			problems++
		}
	}

	// The metrics port has to differ from the API port: they are two
	// listeners, and the API does not serve /metrics.
	metricsPort, ok := values["service.metricsPort"]
	if !ok {
		report("the chart has no service.metricsPort")
		problems++
	} else if metricsPort == port {
		report("the chart serves the API and the metrics on the same port %s; they are separate listeners", port)
		problems++
	} else if def, ok := cuda["metrics"]; ok {
		if !strings.HasSuffix(def.deflt, metricsPort) {
			report("the chart serves the metrics on port %s, but the default of -metrics is %q",
				metricsPort, def.deflt)
			problems++
		}
	}

	return problems
}

// sameNumber compares two numbers written differently.
func sameNumber(a, b string) bool {
	if a == b {
		return true
	}
	af, aerr := strconv.ParseFloat(a, 64)
	bf, berr := strconv.ParseFloat(b, 64)
	if aerr != nil || berr != nil {
		return false
	}
	return af == bf
}

// metricDeclRe matches a metric registration.
var metricDeclRe = regexp.MustCompile(`promauto\.New(Counter|CounterVec|Gauge|GaugeVec|Histogram|HistogramVec|Summary|SummaryVec)\(`)

// declaredMetrics reads the metrics the code declares, with their types.
func declaredMetrics() (map[string]metricDef, error) {
	metrics := map[string]metricDef{}

	entries, err := readRepoDir(metricsDir)
	if err != nil {
		return nil, err
	}
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		raw, err := readRepoFile(filepath.Join(metricsDir, name))
		if err != nil {
			return nil, err
		}
		source := string(raw)

		// Each registration is a "promauto.New<Kind>(" with a Name in it.
		for _, match := range metricDeclRe.FindAllStringIndex(source, -1) {
			start := match[0]
			end := match[1]
			kind := strings.TrimSuffix(strings.TrimPrefix(source[start:end], "promauto.New"), "(")
			kind = strings.TrimSuffix(kind, "Vec")

			// The name is in the registration.
			body := source[end:]
			if len(body) > 400 {
				body = body[:400]
			}
			nameMatch := regexp.MustCompile(`Name:\s*"([a-zA-Z0-9_:]+)"`).FindStringSubmatch(body)
			if nameMatch == nil {
				continue
			}
			metrics[nameMatch[1]] = metricDef{name: nameMatch[1], kind: kind}
		}
	}
	return metrics, nil
}

// promqlWords are the PromQL keywords, not metric names.
var promqlWords = map[string]bool{
	"rate": true, "increase": true, "sum": true, "avg": true, "max": true,
	"min": true, "count": true, "histogram_quantile": true, "le": true,
	"by": true, "without": true, "irate": true, "absent": true, "topk": true,
	"clamp": true, "stddev": true, "stdvar": true, "quantile": true,
	"delta": true, "idelta": true, "group": true, "offset": true,
	"on": true, "ignoring": true, "bool": true,
}

// checkDashboard verifies the Grafana dashboard against the metrics.
func checkDashboard(metrics map[string]metricDef, report func(string, ...interface{}), verbose bool) int {
	problems := 0

	raw, err := readRepoFile(dashboardPath)
	if err != nil {
		report("failed to read %s: %v", dashboardPath, err)
		return 1
	}

	var dashboard struct {
		Panels []struct {
			Title   string `json:"title"`
			Targets []struct {
				Expr string `json:"expr"`
			} `json:"targets"`
		} `json:"panels"`
	}
	if err := json.Unmarshal(raw, &dashboard); err != nil {
		report("failed to parse %s: %v", dashboardPath, err)
		return 1
	}

	queried := map[string]bool{}
	for _, panel := range dashboard.Panels {
		for _, target := range panel.Targets {
			for _, name := range metricNames(target.Expr) {
				def, ok := metrics[name]
				if !ok {
					report("the panel %q queries %q, which is not a declared metric", panel.Title, name)
					problems++
					continue
				}
				queried[name] = true

				// The query has to match the type: a histogram read raw
				// shows the bucket counts, and a gauge read with a rate
				// shows nothing.
				usesBucket := strings.Contains(target.Expr, "_bucket")
				usesRate := strings.Contains(target.Expr, "rate(") || strings.Contains(target.Expr, "increase(")
				switch def.kind {
				case "Histogram":
					if !usesBucket {
						report("the panel %q reads the histogram %q without the bucket suffix", panel.Title, name)
						problems++
					}
				case "Gauge":
					if usesBucket || usesRate {
						report("the panel %q reads the gauge %q as if it were a counter or a histogram", panel.Title, name)
						problems++
					}
				}
			}
		}
	}

	// Every metric the code declares has to be on the dashboard, or it has
	// to be on the list of the ones that deliberately are not.
	notOnDashboard := loadNotOnDashboard()
	var missing []string
	for name := range metrics {
		if !queried[name] {
			if notOnDashboard[name] {
				continue
			}
			missing = append(missing, name)
		}
	}
	sort.Strings(missing)
	for _, name := range missing {
		report("the metric %q is declared but is not on the Grafana dashboard; add a panel for it or list it in scripts/dashboard_allowlist.txt", name)
		problems++
	}

	if verbose {
		fmt.Printf("the dashboard queries %d of the %d declared metrics\n", len(queried), len(metrics))
	}
	return problems
}

// metricNames returns the metric names an expression refers to.
func metricNames(expr string) []string {
	var names []string
	for _, word := range regexp.MustCompile(`\b([a-z][a-z0-9_]*)\b`).FindAllStringSubmatch(expr, -1) {
		name := word[1]
		if promqlWords[name] {
			continue
		}
		for _, suffix := range []string{"_bucket", "_sum", "_count"} {
			if strings.HasSuffix(name, suffix) {
				name = strings.TrimSuffix(name, suffix)
				break
			}
		}
		names = append(names, name)
	}
	return names
}

// loadNotOnDashboard reads the list of the metrics that are deliberately
// not on the dashboard.
func loadNotOnDashboard() map[string]bool {
	return loadAllowed("scripts/dashboard_allowlist.txt")
}

// loadAllowed reads a list of names, one per line, with the comments
// and the blank lines removed.
func loadAllowed(path string) map[string]bool {
	allowed := map[string]bool{}
	raw, err := readRepoFile(path)
	if err != nil {
		return allowed
	}
	for _, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		allowed[line] = true
	}
	return allowed
}

// flagsUnder reads the flag registrations of every Go file under the
// given directories.
func flagsUnder(root string, dirs []string) (map[string]flagDef, error) {
	flags := map[string]flagDef{}
	for _, dir := range dirs {
		err := filepath.Walk(filepath.Join(root, dir), func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return nil
			}
			if info.IsDir() || !strings.HasSuffix(path, ".go") {
				return nil
			}
			// The command line builds are read separately, with their
			// defaults.
			if strings.HasPrefix(path, filepath.Join(root, commandDir)) {
				return nil
			}
			found, err := flagsInFile(path)
			if err != nil {
				return nil
			}
			for name, def := range found {
				if _, ok := flags[name]; !ok {
					flags[name] = def
				}
			}
			return nil
		})
		if err != nil {
			return nil, err
		}
	}
	return flags, nil
}

// routeRe matches a route the API server registers.
var routeRe = regexp.MustCompile(`mux\.Handle(?:Func)?\("([^"]+)"`)

// apiServerRoutes reads the routes the server registers: the API listener
// and the metrics listener.
func apiServerRoutes() (map[string]bool, error) {
	routes := map[string]bool{}

	raw, err := readRepoFile("internal/api/server.go")
	if err != nil {
		return nil, err
	}
	for _, match := range routeRe.FindAllStringSubmatch(string(raw), -1) {
		routes[match[1]] = true
	}

	// The metrics listener is set up in the command, not in the API
	// package.
	entries, err := readRepoDir(commandDir)
	if err != nil {
		return nil, err
	}
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".go") {
			continue
		}
		raw, err := readRepoFile(filepath.Join(commandDir, entry.Name()))
		if err != nil {
			return nil, err
		}
		for _, match := range routeRe.FindAllStringSubmatch(string(raw), -1) {
			routes[match[1]] = true
		}
	}
	return routes, nil
}

// docEndpointRe matches an endpoint the documentation lists.
var docEndpointRe = regexp.MustCompile(`(?m)(?:^#{2,4} (?:GET|POST|PUT|DELETE) |^\| (?:GET|POST|PUT|DELETE) )(/[a-z0-9/_-]*)`)

// checkDocumentedEndpoints verifies that the endpoints the API reference
// lists are the ones the server registers.
//
// The reference used to describe a set of endpoints that never existed --
// /generate, /stream, /ws, /models, /hotswap -- and it put /metrics on
// the API port, where nothing serves it.
func checkDocumentedEndpoints(report func(string, ...interface{}), verbose bool) int {
	problems := 0

	routes, err := apiServerRoutes()
	if err != nil {
		report("failed to read the API server: %v", err)
		return 1
	}
	if len(routes) == 0 {
		report("no routes were found in the API server")
		return 1
	}

	raw, err := readRepoFile("docs/api.md")
	if err != nil {
		report("failed to read docs/api.md: %v", err)
		return 1
	}

	document := string(raw)
	// The section that lists the endpoints which do not exist is not a
	// claim about the server.
	if start := strings.Index(document, "## The endpoints that do not exist"); start >= 0 {
		if end := strings.Index(document[start:], "\n## "); end > 0 {
			document = document[:start] + document[start+end:]
		}
	}

	documented := map[string]bool{}
	for _, match := range docEndpointRe.FindAllStringSubmatch(document, -1) {
		path := match[1]
		documented[path] = true
		if !routes[path] {
			report("docs/api.md documents %s, but the API server does not register it", path)
			problems++
		}
	}
	if len(documented) == 0 {
		report("no endpoints were found in docs/api.md")
		problems++
	}

	// Every route the server registers has to be in the reference.
	for path := range routes {
		if !documented[path] {
			report("the API server registers %s, but docs/api.md does not mention it", path)
			problems++
		}
	}

	// The Helm chart's probes have to be routes the server registers.
	values, err := chartValues()
	if err == nil {
		for _, path := range []string{"probe.livenessPath", "probe.readinessPath"} {
			value, ok := values[path]
			if !ok {
				continue
			}
			if !routes[value] {
				report("the chart's %s is %s, which the API server does not register", path, value)
				problems++
			}
		}
	}

	if verbose {
		fmt.Printf("the API server registers %d routes and the reference documents %d\n",
			len(routes), len(documented))
	}
	return problems
}

// metricCountRe matches a claim about how many metrics there are.
var metricCountRe = regexp.MustCompile(`(?i)(\d+)\s+(?:prometheus\s+)?metrics`)

// checkTheClaimedMetricCount verifies the number of metrics the documents
// claim.
//
// The comparison table said 191 metrics, and the real number was 133.
func checkTheClaimedMetricCount(metrics map[string]metricDef, report func(string, ...interface{}), verbose bool) int {
	problems := 0

	// The development logs record the numbers the documents used to
	// claim, so a count in them is history rather than a claim.
	docs := []string{
		"README.md",
		"docs/comparison.md",
		"docs/features.md",
		"docs/usage.md",
		"docs/api.md",
	}

	for _, doc := range docs {
		raw, err := readRepoFile(doc)
		if err != nil {
			report("failed to read %s: %v", doc, err)
			problems++
			continue
		}
		for _, match := range metricCountRe.FindAllStringSubmatch(string(raw), -1) {
			claimed, err := strconv.Atoi(match[1])
			if err != nil {
				continue
			}
			// A number that is nowhere near the real one is a count of
			// something else, and not a claim about the metrics.
			if claimed < len(metrics)/2 || claimed > len(metrics)*2 {
				continue
			}
			if claimed != len(metrics) {
				report("%s claims %d metrics, but the code declares %d",
					filepath.Base(doc), claimed, len(metrics))
				problems++
				continue
			}
			if verbose {
				fmt.Printf("%s claims the right number of metrics\n", filepath.Base(doc))
			}
		}
	}

	return problems
}

// checkTheDocumentIndex verifies that every document is in the index.
//
// The README links to the index, and the index used to be a directory
// listing rather than a document, so a new document could be forgotten
// without anything noticing.
func checkTheDocumentIndex(report func(string, ...interface{}), verbose bool) int {
	problems := 0

	entries, err := readRepoDir("docs")
	if err != nil {
		report("failed to list the documentation: %v", err)
		return 1
	}

	index, err := readRepoFile("docs/README.md")
	if err != nil {
		report("docs/README.md is missing: the README links to it as the documentation index", err)
		return 1
	}

	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".md") {
			continue
		}
		if !strings.Contains(string(index), entry.Name()) {
			report("docs/%s is not in the documentation index", entry.Name())
			problems++
			continue
		}
		if verbose {
			fmt.Printf("docs/%s is in the index\n", entry.Name())
		}
	}

	return problems
}

// moduleRe matches a module requirement in a go.mod.
var moduleRe = regexp.MustCompile(`^\s*([a-zA-Z0-9./_-]+)\s+(v[0-9][^\s/]*)`)

// replaceRe matches a replace directive in a go.mod.
var replaceRe = regexp.MustCompile(`^replace\s+([a-zA-Z0-9./_-]+)\s+=>\s+([a-zA-Z0-9./_-]+)\s+(v[0-9][^\s/]*)`)

// checkPinnedVersions verifies that the pinned versions in
// docs/dependencies.md are the ones the module files hold.
//
// The document used to carry versions that were stale against go.mod, and
// the emlgo one named the fork's version where go.mod carries a
// placeholder and a replace directive.
func checkPinnedVersions(report func(string, ...interface{}), verbose bool) int {
	problems := 0

	raw, err := readRepoFile("docs/dependencies.md")
	if err != nil {
		report("failed to read docs/dependencies.md: %v", err)
		return 1
	}
	document := string(raw)

	modules, replaces, err := moduleVersions("go.mod")
	if err != nil {
		report("failed to read go.mod: %v", err)
		return 1
	}

	for module, version := range modules {
		if !strings.Contains(document, module) {
			continue
		}
		// The emlgo module is required under a placeholder and
		// replaced by the fork, so the version in the document is the
		// fork's.
		if replacement, ok := replaces[module]; ok {
			version = replacement
		}
		if !strings.Contains(document, version) {
			report("docs/dependencies.md does not pin %s at %s, which is what go.mod has",
				module, version)
			problems++
			continue
		}
		if verbose {
			fmt.Printf("%s is pinned at %s\n", module, version)
		}
	}

	return problems
}

// moduleVersions reads the requirements and the replace directives of a
// go.mod.
func moduleVersions(path string) (map[string]string, map[string]string, error) {
	raw, err := readRepoFile(path)
	if err != nil {
		return nil, nil, err
	}

	modules := map[string]string{}
	replaces := map[string]string{}
	inRequire := false

	for _, line := range strings.Split(string(raw), "\n") {
		trimmed := strings.TrimSpace(line)
		if trimmed == "require (" {
			inRequire = true
			continue
		}
		if trimmed == ")" {
			inRequire = false
			continue
		}
		if strings.HasPrefix(trimmed, "replace ") {
			if match := replaceRe.FindStringSubmatch(trimmed); match != nil {
				replaces[match[1]] = match[3]
			}
			continue
		}
		if !inRequire {
			continue
		}
		if match := moduleRe.FindStringSubmatch(trimmed); match != nil {
			modules[match[1]] = match[2]
		}
	}
	return modules, replaces, nil
}

// docFlagRe matches a flag mentioned in the documentation.
var docFlagRe = regexp.MustCompile("(?:^|[\\s`\"(])-([a-z][a-z0-9-]+)")

// checkDocumentedFlags verifies that the flags the reference documentation
// mentions exist.
func checkDocumentedFlags(flags map[string]map[string]flagDef, report func(string, ...interface{}), verbose bool) int {
	problems := 0

	docs, err := filepath.Glob(filepath.Join(repoRoot, "docs", "*.md"))
	if err != nil {
		report("failed to list the documentation: %v", err)
		return 1
	}

	known := map[string]bool{}
	for _, build := range flags {
		for name := range build {
			known[name] = true
		}
	}
	// The documents describe every command in the repository, not only
	// the server.
	otherFlags, err := flagsUnder(repoRoot, commandDirs)
	if err != nil {
		report("failed to read the commands: %v", err)
		return 1
	}
	for name := range otherFlags {
		known[name] = true
	}
	allowed := loadAllowed("scripts/docsync_allowlist.txt")

	for _, doc := range docs {
		raw, err := readRepoFile(strings.TrimPrefix(doc, repoRoot+string(filepath.Separator)))
		if err != nil {
			report("failed to read %s: %v", doc, err)
			problems++
			continue
		}
		seen := map[string]bool{}
		for _, match := range docFlagRe.FindAllStringSubmatch(string(raw), -1) {
			name := match[1]
			if seen[name] || known[name] || allowed[name] {
				continue
			}
			seen[name] = true
			report("%s mentions -%s, which no build of the command declares", filepath.Base(doc), name)
			problems++
		}
	}

	if verbose {
		fmt.Printf("checked %d documentation files against %d flags\n", len(docs), len(known))
	}
	return problems
}
