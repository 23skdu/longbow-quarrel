// Command vacuous_tests reports Go test functions that cannot fail.
//
// A test cannot fail when its body (and every t.Run subtest it spawns)
// contains no assertion: no t.Error, no t.Fatal, no require/assert call
// and no helper that asserts. Such a test is a test in name only: it
// passes whatever the code under test does.
//
// The list of known offenders lives in scripts/vacuous_tests_allowlist.txt.
// CI fails when a test that is not on that list cannot fail, so the debt can
// only shrink.
//
// What counts as an assertion:
//
//   - t.Error, t.Errorf, t.Fatal, t.Fatalf, t.Fail, t.FailNow
//   - a require.* or assert.* call
//   - a call to a helper that takes the *testing.T and asserts itself,
//     in this package or in another one
//
// What does not:
//
//   - t.Skip, t.Skipf, t.SkipNow. A test that only skips is not a test
//     that passes, it is a test that did not run.
//   - a helper that only logs.
//
// The helper bodies are resolved transitively: a test that calls a helper
// which calls another helper which asserts is not vacuous. The helper map
// is built from every Go file in the module, so a helper in a test-only
// package (internal/promtest) counts as well.
//
// Usage:
//
//	vacuous_tests [-dir DIR] [-allowlist FILE] [-update]
package main

import (
	"bufio"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// assertNames are *testing.T methods that make a test fail.
var assertNames = map[string]bool{
	"Error":   true,
	"Errorf":  true,
	"Fatal":   true,
	"Fatalf":  true,
	"Fail":    true,
	"FailNow": true,
}

// skipNames are *testing.T methods that stop a test without failing it.
// A test whose only action is a skip did not run.
var skipNames = map[string]bool{
	"Skip":    true,
	"Skipf":   true,
	"SkipNow": true,
}

// helperOnlyNames are *testing.T methods that neither assert nor prove anything
// on their own; a test that only calls these is still vacuous.
var helperOnlyNames = map[string]bool{
	"Helper":   true,
	"Cleanup":  true,
	"Setenv":   true,
	"TempDir":  true,
	"Deadline": true,
	"Name":     true,
	"Parallel": true,
}

// modulePath is the import path of the module. It maps an import path
// onto the directory that holds it.
const modulePath = "github.com/23skdu/longbow-quarrel"

// packageInfo is what is known about one package in the module.
type packageInfo struct {
	// helpers maps a function name onto its declaration. It holds the
	// functions whose first parameter is a testing handle.
	helpers map[string]*ast.FuncDecl
	// imports maps an import alias onto its import path.
	imports map[string]string
	// interfaces holds the interfaces the package declares, by name. A
	// helper that takes one of them takes a testing handle.
	interfaces map[string]*ast.InterfaceType
}

// helperSignature reports whether the function takes a testing handle as
// its first parameter: a *testing.T, a testing.TB, or an interface with
// Errorf or Fatalf (which is how a test-only helper package spells it).
func helperSignature(fn *ast.FuncDecl, interfaces map[string]*ast.InterfaceType) bool {
	if fn.Type.Params == nil || len(fn.Type.Params.List) == 0 {
		return false
	}
	field := fn.Type.Params.List[0]
	return isTestingHandle(field.Type, interfaces)
}

func isTestingHandle(expr ast.Expr, interfaces map[string]*ast.InterfaceType) bool {
	switch t := expr.(type) {
	case *ast.StarExpr:
		// *testing.T or *testing.B
		sel, ok := t.X.(*ast.SelectorExpr)
		if !ok {
			return false
		}
		pkg, ok := sel.X.(*ast.Ident)
		return ok && pkg.Name == "testing" && (sel.Sel.Name == "T" || sel.Sel.Name == "B" || sel.Sel.Name == "TB")
	case *ast.SelectorExpr:
		// testing.TB
		pkg, ok := t.X.(*ast.Ident)
		return ok && pkg.Name == "testing" && t.Sel.Name == "TB"
	case *ast.InterfaceType:
		// An interface with Errorf or Fatalf is a testing handle: that is
		// what internal/promtest declares.
		return interfaceHasAssertion(t)
	case *ast.Ident:
		// A named interface: promtest.TestingT.
		iface, ok := interfaces[t.Name]
		if !ok {
			return false
		}
		return interfaceHasAssertion(iface)
	}
	return false
}

func interfaceHasAssertion(iface *ast.InterfaceType) bool {
	for _, method := range iface.Methods.List {
		if len(method.Names) == 0 {
			continue
		}
		name := method.Names[0].Name
		if name == "Errorf" || name == "Fatalf" || name == "Error" || name == "Fatal" {
			return true
		}
	}
	return false
}

func main() {
	dir := flag.String("dir", ".", "root directory to scan")
	allowlist := flag.String("allowlist", "scripts/vacuous_tests_allowlist.txt", "allowlist file")
	update := flag.Bool("update", false, "rewrite the allowlist instead of checking against it")
	flag.Parse()

	found, err := scan(*dir)
	if err != nil {
		fmt.Fprintln(os.Stderr, "vacuous_tests:", err)
		os.Exit(2)
	}

	if *update {
		if err := writeAllowlist(*allowlist, found); err != nil {
			fmt.Fprintln(os.Stderr, "vacuous_tests:", err)
			os.Exit(2)
		}
		fmt.Printf("vacuous_tests: wrote %d entries to %s\n", len(found), *allowlist)
		return
	}

	allowed, err := readAllowlist(*allowlist)
	if err != nil {
		fmt.Fprintln(os.Stderr, "vacuous_tests:", err)
		os.Exit(2)
	}

	var fresh []string
	for _, key := range found {
		if !allowed[key] {
			fresh = append(fresh, key)
		}
	}
	var stale []string
	for key := range allowed {
		if _, ok := lookup(found, key); !ok {
			stale = append(stale, key)
		}
	}
	sort.Strings(fresh)
	sort.Strings(stale)

	if len(stale) > 0 {
		fmt.Printf("vacuous_tests: %d allowlist entries no longer apply (remove them): %s\n",
			len(stale), strings.Join(stale, ", "))
	}
	if len(fresh) > 0 {
		fmt.Printf("vacuous_tests: %d test(s) cannot fail:\n", len(fresh))
		for _, key := range fresh {
			fmt.Println("  " + key)
		}
		fmt.Println("fix them (add a real assertion) or add them to " + *allowlist)
		os.Exit(1)
	}
	fmt.Printf("vacuous_tests: OK (%d known vacuous tests allowlisted, %d new)\n", len(allowed), 0)
}

func lookup(list []string, key string) (string, bool) {
	for _, k := range list {
		if k == key {
			return k, true
		}
	}
	return "", false
}

func readAllowlist(path string) (map[string]bool, error) {
	out := map[string]bool{}
	f, err := os.Open(path) // #nosec G304 -- path comes from -allowlist flag, not untrusted input
	if os.IsNotExist(err) {
		return out, nil
	}
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		out[line] = true
	}
	return out, sc.Err()
}

func writeAllowlist(path string, keys []string) error {
	sorted := append([]string(nil), keys...)
	sort.Strings(sorted)
	var b strings.Builder
	b.WriteString("# Tests that currently cannot fail. Every entry is debt:\n")
	b.WriteString("# add a real assertion to the test and delete its line here.\n")
	b.WriteString("# Generated by: go run ./scripts/vacuous_tests -update\n")
	for _, k := range sorted {
		b.WriteString(k)
		b.WriteString("\n")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o750); err != nil {
		return err
	}
	return os.WriteFile(path, []byte(b.String()), 0o600) // #nosec G306
}

// scan walks the module, collects the helper functions of every package
// and reports the test functions that cannot fail.
func scan(root string) ([]string, error) {
	fset := token.NewFileSet()

	// The helper map, per package directory.
	packages := map[string]*packageInfo{}

	// The test functions, per file.
	type testFunc struct {
		key string
		fn  *ast.FuncDecl
		dir string
	}
	var tests []testFunc

	// The helper candidates, checked once the whole module is known: a
	// helper may take an interface that another file of its package
	// declares.
	type helperCandidate struct {
		fn  *ast.FuncDecl
		pkg *packageInfo
	}
	var candidates []helperCandidate

	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			name := info.Name()
			if name == ".git" || name == "node_modules" || name == "vendor" || name == "bin" {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") {
			return nil
		}
		file, perr := parser.ParseFile(fset, path, nil, parser.ParseComments)
		if perr != nil {
			return nil // unparseable files are somebody else's problem
		}

		dir := filepath.Dir(path)
		pkg, ok := packages[dir]
		if !ok {
			pkg = &packageInfo{
				helpers:    map[string]*ast.FuncDecl{},
				imports:    map[string]string{},
				interfaces: map[string]*ast.InterfaceType{},
			}
			packages[dir] = pkg
		}
		for _, decl := range file.Decls {
			gen, ok := decl.(*ast.GenDecl)
			if !ok || gen.Tok != token.TYPE {
				continue
			}
			for _, spec := range gen.Specs {
				ts, ok := spec.(*ast.TypeSpec)
				if !ok {
					continue
				}
				iface, ok := ts.Type.(*ast.InterfaceType)
				if !ok {
					continue
				}
				pkg.interfaces[ts.Name.Name] = iface
			}
		}
		for _, imp := range file.Imports {
			path := strings.Trim(imp.Path.Value, `"`)
			alias := ""
			if imp.Name != nil {
				alias = imp.Name.Name
			} else {
				alias = path
				if idx := strings.LastIndex(alias, "/"); idx >= 0 {
					alias = alias[idx+1:]
				}
			}
			pkg.imports[alias] = path
		}

		for _, decl := range file.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Body == nil {
				continue
			}
			candidates = append(candidates, helperCandidate{fn: fn, pkg: pkg})
			if fn.Recv != nil || !strings.HasPrefix(fn.Name.Name, "Test") {
				continue
			}
			if !takesTestingT(fn) {
				continue
			}
			rel, rerr := filepath.Rel(root, path)
			if rerr != nil {
				rel = path
			}
			tests = append(tests, testFunc{
				key: filepath.ToSlash(rel) + ":" + fn.Name.Name,
				fn:  fn,
				dir: dir,
			})
		}
		return nil
	})
	if err != nil {
		return nil, err
	}

	for _, candidate := range candidates {
		if helperSignature(candidate.fn, candidate.pkg.interfaces) {
			candidate.pkg.helpers[candidate.fn.Name.Name] = candidate.fn
		}
	}

	var keys []string
	for _, test := range tests {
		resolver := &resolver{
			root:     root,
			packages: packages,
			current:  test.dir,
			visiting: map[string]bool{},
		}
		if !resolver.asserted(test.fn.Body) {
			keys = append(keys, test.key)
		}
	}
	return keys, nil
}

// resolver answers "can this body fail?" with the helper map in hand.
type resolver struct {
	root     string
	packages map[string]*packageInfo
	current  string
	visiting map[string]bool
}

// asserted reports whether b can make the test fail, either directly or
// through a t.Run subtest that itself can fail.
func (r *resolver) asserted(b *ast.BlockStmt) bool {
	if b == nil {
		return false
	}

	direct := false
	ast.Inspect(b, func(n ast.Node) bool {
		if direct {
			return false
		}
		if call, ok := n.(*ast.CallExpr); ok && isTRun(call) && len(call.Args) == 2 {
			// Subtests are evaluated separately below; skip their bodies here
			// so a t.Run-wrapped assertion is not double counted.
			return false
		}
		if _, ok := n.(*ast.FuncLit); ok {
			// A bare closure (a goroutine body, a deferred func, a callback
			// passed to ParallelFor, ...) can still assert. Descend into it
			// rather than treating it as opaque. Only t.Run closures are
			// excluded, and those are handled above.
			return true
		}
		if r.isAssertCall(n) {
			direct = true
			return false
		}
		return true
	})
	if direct {
		return true
	}

	subtests := false
	ast.Inspect(b, func(n ast.Node) bool {
		if subtests {
			return false
		}
		call, ok := n.(*ast.CallExpr)
		if !ok || !isTRun(call) || len(call.Args) != 2 {
			return true
		}
		lit, ok := call.Args[1].(*ast.FuncLit)
		if !ok {
			return true
		}
		if r.asserted(lit.Body) {
			subtests = true
			return false
		}
		return true
	})
	return subtests
}

// isAssertCall reports whether the call can fail the test.
func (r *resolver) isAssertCall(n ast.Node) bool {
	call, ok := n.(*ast.CallExpr)
	if !ok {
		return false
	}

	// require.X(...), assert.X(...)
	if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
		if id, ok := sel.X.(*ast.Ident); ok && (id.Name == "require" || id.Name == "assert") {
			return true
		}
		// t.Error(...), t.Fatal(...)
		if id, ok := sel.X.(*ast.Ident); ok && id.Name == "t" {
			if assertNames[sel.Sel.Name] {
				return true
			}
			if skipNames[sel.Sel.Name] || helperOnlyNames[sel.Sel.Name] {
				return false
			}
		}
	}

	// A helper that takes the testing handle decides whether the test
	// passes. The helper body is resolved, so a helper that only logs does
	// not count.
	if len(call.Args) > 0 {
		if id, ok := call.Args[0].(*ast.Ident); ok && id.Name == "t" {
			return r.helperAsserts(call)
		}
	}
	return false
}

// helperAsserts resolves the helper a call refers to and reports whether
// its body can fail the test.
func (r *resolver) helperAsserts(call *ast.CallExpr) bool {
	switch fun := call.Fun.(type) {
	case *ast.Ident:
		// A call in this package.
		return r.packageAsserts(r.current, fun.Name)
	case *ast.SelectorExpr:
		// A call in another package: promtest.RequireDelta(t, ...).
		id, ok := fun.X.(*ast.Ident)
		if !ok {
			return false
		}
		pkg, ok := r.packages[r.current]
		if !ok {
			return false
		}
		importPath, ok := pkg.imports[id.Name]
		if !ok {
			return false
		}
		dir := r.dirForImport(importPath)
		if dir == "" {
			return false
		}
		return r.packageAsserts(dir, fun.Sel.Name)
	}
	return false
}

// dirForImport maps an import path onto the directory that holds it.
func (r *resolver) dirForImport(importPath string) string {
	if importPath == modulePath {
		return r.root
	}
	if !strings.HasPrefix(importPath, modulePath+"/") {
		return ""
	}
	rel := strings.TrimPrefix(importPath, modulePath+"/")
	return filepath.Join(r.root, rel)
}

// packageAsserts reports whether the named function of a package can fail
// the test.
func (r *resolver) packageAsserts(dir, name string) bool {
	pkg, ok := r.packages[dir]
	if !ok {
		return false
	}
	fn, ok := pkg.helpers[name]
	if !ok {
		return false
	}

	key := dir + ":" + name
	if r.visiting[key] {
		// A cycle: assume it asserts, so a recursive helper does not
		// make every test that calls it vacuous.
		return true
	}
	r.visiting[key] = true
	defer delete(r.visiting, key)

	return r.asserted(fn.Body)
}

func takesTestingT(fn *ast.FuncDecl) bool {
	if fn.Type.Params == nil || len(fn.Type.Params.List) != 1 {
		return false
	}
	field := fn.Type.Params.List[0]
	if len(field.Names) != 1 || field.Names[0].Name != "t" {
		return false
	}
	star, ok := field.Type.(*ast.StarExpr)
	if !ok {
		return false
	}
	sel, ok := star.X.(*ast.SelectorExpr)
	if !ok {
		return false
	}
	pkg, ok := sel.X.(*ast.Ident)
	return ok && pkg.Name == "testing" && sel.Sel.Name == "T"
}

func isTRun(call *ast.CallExpr) bool {
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok || sel.Sel.Name != "Run" {
		return false
	}
	id, ok := sel.X.(*ast.Ident)
	return ok && id.Name == "t"
}
