// Command dupcheck reports package-level identifiers that are declared more
// than once across the files selected for a build configuration.
//
// Duplicate top-level declarations are a compile error ("redeclared in this
// block"), but only when two files declaring the same name land in the same
// package. Because this repo selects files with build tags, a duplicate can
// sit unnoticed in a configuration nobody builds locally -- the
// darwin+metal combination had exactly that problem, and it broke the macOS
// Metal build.
//
// The file list is obtained from `go list -json`, which is what the compiler
// itself uses, so the selection matches a real build exactly. `go list` does
// not need a C compiler, so a darwin or cuda configuration can be checked
// from a Linux host.
//
// Usage:
//
//	dupcheck [-dir <pkg>] [-tags "<tags>"] [-goos <os>] [-goarch <arch>] [-cgo 0|1]
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
)

type goListPackage struct {
	Dir         string
	GoFiles     []string
	CgoFiles    []string
	TestGoFiles []string
	Error       *struct{ Err string }
}

func main() {
	dir := flag.String("dir", "./internal/engine", "package directory to check")
	tags := flag.String("tags", "", "build tags (space separated)")
	goos := flag.String("goos", "linux", "target GOOS")
	goarch := flag.String("goarch", "amd64", "target GOARCH")
	cgo := flag.Bool("cgo", true, "CGO_ENABLED for the simulated build")
	flag.Parse()

	files, err := selectedFiles(*dir, *tags, *goos, *goarch, *cgo)
	if err != nil {
		fmt.Fprintf(os.Stderr, "dupcheck: %v\n", err)
		os.Exit(1)
	}
	if len(files) == 0 {
		fmt.Printf("dupcheck: OK - no files selected in %s (goos=%s goarch=%s cgo=%v tags=%q)\n",
			*dir, *goos, *goarch, *cgo, *tags)
		return
	}

	fset := token.NewFileSet()
	decls := map[string][]string{}
	for _, f := range files {
		collect(fset, f, decls)
	}

	names := make([]string, 0, len(decls))
	for n := range decls {
		names = append(names, n)
	}
	sort.Strings(names)

	dupes := 0
	for _, n := range names {
		if len(decls[n]) < 2 {
			continue
		}
		dupes++
		fmt.Printf("DUPLICATE %s:\n", n)
		for _, d := range decls[n] {
			fmt.Printf("    %s\n", d)
		}
	}
	if dupes == 0 {
		fmt.Printf("dupcheck: OK - no duplicate package-level declarations in %s (goos=%s goarch=%s cgo=%v tags=%q, %d files)\n",
			*dir, *goos, *goarch, *cgo, *tags, len(files))
		return
	}
	os.Exit(1)
}

// selectedFiles returns the files the compiler would include for the given
// configuration.
func selectedFiles(dir, tags, goos, goarch string, cgo bool) ([]string, error) {
	// `go list` treats a bare "scripts/foo" as an import path, so a
	// directory argument must be spelled "./scripts/foo".
	pkgArg := dir
	if !strings.HasPrefix(pkgArg, "./") && !strings.HasPrefix(pkgArg, "/") {
		pkgArg = "./" + pkgArg
	}
	if strings.HasPrefix(pkgArg, "/") {
		// Absolute paths are not valid package patterns; make them relative
		// to the working directory.
		if wd, err := os.Getwd(); err == nil {
			if rel, err := filepath.Rel(wd, pkgArg); err == nil {
				pkgArg = "./" + rel
			}
		}
	}

	args := []string{"list", "-e", "-json"}
	if tags != "" {
		args = append(args, "-tags", tags)
	}
	args = append(args, pkgArg)

	cmd := exec.Command("go", args...)
	cmd.Env = append(os.Environ(),
		"GOOS="+goos,
		"GOARCH="+goarch,
		fmt.Sprintf("CGO_ENABLED=%d", boolToInt(cgo)),
	)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("go list: %w: %s", err, strings.TrimSpace(stderr.String()))
	}

	dec := json.NewDecoder(strings.NewReader(string(out)))
	var files []string
	for {
		var pkg goListPackage
		if err := dec.Decode(&pkg); err != nil {
			break
		}
		if pkg.Error != nil {
			// A package excluded entirely by the build constraints of this
			// configuration ("build constraints exclude all Go files") has
			// nothing to check.
			if strings.Contains(pkg.Error.Err, "exclude all Go files") ||
				strings.Contains(pkg.Error.Err, "no Go files") {
				continue
			}
			return nil, fmt.Errorf("go list: %s", pkg.Error.Err)
		}
		for _, group := range [][]string{pkg.GoFiles, pkg.CgoFiles, pkg.TestGoFiles} {
			for _, f := range group {
				files = append(files, filepath.Join(pkg.Dir, f))
			}
		}
	}
	if len(files) == 0 {
		// Nothing selected in this configuration; not an error.
		return nil, nil
	}
	return files, nil
}

func collect(fset *token.FileSet, path string, decls map[string][]string) {
	f, err := parser.ParseFile(fset, path, nil, 0)
	if err != nil {
		return
	}
	base := filepath.Base(path)
	add := func(name string, pos token.Pos) {
		if name == "_" {
			return
		}
		decls[name] = append(decls[name], fmt.Sprintf("%s:%d", base, fset.Position(pos).Line))
	}
	for _, d := range f.Decls {
		switch decl := d.(type) {
		case *ast.GenDecl:
			for _, spec := range decl.Specs {
				switch s := spec.(type) {
				case *ast.ValueSpec:
					for _, n := range s.Names {
						add(n.Name, n.Pos())
					}
				case *ast.TypeSpec:
					add(s.Name.Name, s.Name.Pos())
				}
			}
		case *ast.FuncDecl:
			// Methods and struct fields have their own scopes; only
			// package-level funcs and vars collide.
			if decl.Recv == nil && decl.Name.Name != "init" {
				add(decl.Name.Name, decl.Name.Pos())
			}
		}
	}
}

func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}
