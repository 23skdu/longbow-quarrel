//go:build (cuda && !(linux && amd64 && cgo)) || (metal && !darwin) || (tpu && !linux)

package main

import (
	"fmt"
	"os"
)

// This stub only compiles when a GPU backend tag was requested that cannot be
// built for the current host or architecture -- for example `go build -tags
// cuda` on darwin, or `-tags cuda` without cgo on linux.
//
// The default build (no backend tag) selects main_cpu.go, which runs the
// pure-CPU engine. Each supported backend has its own main:
//
//	main.go       linux && amd64 && cuda && cgo   -> CUDA
//	main_metal.go darwin && metal                 -> Apple Metal
//	main_tpu.go   linux && tpu                    -> Google TPU
//	main_cpu.go   no backend tag                  -> CPU (default)
//
// The constraints below are the exact negation of those four main files, so
// at most one main is ever compiled into a given build.
func main() {
	fmt.Println("Longbow-Quarrel: the requested backend cannot be built for this platform.")
	fmt.Println("  CPU   (default):              go build ./cmd/quarrel")
	fmt.Println("  Metal (macOS):                 go build -tags metal ./cmd/quarrel")
	fmt.Println("  CUDA  (Linux x86-64, cgo):    go build -tags cuda ./cmd/quarrel")
	fmt.Println("  TPU   (Linux):                go build -tags tpu ./cmd/quarrel")
	os.Exit(1)
}
