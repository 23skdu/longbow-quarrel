//go:build !cuda && !metal
// +build !cuda,!metal

package main

import (
	"fmt"
	"os"
)

func main() {
	fmt.Println("Longbow-Quarrel: No GPU backend (CUDA or Metal) detected.")
	fmt.Println("To build with Metal support (MacOS): go build -tags metal ./cmd/quarrel")
	fmt.Println("To build with CUDA support (Linux): go build -tags cuda ./cmd/quarrel")
	os.Exit(1)
}
