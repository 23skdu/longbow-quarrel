package main

import (
	"flag"
	"fmt"
	"os"
)

// Version and BuildTime are populated at link time by the Makefile / Dockerfile
// via -ldflags "-X main.Version=... -X main.BuildTime=...". Without these
// declarations the -X stamps were silently dropped by the linker.
var (
	Version   = "dev"
	BuildTime = "unknown"
)

var (
	helpFlag    = flag.Bool("help", false, "Show usage information and exit")
	versionFlag = flag.Bool("version", false, "Print version information and exit")
)

// handleHelpFlag implements the -help behaviour shared by every backend.
//
// The flag package exits with status 2 when it meets an undefined flag, and
// "--help" is undefined unless a program defines it explicitly. Every
// Dockerfile HEALTHCHECK ran `quarrel --help`, so the container was
// permanently reported unhealthy.
//
// It must be called *before* flag.Parse, because the point is to intercept
// the help request before the parser can reject it. It returns true when the
// process should exit successfully.
func handleHelpFlag() bool {
	for _, a := range os.Args[1:] {
		switch a {
		case "-h", "-help", "--help":
			flag.Usage()
			return true
		}
	}
	return false
}

// handleVersionFlag prints the version and reports whether the process should
// exit. It must be called *after* flag.Parse so the flag value is populated.
func handleVersionFlag() bool {
	if *versionFlag {
		printVersion()
		return true
	}
	if *helpFlag {
		flag.Usage()
		return true
	}
	return false
}

func printVersion() {
	fmt.Printf("longbow-quarrel %s (built %s)\n", Version, BuildTime)
}
