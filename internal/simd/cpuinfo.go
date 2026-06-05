package simd

import (
	"os"
	"runtime"
	"strings"
)

var (
	hasAVX2      bool
	hasAVX512    bool
	hasAVXVNNI   bool
	hasFMA       bool
	hasGFNI      bool
	hasVAES      bool
	hasVPCLMULQDQ bool
	cpuInitDone  bool
	cpuLevel     int
)

const (
	CPULevelScalar = 0
	CPULevelAVX2   = 1
	CPULevelAVX512 = 2
)

func detectCPU() {
	if cpuInitDone {
		return
	}
	cpuInitDone = true

	cpuLevel = CPULevelScalar

	if os.Getenv("DISABLE_SIMD") == "1" {
		return
	}

	flags := readCPUFlags()

	hasAVX2 = containsFlag(flags, "avx2")
	hasAVX512 = containsFlag(flags, "avx512f") && containsFlag(flags, "avx512bw")
	hasAVXVNNI = containsFlag(flags, "avx_vnni")
	hasFMA = containsFlag(flags, "fma")
	hasGFNI = containsFlag(flags, "gfni")
	hasVAES = containsFlag(flags, "vaes")
	hasVPCLMULQDQ = containsFlag(flags, "vpclmulqdq")

	if os.Getenv("DISABLE_AVX512") == "1" {
		hasAVX512 = false
	}
	if os.Getenv("DISABLE_AVX2") == "1" {
		hasAVX2 = false
		hasAVX512 = false
	}

	if hasAVX512 {
		cpuLevel = CPULevelAVX512
	} else if hasAVX2 {
		cpuLevel = CPULevelAVX2
	}
}

func readCPUFlags() string {
	if runtime.GOOS != "linux" {
		return ""
	}
	data, err := os.ReadFile("/proc/cpuinfo")
	if err != nil {
		return ""
	}
	for _, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, "flags") || strings.HasPrefix(line, "Features") {
			idx := strings.IndexByte(line, ':')
			if idx >= 0 {
				return line[idx+1:]
			}
		}
	}
	return ""
}

func containsFlag(flags, flag string) bool {
	if flags == "" {
		return false
	}
	idx := strings.Index(flags, flag)
	if idx < 0 {
		return false
	}
	// Ensure word boundary
	before := idx == 0 || flags[idx-1] == ' ' || flags[idx-1] == '\t'
	after := idx+len(flag) >= len(flags) || flags[idx+len(flag)] == ' ' || flags[idx+len(flag)] == '\t' || flags[idx+len(flag)] == '\n'
	return before && after
}

func GetCPULevel() int {
	if !cpuInitDone {
		detectCPU()
	}
	return cpuLevel
}
