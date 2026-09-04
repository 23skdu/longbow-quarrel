package simd

import (
	"os"
	"os/exec"
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
	switch runtime.GOOS {
	case "linux":
		return readCPUFlagsLinux()
	case "darwin":
		return readCPUFlagsDarwin()
	default:
		return ""
	}
}

func readCPUFlagsLinux() string {
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

func readCPUFlagsDarwin() string {
	cmd := exec.Command("sysctl", "-a")
	out, err := cmd.Output()
	if err != nil {
		cmd = exec.Command("sysctl", "hw.optional")
		out, err = cmd.Output()
		if err != nil {
			return ""
		}
	}
	return parseDarwinSysctl(string(out))
}

func parseDarwinSysctl(data string) string {
	var sb strings.Builder
	lines := strings.Split(data, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		idx := strings.IndexAny(line, ":=")
		if idx < 0 {
			continue
		}
		key := strings.TrimSpace(line[:idx])
		val := strings.TrimSpace(line[idx+1:])
		if val != "1" {
			continue
		}
		switch strings.ToLower(key) {
		case "hw.optional.avx2_0":
			sb.WriteString("avx2 ")
		case "hw.optional.avx512f":
			sb.WriteString("avx512f ")
		case "hw.optional.avx512bw":
			sb.WriteString("avx512bw ")
		case "hw.optional.avx512dq":
			sb.WriteString("avx512dq ")
		case "hw.optional.avx512vl":
			sb.WriteString("avx512vl ")
		case "hw.optional.fma":
			sb.WriteString("fma ")
		case "hw.optional.neon", "hw.optional.arm64":
			sb.WriteString("neon ")
		case "hw.optional.arm.feat_dotprod", "hw.optional.armfe_dotprod":
			sb.WriteString("asimddp ")
		case "hw.optional.vaes":
			sb.WriteString("vaes ")
		case "hw.optional.vpclmulqdq":
			sb.WriteString("vpclmulqdq ")
		case "hw.optional.gfni":
			sb.WriteString("gfni ")
		case "hw.optional.avxvnni":
			sb.WriteString("avx_vnni ")
		}
	}
	return sb.String()
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
