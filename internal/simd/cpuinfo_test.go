package simd

import (
	"os"
	"strings"
	"testing"
)

func TestCPUDetection(t *testing.T) {
	// Reset and re-detect
	cpuInitDone = false
	detectCPU()

	if !cpuInitDone {
		t.Fatal("detectCPU() should mark detection as done")
	}

	if GetCPULevel() < CPULevelAVX2 {
		t.Logf("CPU level: %d (AVX2 not detected, running on non-x86?)", GetCPULevel())
	}
}

func TestContainsFlag(t *testing.T) {
	tests := []struct {
		flags string
		flag  string
		want  bool
	}{
		{"avx2 avx512f fma", "avx2", true},
		{"avx2 avx512f fma", "avx512f", true},
		{"avx2 avx512f fma", "avx3", false},
		{"avx2 avx_vnni fma", "avx_vnni", true},
		{"avx2", "avx2", true},
		{"", "avx2", false},
	}

	for _, tt := range tests {
		if got := containsFlag(tt.flags, tt.flag); got != tt.want {
			t.Errorf("containsFlag(%q, %q) = %v, want %v", tt.flags, tt.flag, got, tt.want)
		}
	}
}

func TestReadCPUFlags(t *testing.T) {
	flags := readCPUFlags()
	if flags == "" {
		t.Skip("not on Linux/Darwin or CPU info not available")
	}
	if !strings.Contains(flags, "avx2") && !strings.Contains(flags, "asimd") && !strings.Contains(flags, "neon") {
		t.Logf("No SIMD flags found: %s", flags[:min(len(flags), 200)])
	}
}

func TestParseDarwinSysctl(t *testing.T) {
	sample := `
hw.optional.avx1_0: 1
hw.optional.avx2_0: 1
hw.optional.avx512f: 1
hw.optional.avx512bw: 1
hw.optional.fma: 1
hw.optional.neon: 0
hw.optional.vaes: 1
`
	flags := parseDarwinSysctl(sample)
	if !containsFlag(flags, "avx2") {
		t.Errorf("expected avx2 in %q", flags)
	}
	if !containsFlag(flags, "avx512f") {
		t.Errorf("expected avx512f in %q", flags)
	}
	if !containsFlag(flags, "avx512bw") {
		t.Errorf("expected avx512bw in %q", flags)
	}
	if !containsFlag(flags, "fma") {
		t.Errorf("expected fma in %q", flags)
	}
	if !containsFlag(flags, "vaes") {
		t.Errorf("expected vaes in %q", flags)
	}
	if containsFlag(flags, "neon") {
		t.Errorf("did not expect neon in %q", flags)
	}
}

func TestParseDarwinSysctl_AppleSilicon(t *testing.T) {
	sample := `
hw.optional.arm64: 1
hw.optional.neon: 1
hw.optional.arm.FEAT_DotProd: 1
hw.optional.avx2_0: 0
`
	flags := parseDarwinSysctl(sample)
	if !containsFlag(flags, "neon") {
		t.Errorf("expected neon in %q", flags)
	}
	if !containsFlag(flags, "asimddp") {
		t.Errorf("expected asimddp in %q", flags)
	}
	if containsFlag(flags, "avx2") {
		t.Errorf("did not expect avx2 in %q", flags)
	}
}

func TestDisableAVX2Env(t *testing.T) {
	os.Setenv("DISABLE_AVX2", "1")
	defer os.Unsetenv("DISABLE_AVX2")

	cpuInitDone = false
	detectCPU()

	if hasAVX2 {
		t.Error("hasAVX2 should be false when DISABLE_AVX2=1")
	}
	if hasAVX512 {
		t.Error("hasAVX512 should be false when DISABLE_AVX2=1")
	}
}

func TestDisableAVX512Env(t *testing.T) {
	os.Setenv("DISABLE_AVX512", "1")
	defer os.Unsetenv("DISABLE_AVX512")

	cpuInitDone = false
	detectCPU()

	if hasAVX512 {
		t.Error("hasAVX512 should be false when DISABLE_AVX512=1")
	}
}
